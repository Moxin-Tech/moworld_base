import os
import warnings
import copy
import threading
import queue

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers.training_args import TrainingArguments

from megatron.training import get_args
from megatron.training.utils import is_rank0
from megatron.core import mpu

from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DataArgumentsForRewardVideo,
    DatasetAttr,
    load_tokenizer,
    align_dataset,
    SupervisedDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.data_utils.video_processor import VideoProcessor
from mindspeed_mm.data.data_utils.video_reader import VideoReader
from mindspeed_mm.data.data_utils.reward_preprocess import build_reward_data

logger = get_logger(__name__)


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank=None):
        self.dataset = dataset
        self.num_dp = mpu.get_data_parallel_world_size()
        self.dp_rank = mpu.get_data_parallel_rank()

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if idx % self.num_dp == self.dp_rank:
                yield item


class AsyncPreprocessIterableDataset(IterableDataset):
    def __init__(self, dataset, preprocess_fn, buffer_size=None, num_workers=None):
        self.dataset = dataset
        self.preprocess_fn = preprocess_fn
        self.buffer_size, self.num_workers = self._resolve_async_config(buffer_size, num_workers)

    @staticmethod
    def _resolve_async_config(buffer_size, num_workers):
        """Normalize async worker settings while preserving the existing defaults."""
        if buffer_size is None and num_workers is None:
            normalized_buffer_size = 8
            normalized_num_workers = max(1, min(normalized_buffer_size, os.cpu_count() or 1))
        elif buffer_size is not None and num_workers is None:
            normalized_buffer_size = max(1, buffer_size)
            normalized_num_workers = max(1, min(normalized_buffer_size, os.cpu_count() or 1))
        elif buffer_size is None and num_workers is not None:
            normalized_num_workers = max(1, num_workers)
            normalized_buffer_size = normalized_num_workers
        else:
            normalized_num_workers = max(1, num_workers)
            normalized_buffer_size = max(1, buffer_size)
            if normalized_buffer_size < normalized_num_workers:
                logger.warning(
                    "[AsyncPreprocessIterableDataset] buffer_size=%s is smaller than num_workers=%s. "
                    "This may cause queue blocking and reduce preprocessing throughput. ",
                    normalized_buffer_size,
                    normalized_num_workers,
                )
            
        return normalized_buffer_size, normalized_num_workers

    def _preprocess_item(self, item):
        batch_dict = {k: [v] for k, v in item.items()}
        processed = self.preprocess_fn(batch_dict)
        if not processed:
            return []

        num_items = len(next(iter(processed.values())))
        return [{k: v[i] for k, v in processed.items()} for i in range(num_items)]

    def __iter__(self):
        task_queue = queue.Queue(maxsize=self.num_workers)
        result_queue = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()
        task_done = object()

        def put_with_stop(target_queue, value):
            while not stop_event.is_set():
                try:
                    target_queue.put(value, timeout=0.1)
                    return True
                except queue.Full:
                    if stop_event.is_set():
                        return False
            return False

        def get_with_stop(source_queue):
            while not stop_event.is_set():
                try:
                    return source_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        return None
            return None

        def producer():
            try:
                # Only one thread consumes the upstream iterable so DP-based sharding stays deterministic.
                for sequence_idx, item in enumerate(self.dataset):
                    if stop_event.is_set():
                        break
                    if not put_with_stop(task_queue, (sequence_idx, item)):
                        return
            except Exception as exc:
                put_with_stop(result_queue, ("error", "Failed to iterate dataset for preprocessing.", exc))
                stop_event.set()
            finally:
                for _ in range(self.num_workers):
                    if not put_with_stop(task_queue, task_done):
                        break

        def worker():
            while not stop_event.is_set():
                task = get_with_stop(task_queue)
                if task is None:
                    return
                if task is task_done:
                    put_with_stop(result_queue, ("done", None, None))
                    return

                sequence_idx, item = task
                try:
                    processed_items = self._preprocess_item(item)
                except Exception as exc:
                    put_with_stop(result_queue, ("error", "Preprocessing failed. Check input data and preprocessing function.", exc))
                    stop_event.set()
                    return

                if not put_with_stop(result_queue, ("result", sequence_idx, processed_items)):
                    return

        producer_thread = threading.Thread(target=producer, daemon=True)
        worker_threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.num_workers)]
        producer_thread.start()
        for thread in worker_threads:
            thread.start()

        try:
            pending_results = {}
            next_sequence_idx = 0
            finished_workers = 0

            while finished_workers < self.num_workers:
                while next_sequence_idx in pending_results:
                    for item in pending_results.pop(next_sequence_idx):
                        yield item
                    next_sequence_idx += 1

                message = get_with_stop(result_queue)
                if message is None:
                    break

                message_type, payload, exc = message
                if message_type == "result":
                    pending_results[payload] = exc
                elif message_type == "done":
                    finished_workers += 1
                else:
                    stop_event.set()
                    raise RuntimeError(payload) from exc

            # Reorder completed tasks before yielding so all ranks in the same DP replica see the same sample order.
            while next_sequence_idx in pending_results:
                for item in pending_results.pop(next_sequence_idx):
                    yield item
                next_sequence_idx += 1
        finally:
            stop_event.set()
            producer_thread.join(timeout=1)
            for thread in worker_threads:
                thread.join(timeout=1)


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    data_args = DataArguments(**basic_param)
    data_args.cutoff_len = get_args().seq_length
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)

    args = get_args()
    consumed_samples = args.consumed_train_samples

    # Ensure main process handles data processing, while other processes reuse cache to avoid redundant calculations.
    # This strategy is consistent with the data processing strategy used by LLaMA Factory.
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # load dataset from file
        train_dataset = load_dataset(path="json", data_files=data_args.dataset, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))

        # Skip consumed samples only in streaming mode (single-epoch training).
        if data_args.streaming and consumed_samples > 0:
            logger.info(f"Skipping first {consumed_samples} samples to resume from checkpoint.")
            train_dataset.skip(consumed_samples)

        val_dataset = None
        if data_args.val_dataset:
            val_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming
            )
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # If overwrite_cache is false (default), only non-rank-0 nodes load cache without map processing.
                # If overwrite_cache is true, all nodes read the cache and none of them perform map processing.
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
        # convert to sharegpt
        train_dataset = align_dataset(train_dataset, dataset_attr, data_args)
        if val_dataset:
            val_dataset = align_dataset(val_dataset, dataset_attr, data_args)

        # convert text to token id
        if dataset_attr.ranking:
            dataset_processor_cls = PairwiseDatasetProcessor
        elif dataset_attr.packing:
            data_args.cutoff_len -= 1
            dataset_processor_cls = PackedSupervisedDatasetProcessor
        elif dataset_attr.pretrain:
            # Will automatically enable sequences packing in pre-training.
            data_args.packing = data_args.packing if data_args.packing is not None else True
            dataset_processor_cls = PretrainDatasetProcessor
        else:
            dataset_processor_cls = SupervisedDatasetProcessor
        dataset_processor = dataset_processor_cls(template=template, tokenizer=tokenizer, processor=processor,
                                                data_args=data_args)
        preprocess_func = dataset_processor.preprocess_dataset
        if data_args.streaming:
            if data_args.async_preprocess:
                train_dataset = DistributedIterableDataset(train_dataset)
                train_dataset = AsyncPreprocessIterableDataset(
                    train_dataset,
                    preprocess_func,
                    buffer_size=data_args.async_preprocess_buffer_size,
                    num_workers=data_args.preprocessing_num_workers,
                )
            else:
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(train_dataset)).keys())),
                    **kwargs,
                )
                train_dataset = DistributedIterableDataset(train_dataset)

            if val_dataset:
                if data_args.async_preprocess:
                    val_dataset = DistributedIterableDataset(val_dataset)
                    val_dataset = AsyncPreprocessIterableDataset(
                        val_dataset,
                        preprocess_func,
                        buffer_size=data_args.async_preprocess_buffer_size,
                        num_workers=data_args.preprocessing_num_workers,
                    )
                else:
                    val_dataset = val_dataset.map(
                        preprocess_func,
                        batched=True,
                        batch_size=data_args.preprocessing_batch_size,
                        remove_columns=(list(next(iter(val_dataset)).keys())),
                        **kwargs,
                    )
                    val_dataset = DistributedIterableDataset(val_dataset)
                return train_dataset, val_dataset
        else:
            if data_args.preprocess_on_fly:
                train_dataset.set_transform(
                    preprocess_func,
                    output_all_columns=True,
                )
            else:
                train_dataset = train_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(train_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on train_dataset",
                    **kwargs,
                )
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on val_dataset",
                    **kwargs,
                )
                return train_dataset, val_dataset
        if is_rank0():
            print("training example:")
            dataset_processor.print_data_example(next(iter(train_dataset)))
        return train_dataset


def process_reward_dataset(dataset, data_folder, preprocess_param):
    def add_idx(example, idx):
        example['metainfo_idx'] = idx
        return example

    dataset = dataset.map(lambda example, idx: add_idx(example, idx), with_indices=True)
    if not preprocess_param.get('use_tied_data', True):
        filter_func = lambda example: any(example[f"{dim}"] != "same" for dim in dataset.eval_dim)
        dataset = dataset.filter(filter_func)

    convert_func = lambda example: build_reward_data(example, data_folder, **preprocess_param)
    dataset = dataset.map(convert_func, remove_columns=dataset.column_names)
    return dataset


def get_reward_video_dataset(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")

    data_args = DataArgumentsForRewardVideo(**basic_param)

    # Ensure main process handles data processing, while other processes reuse cache to avoid redundant calculations.
    # This strategy is consistent with the data processing strategy used by LLaMA Factory.
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # load dataset from file
        train_dataset = load_dataset(path="csv", data_files=data_args.data_path, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))

        train_dataset = process_reward_dataset(train_dataset, data_args.data_folder, preprocess_param)

        val_dataset = None
        if data_args.val_dataset:
            if data_args.data_path_val:
                val_dataset = load_dataset(path="csv", data_files=data_args.data_path_val, split="train",
                                        cache_dir=data_args.cache_dir,
                                        streaming=data_args.streaming
                )
                val_dataset = process_reward_dataset(val_dataset, data_args.data_folder, preprocess_param)
                train_dataset = train_dataset
            else:
                dataset = train_dataset.train_test_split(test_size=0.02, seed=42)
                train_dataset = dataset['train']
                val_dataset = dataset['test']
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)
        else:
            train_dataset = train_dataset

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # If overwrite_cache is false (default), only non-rank-0 nodes load cache without map processing.
                # If overwrite_cache is true, all nodes read the cache and none of them perform map processing.
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)

        if data_args.streaming:
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = DistributedIterableDataset(val_dataset)

        return [train_dataset, val_dataset]
