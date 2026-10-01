import os
import warnings
from functools import partial, wraps

from datasets import Dataset, load_dataset
from transformers.training_args import TrainingArguments

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.utils import is_rank0

from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.utils.hetero_parallel import change_parallel_state
from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DatasetAttr,
    load_tokenizer,
    align_dataset,
    SupervisedDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.datasets.qwen2vl_dataset import DistributedIterableDataset, AsyncPreprocessIterableDataset
logger = get_logger(__name__)


def build_train_valid_test_datasets_wrapper(build_train_valid_test_datasets):
    @wraps(build_train_valid_test_datasets)
    def wrapper(*args, **kwargs):
        args = (build_train_valid_test_datasets_provider,) + args[1:]
        return build_train_valid_test_datasets(*args, **kwargs)

    return wrapper


def build_train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    if args.hetero_parallel:
        print_rank_0("change parallel state for data loader ...")
        change_parallel_state("text_decoder")

        if args.hetero_encoder_mbs_scale > 1:
            pp_mbs = args.micro_batch_size
            args.micro_batch_size = pp_mbs * args.hetero_encoder_mbs_scale

    datasets = build_mm_dataset_ldt(data_config.dataset_param)
    if not mpu.is_pipeline_first_stage():
        return None, None, None
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )

    micro_batch_size = args.micro_batch_size
    if args.use_data_balance:
        global_batch_size = args.micro_batch_size * get_num_microbatches()
        if args.hetero_encoder_mbs_scale > 1:
            global_batch_size = global_batch_size // args.hetero_encoder_mbs_scale
        args.micro_batch_size = global_batch_size

    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        args.micro_batch_size = micro_batch_size
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            valid_dataloader = build_dataloader(valid_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            args.micro_batch_size = micro_batch_size
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    if args.hetero_parallel and args.hetero_encoder_mbs_scale > 1:
        args.micro_batch_size = pp_mbs

    return train_dataloader, valid_dataloader, test_dataloader


def build_mm_dataset_ldt(dataset_param):
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    for check_key in ["dataset_type", "basic_parameters", "preprocess_parameters"]:
        if check_key not in dataset_param:
            raise AssertionError(f"Key parameter missing: {check_key}")
    dataset_type = dataset_param["dataset_type"]
    basic_param = dataset_param["basic_parameters"]
    preprocess_param = dataset_param["preprocess_parameters"]
    if dataset_type == "huggingface":
        return get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param)

    return build_mm_dataset(dataset_param)


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
        train_dataset = None
        val_dataset = None
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            train_dataset = load_dataset(path="json", data_files=data_args.dataset, split="train",
                                        cache_dir=data_args.cache_dir,
                                        streaming=data_args.streaming)
            if data_args.max_samples and not data_args.streaming:
                train_dataset = train_dataset.select(range(data_args.max_samples))

            if consumed_samples > 0:
                logger.info(f"Skipping first {consumed_samples} samples to resume from checkpoint.")
                train_dataset.skip(consumed_samples)

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
        if train_dataset:
            train_dataset = align_dataset(train_dataset, dataset_attr, data_args)
        if val_dataset:
            val_dataset = align_dataset(val_dataset, dataset_attr, data_args)

        # convert text to token id
        if dataset_attr.ranking:
            dataset_processor_cls = PairwiseDatasetProcessor
        elif dataset_attr.packing:
            data_args.cutoff_len -= 1
            dataset_processor_cls = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_cls = SupervisedDatasetProcessor
        dataset_processor = dataset_processor_cls(template=template, tokenizer=tokenizer, processor=processor,
                                                data_args=data_args)
        preprocess_func = dataset_processor.preprocess_dataset
        if data_args.streaming:
            if train_dataset:
                if data_args.async_preprocess:
                    train_dataset = DistributedIterableDataset(train_dataset)
                    train_dataset = AsyncPreprocessIterableDataset(train_dataset, preprocess_func, buffer_size=8)
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
                    val_dataset = AsyncPreprocessIterableDataset(val_dataset, preprocess_func, buffer_size=8)
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
            if train_dataset:
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
