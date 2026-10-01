# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, Sampler
from megatron.training import get_args

from mindspeed_mm.data.data_utils.utils import (
    get_seed_worker,
    collate_fn_default,
    cal_gradient_accumulation_size
)
from mindspeed_mm.data.data_utils.constants import GLOBAL_STEP_TOKEN_NUM, AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.data.datasets.t2v_dataset import DynamicVideoTextDataset
from mindspeed_mm.data.dataloader.sampler import (
    LengthGroupedSampler,
    StatefulDistributedSampler,
    VariableVideoBatchSampler,
    BaseRandomBatchSampler,
    BucketBatchSampler,
    AESampler,
    LuminaMetaLenDistSampler
)
from mindspeed_mm.data.dataloader.data_collator import DATA_COLLATOR


class PrefetchGradAccDataLoader:
    """
    A DataLoader wrapper that prefetches a fixed number of batches (equal to gradient
    accumulation steps), computes total and average valid token counts across them,
    and injects these metrics into each batch before yielding.

    This is Used for calculate per-token-loss
    """
    def __init__(self, base_dataloader, grad_acc_step: int):
        """
        Args:
            base_dataloader: The underlying PyTorch DataLoader to wrap.
            grad_acc_step (int): Number of batches to accumulate gradients over.
        """
        if grad_acc_step <= 0:
            raise ValueError("grad_acc_step must be a positive integer.")
        self.grad_acc_step = grad_acc_step
        self.base_dataloader = base_dataloader
        self._current_iterator = None  # Holds the active generator

    def __iter__(self):
        # Start a new iteration (reset state)
        self._current_iterator = self._generate_batches()
        return self  # Return self as the iterator

    def __next__(self):
        if self._current_iterator is None:
            # If __next__ is called before __iter__, start iteration
            self._current_iterator = self._generate_batches()
        try:
            return next(self._current_iterator)
        except StopIteration:
            self._current_iterator = None  # Reset on exhaustion
            raise

    def _generate_batches(self):
        """Generator that yields batches with injected token counts."""
        base_iter = iter(self.base_dataloader)
        try:
            while True:
                buffer = []
                total_tokens = 0
                fetched = 0

                # Prefetch grad_acc_step batches
                try:
                    for _ in range(self.grad_acc_step):
                        batch = next(base_iter)
                        valid_token_count = (batch["labels"] > -1).sum()
                        total_tokens += valid_token_count
                        buffer.append(batch)
                        fetched += 1
                except StopIteration:
                    if fetched == 0:
                        break  # No more data

                avg_tokens = total_tokens / self.grad_acc_step
                for batch in buffer:
                    batch_dict = dict(batch)
                    batch_dict[GLOBAL_STEP_TOKEN_NUM] = total_tokens
                    batch_dict[AVG_PER_STEP_TOKEN_NUM] = avg_tokens
                    yield batch_dict
        finally:
            if hasattr(base_iter, 'close'):
                base_iter.close()


def prepare_base_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=None,
    collate_param=None,
    dataset_param=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    collate_fn = None
    if collate_param:
        data_collate_type = collate_param.pop("model_name")
        collate_fn = DATA_COLLATOR[data_collate_type](dataset_param=dataset_param, **collate_param)
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=get_seed_worker(seed),
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )


def prepare_sampler_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=None,
    process_group: Optional[ProcessGroup] = None,
    consumed_samples=0,
    data_sharding=False,
    sampler_type="stateful_distributed_sampler",
    group_frame=False,
    group_resolution=False,
    group_data=False,
    initial_global_step_for_sampler=0,
    collate_param=None,
    dataset_param=None,
    priority_mode="data_bucketing_img",
    generator=None
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
        num_workers = 0

    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    if sampler_type == "stateful_distributed_sampler":
        collate_fn = None
        if collate_param:
            data_collate_type = collate_param.pop("model_name")
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param, dataset_param=dataset_param)

        if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
            sampler = None
        else:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=shuffle,
                consumed_samples=consumed_samples,
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=get_seed_worker(seed),
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=generator
        )

    elif sampler_type == "LengthGroupedSampler":
        gradient_accumulation_size = cal_gradient_accumulation_size()
        if group_data and (group_frame or group_resolution):
            raise AssertionError(
                "group_data and (group_frame or group_resolution) cannot be true at the same time!"
            )
        sample_num_frames = []
        sample_sizes = []
        for data_sample in dataset.data_samples:
            sample_num_frames.append(data_sample["sample_num_frames"])
            if data_sample.get("sample_size", None):
                sample_size = data_sample["sample_size"]
            else:
                sample_size = f'{len(data_sample["sample_frame_index"])}x{data_sample["resolution"]["sample_height"]}x{data_sample["resolution"]["sample_width"]}'
            sample_sizes.append(sample_size)
        sampler = (
            LengthGroupedSampler(
                batch_size,
                world_size=process_group.size(),
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=shuffle,
                gradient_accumulation_size=gradient_accumulation_size,
                initial_global_step=initial_global_step_for_sampler,
                lengths=sample_num_frames if not group_data else sample_sizes,
                group_frame=group_frame,
                group_resolution=group_resolution,
                group_data=group_data,
                consumed_samples=consumed_samples,
            )
            if (group_frame or group_resolution or group_data)
            else None
        )
        if sampler is None:
            sampler = StatefulDistributedSampler(
                dataset,
                num_replicas=process_group.size(),
                rank=process_group.rank(),
                shuffle=shuffle,
                consumed_samples=consumed_samples,
            )

        if collate_param is None:
            collate_fn = None
        elif "model_name" not in collate_param:
            raise ValueError("collate_param with model_name must be provided.")

        if collate_param:
            # Inject params to collate params to avoid duplicate configs
            collate_param.update(dataset_param.preprocess_parameters.to_dict())
            group_param = dict(group_data=group_data, group_resolution=group_resolution, group_frame=group_frame,
                            batch_size=batch_size)
            collate_param.update(group_param)

            data_collate_type = collate_param.pop("model_name")
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param)

        return DataLoader(
            dataset,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler if sampler is not None else None,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=generator
        )

    elif sampler_type == "BaseRandomBatchSampler":
        batch_sampler = BaseRandomBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            consumed_samples=consumed_samples,
            data_sharding=data_sharding,
        )
        if collate_param is None:
            collate_fn = None
        elif "model_name" not in collate_param:
            raise ValueError("collate_param with model_name must be provided.")

        if collate_param:
            data_collate_type = collate_param.pop("model_name")
            collate_fn = DATA_COLLATOR[data_collate_type](**collate_param, dataset_param=dataset_param)

        return DataLoader(
            dataset,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=get_seed_worker(seed),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=generator
        )

    elif sampler_type == "BucketBatchSampler":
        args = get_args()
        data_config = args.mm.data
        gbs = args.global_batch_size
        batch_sampler = BucketBatchSampler(
            dataset,
            data_config=data_config,
            batch_size=batch_size,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            consumed_samples=consumed_samples,
            data_sharding=data_sharding,
            global_batch_size=gbs,
        )

        if collate_param is None or "model_name" not in collate_param:
            raise ValueError("collate_param with model_name must be provided.")
        data_collate_type = collate_param.pop("model_name")
        collate_fn = DATA_COLLATOR[data_collate_type](**collate_param, dataset_param=dataset_param)

        return DataLoader(
            dataset,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=get_seed_worker(seed),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=generator
        )

    elif sampler_type == "SequentialSampler":
        return build_sequential_loader(DataLoaderArgs(dataset,
                                                      batch_size,
                                                      drop_last,
                                                      pin_memory,
                                                      process_group,
                                                      num_workers,
                                                      shuffle,
                                                      prefetch_factor,
                                                      persistent_workers,
                                                      consumed_samples))
    elif sampler_type == "AESampler":
        sampler = AESampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            generator=generator
        )
    elif sampler_type == "LuminaMetaLenDistSampler":
        acc_grad = cal_gradient_accumulation_size()
        sampler = LuminaMetaLenDistSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            batch_size=batch_size,
            acc_grad=acc_grad
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=lambda batch: tuple(zip(*batch)),
            drop_last=drop_last,
            generator=generator
        )
    else:
        raise NotImplementedError(f"sampler type: {sampler_type}")


class DistributedBatchSampler(BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False, gradient_accumulation_steps=None, consumed_samples=0):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            raise ValueError('please select rank')
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps
        self.start_iter += (consumed_samples % len(self.sampler)) // self.effective_batch_size

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= self.batch_size
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        if start >= len(batch):
            return batch[0:1]
        else:
            return batch[start:end]


@dataclass
class DataLoaderArgs:
    dataset: object
    batch_size: int
    drop_last: bool
    pin_memory: bool
    process_group: object
    num_workers: int
    shuffle: bool
    prefetch_factor: Optional[int]
    persistent_workers: int
    consumed_samples: int


def build_sequential_loader(args: DataLoaderArgs):
    sampler = SequentialSampler(args.dataset)

    world_size = torch.distributed.get_world_size(group=args.process_group)
    rank = args.process_group.rank()
    distributed = world_size > 1
    batch_size = args.batch_size * world_size
    consumed_samples = args.consumed_samples


    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size,
                                            args.drop_last,
                                            rank,
                                            world_size,
                                            gradient_accumulation_steps=1,
                                            consumed_samples=consumed_samples)

    data_loader = torch.utils.data.DataLoader(args.dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=args.pin_memory,
                                              collate_fn=None,
                                              persistent_workers=args.persistent_workers,
                                              prefetch_factor=args.prefetch_factor)
    return data_loader


def prepare_variable_dataloader(
        dataset,
        shuffle=False,
        seed=1024,
        auto_gen_bucket=False,
        drop_last=False,
        pin_memory=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=None,
        process_group: Optional[ProcessGroup] = None,
        bucket_config=None,
        num_bucket_build_workers=1,
        sampler_type="variable_video_batch_sampler",
        consumed_samples=0,
        **kwargs,
    ):
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False
    if isinstance(dataset, DynamicVideoTextDataset) and sampler_type == "variable_video_batch_sampler":
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
            auto_gen_bucket=auto_gen_bucket,
            consumed_samples=consumed_samples,
        )

        return DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    worker_init_fn=get_seed_worker(seed),
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    collate_fn=collate_fn_default,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    **kwargs,
                )
    else:
        return NotImplementedError
