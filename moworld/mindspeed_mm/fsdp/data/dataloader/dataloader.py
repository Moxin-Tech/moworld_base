# pylint: skip-file
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

import threading
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from mindspeed_mm.fsdp.data.data_utils.utils import get_seed_worker
from mindspeed_mm.fsdp.data.dataloader.sampler import BaseRandomBatchSampler
from mindspeed_mm.fsdp.data.dataloader.data_collator import DATA_COLLATOR
from mindspeed_mm.fsdp.utils.constants import GLOBAL_STEP_TOKEN_NUM, AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.utils.utils import move_to_device


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

    return StatefulDataLoader(
        dataset,
        pin_memory=pin_memory,
        pin_memory_device=get_device_type(),
        collate_fn=collate_fn,
        worker_init_fn=get_seed_worker(seed),
        num_workers=num_workers,
        batch_sampler=None,
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
    data_sharding=False,
    sampler_type="stateful_distributed_sampler",
    collate_param=None,
    dataset_param=None,
    model=None,
    **kwargs,
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

    if sampler_type == "BaseRandomBatchSampler":
        batch_sampler = BaseRandomBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            data_sharding=data_sharding,
        )
        if collate_param is None:
            collate_fn = None
        elif "model_name" not in collate_param:
            raise ValueError("collate_param with model_name must be provided.")

        if collate_param:
            data_collate_type = collate_param.pop("model_name")
            if hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn')):
                collate_fn = dataset.collate_fn
            else:
                collate_fn = DATA_COLLATOR[data_collate_type](**collate_param, dataset_param=dataset_param, model=model)

        return StatefulDataLoader(
            dataset,
            pin_memory=pin_memory,
            pin_memory_device=get_device_type(),
            collate_fn=collate_fn,
            worker_init_fn=get_seed_worker(seed),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} is not implemented.")


class PrefetchGradAccDataLoader:
    """
    A DataLoader wrapper that prefetches a fixed number of batches (equal to gradient
    accumulation steps), computes total and average valid token counts across them,
    and injects these metrics into each batch before yielding.

    This is Used for calculate per-token-loss
    """

    def __init__(self, base_dataloader: StatefulDataLoader, grad_acc_step: int):
        """
        Args:
            base_dataloader: The underlying PyTorch DataLoader to wrap.
            grad_acc_step (int): Number of batches to accumulate gradients over.
        """
        if grad_acc_step <= 0:
            raise ValueError("grad_acc_step must be a positive integer.")
        self.grad_acc_step = grad_acc_step
        self.base_dataloader = base_dataloader
        self.base_iter, _, _ = build_iterations(self.base_dataloader)
        self._current_iterator = None  # Holds the active generator

    def __iter__(self):
        # Start a new iteration (reset state)
        self._current_iterator = self._generate_batches()
        return self  # Return self as the iterator

    def __len__(self):
        if hasattr(self.base_dataloader, "__len__"):
            return len(self.base_dataloader)
        else:
            return 0

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
        try:
            while True:
                buffer = []
                total_tokens = 0
                fetched = 0

                # Prefetch grad_acc_step batches
                try:
                    for _ in range(self.grad_acc_step):
                        batch = next(self.base_iter)
                        valid_token_count = (batch["labels"] > -1).sum()  # tokens of shift labels
                        total_tokens += valid_token_count
                        buffer.append(batch)
                        fetched += 1
                except StopIteration:
                    if fetched == 0:
                        break  # No more data

                # 尾端不足时，平均tokens数应该用实际值做除法
                denom = fetched if fetched > 0 else self.grad_acc_step
                avg_tokens = total_tokens / denom
                for batch in buffer:
                    batch_dict = dict(batch)
                    batch_dict[GLOBAL_STEP_TOKEN_NUM] = total_tokens
                    batch_dict[AVG_PER_STEP_TOKEN_NUM] = avg_tokens
                    yield batch_dict
        finally:
            if hasattr(self.base_iter, 'close'):
                self.base_iter.close()

    def state_dict(self):
        return self.base_dataloader.state_dict()

    def load_state_dict(self, **kwargs):
        self.base_dataloader.load_state_dict(**kwargs)


class Preloader:
    """
    Async data preloader that overlaps CPU data loading with training computation.

    Uses a background thread to preload the next CPU batch while the current batch
    is being processed by forward/backward.
    """

    def __init__(self, data_iterator, param_dtype=None):
        self.data_iterator = data_iterator
        self.param_dtype = param_dtype
        self.next_batch = None
        self._thread = None
        self._exhausted = False
        self._load_next()

    def _load_next(self):
        """Load the next batch from the iterator."""
        try:
            self.next_batch = next(self.data_iterator)
        except (StopIteration, Exception):
            self.next_batch = None
            self._exhausted = True

    def _start_async_preload(self):
        """Start a background thread to preload the next CPU batch."""
        if self._exhausted:
            return
        self._thread = threading.Thread(target=self._load_next, daemon=True)
        self._thread.start()

    def next(self):
        """Wait for CPU preload, and start next CPU preload."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self.next_batch is None:
            return None
        batch = move_to_device(self.next_batch, float_dtype=self.param_dtype)
        self._start_async_preload()
        return batch

    def __iter__(self):
        yield self.next()

    def __next__(self):
        batch = self.next()
        if batch is None:
            raise StopIteration("Dataloader has been exhausted, no more data available.")
        return batch
