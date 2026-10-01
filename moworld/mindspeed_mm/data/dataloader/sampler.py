from typing import Iterator, List, Optional
import math
import logging
import random
import time
from collections import Counter, OrderedDict, defaultdict
from pprint import pformat

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from megatron.legacy.data.data_samplers import RandomSeedDataset
from pandarallel import pandarallel
from transformers import AutoProcessor

from mindspeed_mm.data.datasets.t2v_dataset import DynamicVideoTextDataset
from mindspeed_mm.data.data_utils.bucket import Bucket
from mindspeed_mm.data.data_utils.aspect_ratio import get_num_pixels, get_resolution_with_aspect_ratio
from mindspeed_mm.data.data_utils.utils import format_numel_str
from mindspeed_mm.data.dataloader.bucket_manager import BucketManager_qwen2vl, BucketManager_internvl2, BucketManager


def split_to_even_chunks(indices, lengths, num_chunks, batch_size):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        chunks = [indices[i::num_chunks] for i in range(num_chunks)]
    else:
        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float("inf")

    pad_chunks = []
    for _, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            if batch_size <= len(chunk):
                raise AssertionError(
                    "The batch_size must be larger than the length of chunk."
                )

            if len(chunk) != 0:
                chunk = chunk + [
                    random.choice(chunk) for _ in range(batch_size - len(chunk))
                ]
            else:
                chunk = random.choice(pad_chunks)
        pad_chunks.append(chunk)
    return pad_chunks


def split_data_to_even_chunks(megabatch, lengths, world_size, batch_size, shuffle=True):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """
    # [1, 2, 3, 4] -> [[1, 2], [3, 4]]
    # [1, 2, 3] -> [[1, 2], [3]]
    # [1, 2] -> [[1], [2]]
    # [1] -> [[1], []]
    chunks = [megabatch[i::world_size] for i in range(world_size)]

    pad_chunks = []
    for _, chunk in enumerate(chunks):
        if batch_size != len(chunk):
            if batch_size <= len(chunk):
                raise AssertionError("batch_size must greater than len_chunk !")
            if len(chunk) != 0:  # [[1, 2], [3]] -> [[1, 2], [3, 3]]
                if shuffle:
                    chunk = chunk + [random.choice(chunk) for _ in range(batch_size - len(chunk))]
                else:
                    chunk = chunk + [chunk[0] for _ in range(batch_size - len(chunk))]
            else: # [[1], []] -> [[1], [1]]
                if shuffle:
                    chunk = random.choice(pad_chunks)
                else:
                    chunk = pad_chunks[0]

        pad_chunks.append(chunk)
    return pad_chunks


def group_frame_fun(indices, lengths):
    # sort by num_frames
    indices.sort(key=lambda i: lengths[i], reverse=True)
    return indices


def group_resolution_fun(indices):
    raise NotImplementedError


def group_frame_and_resolution_fun(indices):
    raise NotImplementedError


def last_group_frame_fun(megabatches, lengths, shuffle=True):
    re_megabatches = []
    for megabatch in megabatches:
        re_megabatch = []
        for batch in megabatch:
            if len(batch) == 0:
                raise AssertionError("The length of batch is zero")
            len_each_batch = [lengths[i] for i in batch]
            idx_length_dict = dict([*zip(batch, len_each_batch)])
            count_dict = Counter(len_each_batch)
            if len(count_dict) != 1:
                sorted_by_value = sorted(count_dict.items(), key=lambda item: item[1])
                pick_length = sorted_by_value[-1][0]  # the highest frequency
                candidate_batch = [
                    idx
                    for idx, length in idx_length_dict.items()
                    if length == pick_length
                ]

                if shuffle:
                    random_select_batch = [
                        random.choice(candidate_batch)
                        for _ in range(len(len_each_batch) - len(candidate_batch))
                    ]
                else:
                    random_select_batch = [
                        candidate_batch[0]
                        for _ in range(len(len_each_batch) - len(candidate_batch))
                    ]

                batch = candidate_batch + random_select_batch
            re_megabatch.append(batch)
        re_megabatches.append(re_megabatch)
    return re_megabatches


def last_group_resolution_fun(indices):
    raise NotImplementedError


def last_group_frame_and_resolution_fun(indices):
    raise NotImplementedError


def get_length_grouped_indices(
    lengths,
    batch_size,
    world_size,
    generator=None,
    group_frame=False,
    group_resolution=False,
    seed=42,
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        generator = torch.Generator().manual_seed(
            seed
        )  # every rank will generate a fixed order but random index

    indices = torch.randperm(len(lengths), generator=generator).tolist()

    if group_frame and not group_resolution:
        indices = group_frame_fun(indices, lengths)
    elif not group_frame and group_resolution:
        indices = group_resolution_fun(indices)
    elif group_frame and group_resolution:
        indices = group_frame_and_resolution_fun(indices)

    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i: i + megabatch_size]
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size, batch_size)
        for megabatch in megabatches
    ]

    indices = torch.randperm(len(megabatches), generator=generator).tolist()
    shuffled_megabatches = [megabatches[i] for i in indices]
    if group_frame and not group_resolution:
        shuffled_megabatches = last_group_frame_fun(shuffled_megabatches, lengths)
    elif not group_frame and group_resolution:
        shuffled_megabatches = last_group_resolution_fun(shuffled_megabatches, indices)
    elif group_frame and group_resolution:
        shuffled_megabatches = last_group_frame_and_resolution_fun(
            shuffled_megabatches, indices
        )

    out_list = []
    for megabatch in shuffled_megabatches:
        for batch in megabatch:
            for i in batch:
                out_list.append(i)
    return out_list


def group_data_fun(lengths, generator=None, shuffle=True):
    # counter is decrease order
    counter = Counter(lengths)  # counter {'1x256x256': 3, ''}   lengths ['1x256x256', '1x256x256', '1x256x256', ...]
    grouped_indices = defaultdict(list)
    for idx, item in enumerate(lengths):  # group idx to a list
        grouped_indices[item].append(idx)

    grouped_indices = dict(grouped_indices)  # {'1x256x256': [0, 1, 2], ...}
    sorted_indices = [grouped_indices[item] for (item, _) in sorted(counter.items(), key=lambda x: x[1], reverse=True)]

    # shuffle in each group
    if shuffle:
        shuffle_sorted_indices = []
        for indice in sorted_indices:
            shuffle_idx = torch.randperm(len(indice), generator=generator).tolist()
            shuffle_sorted_indices.extend([indice[idx] for idx in shuffle_idx])

        return shuffle_sorted_indices
    else:
        unshuffle_sorted_indices = []
        for indice in sorted_indices:
            unshuffle_sorted_indices.extend(indice)

        return unshuffle_sorted_indices


def get_length_grouped_data_indices(
        lengths,
        batch_size,
        world_size,
        gradient_accumulation_size,
        initial_global_step,
        generator=None,
        group_data=False,
        seed=42,
        shuffle=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    if generator is None:
        if world_size == 1:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator()  # every rank will generate a fixed order but random index

    if group_data:
        indices = group_data_fun(lengths, generator, shuffle)
    else:
        if shuffle:
            indices = torch.randperm(len(lengths), generator=generator).tolist()
        else:
            indices = list(range(len(lengths)))

    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]

    megabatches = [split_data_to_even_chunks(megabatch, lengths, world_size, batch_size, shuffle) for megabatch in megabatches]

    if shuffle:
        indices_mega = torch.randperm(len(megabatches), generator=generator).tolist()
    else:
        indices_mega = list(range(len(megabatches)))

    megabatches = [megabatches[i] for i in indices_mega]

    if group_data:
        megabatches = last_group_frame_fun(megabatches, lengths, shuffle)

    initial_global_step = initial_global_step * gradient_accumulation_size
    megabatches = megabatches[initial_global_step:]

    out_list = []
    for megabatch in megabatches:
        for batch in megabatch:
            for i in batch:
                out_list.append(i)
    return out_list


class LengthGroupedSampler(DistributedSampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        gradient_accumulation_size: int = 1,
        initial_global_step: int = 0,
        lengths: Optional[List[int]] = None,
        group_frame=False,
        group_resolution=False,
        group_data=False,
        generator=None,
        consumed_samples: int = 0,
    ):
        super().__init__(dataset=lengths, num_replicas=num_replicas, rank=rank)

        if lengths is None:
            raise ValueError("Lengths must be provided.")
        if world_size == -1:
            raise ValueError("world_size must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.shuffle = shuffle
        self.initial_global_step = initial_global_step
        self.gradient_accumulation_size = gradient_accumulation_size
        self.lengths = lengths
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.group_data = group_data
        self.generator = generator
        self.consumed_samples = consumed_samples

    def __len__(self):
        if self.group_data:
            return len(self.lengths) - self.initial_global_step * self.batch_size * self.world_size * self.gradient_accumulation_size
        return len(self.lengths)

    def __iter__(self):
        if not self.group_data:
            indices = get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                group_frame=self.group_frame,
                group_resolution=self.group_resolution,
                generator=self.generator,
            )
        else:
            indices = get_length_grouped_data_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                self.gradient_accumulation_size,
                self.initial_global_step,
                group_data=self.group_data,
                generator=self.generator,
                shuffle=self.shuffle
            )

        # start sampling from the consumed samples point to continue training from where it left off
        start_index = self.consumed_samples % len(indices)
        indices = indices[start_index:]
        actual_indices_len = len(indices)

        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.consumed_samples += actual_indices_len
        return iter(indices)


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        consumed_samples: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0
        self.consumed_samples = consumed_samples // num_replicas

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        self.start_index = self.consumed_samples % self.num_samples
        indices = indices[self.start_index:]
        actual_indices_len = len(indices)
        self.consumed_samples += actual_indices_len
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_start_index(self, start_index: int) -> None:
        self.start_index = start_index


class BaseRandomBatchSampler(DistributedSampler):
    """
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Default: ``True``. (It is not implemented that the drop_last is false.)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        consumed_samples: int = 0,
        data_sharding: bool = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.total_samples = len(dataset)
        self.micro_batch_size = batch_size
        self.consumed_samples = consumed_samples
        self.data_sharding = data_sharding
        self.epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * self.num_replicas
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        if not drop_last:
            raise ValueError("It is not implemented that the drop_last is false.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.num_replicas
            start_idx = self.rank * bucket_size
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
            else:
                idx_range_bucket = list(range(bucket_size))
            idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_total = \
                    torch.randperm(full_bucket_size, generator=g).tolist()
            else:
                idx_range_total = list(range(full_bucket_size))
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.rank::self.num_replicas]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class BucketBatchSampler(BaseRandomBatchSampler):
    """
    Args:
        dataset (Dataset): The dataset used for sampling. This should be a `torch.utils.data.Dataset` object
                            containing the data to be sampled from.
        batch_size (int, optional): The size of each batch. Default is 1.
        num_replicas (int, optional): The number of processes (replicas) participating in distributed training.
                                      By default, the world size is retrieved from the current distributed group.
        rank (int, optional): The rank of the current process within the `num_replicas`. By default, the rank is
                               retrieved from the current distributed group.
        shuffle (bool, optional): Whether to shuffle the indices of the dataset. If `True` (default), the sampler
                                  will shuffle the indices before sampling. This is important for training as it
                                  helps to reduce model overfitting by providing randomization.
        seed (int, optional): The random seed used to shuffle the sampler if `shuffle=True`. This seed should be
                              the same across all processes in the distributed group to ensure consistent results.
                              Default is 0.
        drop_last (bool, optional): If `True`, the sampler will drop the last batch if it is smaller than
                                    `batch_size` to ensure that each batch is fully utilized. Default is `True`.
                                    (Note: Drop last is not implemented when set to False.)
        consumed_samples (int, optional): The number of samples that have been consumed so far. Default is 0.
        data_sharding (bool, optional): Whether to enable data sharding. If `True`, the data is split across
                                        multiple replicas to ensure that each replica gets a distinct subset of
                                        the dataset. Default is `False`.
    """
    def __init__(
        self,
        dataset,
        data_config,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        consumed_samples: int = 0,
        data_sharding: bool = False,
        global_batch_size: int = 128,
    ):
        self.global_batch_size = global_batch_size
        self.data_config = data_config
        super().__init__(dataset, batch_size, num_replicas, rank, shuffle, seed, drop_last, consumed_samples, data_sharding)
        self.bucket_manager = None

    def __len__(self):
        """Total number of returned samples"""
        return self.total_samples

    def __iter__(self) -> Iterator:
        """Iterator, which generates the index of each batch."""
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        dataset_param = self.data_config.dataset_param
        dataloader_param = self.data_config.dataloader_param
        model_name = dataloader_param.collate_param.model_name
        priority_mode = getattr(dataloader_param, 'priority_mode', 'data_bucketing_img')
        preprocess_parameters = self.data_config.dataset_param.preprocess_parameters

        bucket_manager = None
        if self.bucket_manager is None:
            start_time = time.time()
            if self.rank == 0:
                if model_name == "qwen2vl":
                    image_max_pixels = preprocess_parameters.image_max_pixels
                    image_size = int(math.sqrt(image_max_pixels))
                    processor = AutoProcessor.from_pretrained(preprocess_parameters.model_name_or_path, local_files_only=True)
                    attributes = ["patch_size", "merge_size", "min_pixels", "max_pixels"]
                    values = {}
                    for attr in attributes:
                        values[attr] = getattr(processor.image_processor, attr, -1)
                        if values[attr] == -1:
                            raise AttributeError(f"Error: '{attr}' not found in processor.image_processor. Please check your configuration.")
                    patch_size = values.get("patch_size", None)
                    merge_size = values.get("merge_size", None)
                    min_pixels = values.get("min_pixels", None)
                    max_pixels = values.get("max_pixels", None)

                    if any(v is None for v in [patch_size, merge_size, min_pixels, max_pixels]):
                        raise KeyError("One or more required keys are missing from the 'values' dictionary.")

                    bucket_manager = BucketManager_qwen2vl(
                        image_size=image_size,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        batch_size=self.micro_batch_size,
                        sharding=self.data_sharding,
                        num_replicas=self.num_replicas,
                        keep_remainder=True,
                        rank=self.rank,
                        global_batch_size=self.global_batch_size,
                        priority_mode=priority_mode
                    )
                elif model_name == "internvl":
                    min_dynamic_patch = dataset_param.min_dynamic_patch
                    max_dynamic_patch = dataset_param.max_dynamic_patch
                    image_size = dataset_param.image_size
                    bucket_manager = BucketManager_internvl2(
                        image_size=image_size,
                        min_num=min_dynamic_patch,
                        max_num=max_dynamic_patch,
                        batch_size=self.micro_batch_size,
                        sharding=self.data_sharding,
                        num_replicas=self.num_replicas,
                        keep_remainder=True,
                        rank=self.rank,
                        global_batch_size=self.global_batch_size,
                        priority_mode=priority_mode
                    )
                bucket_manager.group_by_bucket(self.dataset)
                end_time = time.time()
                print(f"create BucketManager & group_by_bucket cost: {end_time - start_time} seconds")
                self.bucket_manager = bucket_manager
            else:
                self.bucket_manager = None
        else:
            bucket_manager = self.bucket_manager

        # data sharding and random sampling
        if priority_mode == "data_bucketing_img":
            if self.rank == 0:
                if self.shuffle:
                    idx_range_total = bucket_manager.generate_index(shuffle=True, seed=self.epoch)
                else:
                    idx_range_total = bucket_manager.generate_index(shuffle=False)
                idx_range_total_b = [idx_range_total]
            else:
                idx_range_total_b = [None]
            dist.broadcast_object_list(idx_range_total_b, src=0)
            idx_range_total = idx_range_total_b[0]
            if self.data_sharding:
                bucket_size = (len(idx_range_total) // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
                bucket_offset = current_epoch_samples // self.num_replicas
                start_idx = self.rank * bucket_size
                idx_range_bucket = idx_range_total[start_idx:start_idx + bucket_size]
                idx_range = [x for x in idx_range_bucket[bucket_offset:]]
            else:
                full_bucket_offset = current_epoch_samples
                idx_range_active = idx_range_total[full_bucket_offset:]
                idx_range = idx_range_active[self.rank::self.num_replicas]

        elif priority_mode == "data_reordering_img":
            if self.rank == 0:
                final_results_dict_b = [bucket_manager.final_results_dict]
            else:
                final_results_dict_b = [None]
            dist.broadcast_object_list(final_results_dict_b, src=0)
            final_results_dict = final_results_dict_b[0]
            if self.data_sharding:
                bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                            * self.micro_batch_size
                bucket_offset = current_epoch_samples // self.num_replicas
                start_idx = self.rank * bucket_size
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
                else:
                    idx_range_bucket = list(range(bucket_size))
                idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
            else:
                full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                    * self.micro_batch_size
                full_bucket_offset = current_epoch_samples
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    idx_range_total = \
                        torch.randperm(full_bucket_size, generator=g).tolist()
                else:
                    idx_range_total = list(range(full_bucket_size))

                idx_range_active = idx_range_total[full_bucket_offset:]
                idx_range = idx_range_active[self.rank::self.num_replicas]
            idx_range = BucketManager.generate_index_by_gbs(idx_range, final_results_dict, self.global_batch_size, self.num_replicas)

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None, fps_max=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        data["fps"],
        frame_interval,
        seed + data["id"] * num_bucket,
        fps_max,
    )


class VariableVideoBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: DynamicVideoTextDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
        consumed_samples: int = 0,
        auto_gen_bucket: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        for resolution, configs in bucket_config.items():
            bucket_config[resolution] = {int(k): tuple(v) for k, v in configs.items()}
        self.bucket = Bucket(bucket_config, auto_gen_bucket)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers
        self.last_micro_batch_access_index += consumed_samples
        self.auto_gen_bucket = auto_gen_bucket

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = (self.last_micro_batch_access_index // self.num_replicas) % num_iters

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // dist.get_world_size()

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        logging.info("Building buckets...")
        bucket_ids = self.dataset.data_samples.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=self.dataset.frame_interval,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
            fps_max=self.dataset.fps_max
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket()
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        if self.auto_gen_bucket:
            keylist = [key[0] for key in num_hwt_dict.keys() if isinstance(key, tuple) and len(key) > 0]
            aspect_ratios = {key: get_resolution_with_aspect_ratio(key) for key in keylist}
            num_hwt_dict = dict(
                sorted(num_hwt_dict.items(), key=lambda x: (aspect_ratios[x[0][0]]), reverse=True)
            )
        else:
            num_hwt_dict = dict(
                sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
            )

        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class AESampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.current_index = 0

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        if len(indices) != self.total_size:
            raise ValueError("The length of indices must equals total_size!")

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        if len(indices) != self.num_samples:
            raise ValueError("The length of indices on each device must equals num_samples!")

        while self.current_index < len(indices):
            yield indices[self.current_index]
            self.current_index += 1
        self.current_index = 0

    def state_dict(self) -> dict:
        return {
            'epoch': self.epoch,
            'seed': self.seed,
            'current_index': self.current_index
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict['epoch']
        self.seed = state_dict['seed']
        self.current_index = state_dict.get('current_index', 0)


class LuminaMetaLenDistSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        batch_size=None,
        acc_grad=1,
        length_clustering=True,
        allow_mixed_task_among_acc=False,
    ):
        self.dataset = dataset
        self.total_samples = len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.acc_grad = acc_grad
        self.length_clustering = length_clustering
        self.allow_mixed_task_among_acc = allow_mixed_task_among_acc

        self.epoch = 0
        self.micro_batch_times_data_parallel_size = self.batch_size * self.num_replicas
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        global_bsz_acc = batch_size * num_replicas * acc_grad

        group_len = defaultdict(int)
        for meta in dataset.meta_collection:
            group_len[meta["type"]] += int(meta["len"] * meta.get("ratio", 1.0))

        group_len = {key: val // global_bsz_acc * global_bsz_acc for key, val in group_len.items()}

        self.total_size = sum(list(group_len.values()))
        self.num_samples = self.total_size // num_replicas

    def __iter__(self) -> Iterator:
        global_batch_size = self.batch_size * self.num_replicas
        global_bsz_acc = self.batch_size * self.num_replicas * self.acc_grad
        rng = np.random.default_rng(self.seed + self.epoch)

        group_indices_and_len = defaultdict(list)

        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for meta in self.dataset.meta_collection:
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]
            indices = list(range(start_idx, end_idx))
            indices_and_len = [[idx, length] for idx, length in zip(indices, meta["item_len_list"])]
            if meta.get("ratio", 1.0) != 1.0:
                indices_and_len = list(rng.choice(indices_and_len, int(meta["len"] * meta["ratio"]), replace=False))
                print(f"meta{i}: sample (ratio = {meta['ratio']}) {len(indices_and_len)} items")
            group_indices_and_len[meta["type"]].extend(indices_and_len)
            # Update the starting index for the next collection
            start_idx = end_idx

        for group_name, indices_and_len in group_indices_and_len.items():
            group_indices_and_len[group_name] = indices_and_len[
                : len(indices_and_len) // global_bsz_acc * global_bsz_acc
            ]

        if self.shuffle:
            for _, indices_and_len in group_indices_and_len.items():
                rng.shuffle(indices_and_len)

            group_indices = {}
            if self.length_clustering:
                for group_name, indices_and_len in group_indices_and_len.items():
                    # shuffle before sorting is still important when lots of samples share the same length
                    indices_and_len.sort(key=lambda x: x[1])
                    group_indices[group_name] = [_[0] for _ in indices_and_len]

                # option1: shuffle among neighboring items
                for group_name, indices in group_indices.items():
                    result = []
                    for pos in range(0, len(indices), global_batch_size * 500):
                        sublist = indices[pos: pos + global_batch_size * 500]
                        rng.shuffle(sublist)
                        result.extend(sublist)
                    group_indices[group_name] = result
            else:
                for group_name, indices_and_len in group_indices_and_len.items():
                    group_indices[group_name] = [_[0] for _ in indices_and_len]

            del group_indices_and_len

            if self.allow_mixed_task_among_acc:
                global_batched_indices = [
                    indices[i: i + global_batch_size]
                    for group_name, indices in group_indices.items()
                    for i in range(0, len(indices), global_batch_size)
                ]
            else:
                global_batched_indices = []
                for _, indices in group_indices.items():
                    group_batched_indices = [indices[i: i + global_batch_size] for i in range(0, len(indices), global_batch_size)]
                    rng.shuffle(group_batched_indices)
                    group_batched_indices = [
                        sum(group_batched_indices[i: i + self.acc_grad], start=[])
                        for i in range(0, len(group_batched_indices), self.acc_grad)
                    ]
                    global_batched_indices.extend(group_batched_indices)
            rng.shuffle(global_batched_indices)
            indices = [_
                for batch_indices in global_batched_indices
                for _ in batch_indices
            ]
        else:
            indices = indices[:self.total_size]

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), global_batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        if len(own_indices) != self.num_samples:
            raise AssertionError(f"The length of own_indices {len(own_indices)} should be equal to num_samples {self.num_samples}!")

        self.epoch += 1
        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples
