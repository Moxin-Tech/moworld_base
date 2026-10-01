import random
from typing import List, Tuple
import os
import math
from multiprocessing import Pool
from abc import abstractmethod

from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from mindspeed_mm.data.data_utils.multimodal_image_video_preprocess import find_closest_aspect_ratio


class BucketManager:
    """
    This class handles the organization and processing of images based on their aspect bucket range
    into various buckets, and then distributes these buckets into packages that can be used for
    further processing or training. The class manages both sharding and non-sharding modes, ensuring
    efficient use of data in distributed or single-machine setups.

    Similar to a normal implementation of a distributed sampler, except the implementation is at
    the batch sampler level. This allows wrapping of arbitrary data samplers (sequential, random,
    WeightedRandomSampler, etc.) with this batch sampler.
    """
    class Bucket:
        """
        Represents a single bucket that stores samples (images) grouped by their aspect bucket range.
        Each bucket may have multiple groups, and each group holds a list of sample indices.
        The class provides functionality to shuffle the samples within each group and fetch samples
        based on their indices.

        This is similar to a normal bucket used for data grouping in distributed systems, where
        data is partitioned and accessed by different workers based on the group and bucket range.
        """
        def __init__(self, bucket_range: Tuple[int, int], num_groups: int = 1):
            self.bucket_range = bucket_range
            self.num_groups = num_groups
            self.samples = [[] for _ in range(num_groups)]
            self.lengths = [0] * num_groups
            self.index_lists = [None] * num_groups
            self.seed = 42

        def __repr__(self):
            return f"Bucket(bucket_range={self.bucket_range}, num_groups={self.num_groups}, lengths={self.lengths})"

        def add_sample(self, group_id: int, sample_idx: int):
            self.samples[group_id].append(sample_idx)
            self.lengths[group_id] += 1

        def refresh_index(self, shuffle: bool = True, seed: int = None):
            self.seed = seed if seed is not None else self.seed
            for group_id in range(self.num_groups):
                self.index_lists[group_id] = list(range(self.lengths[group_id]))
                if shuffle:
                    random.seed(self.seed)
                    random.shuffle(self.index_lists[group_id])

        def get_sample_by_idx(self, group_id: int, idx: int):
            if self.index_lists[group_id] is None:
                raise RuntimeError("Index list not initialized. Call `refresh_index()` before accessing samples.")
            if idx >= self.lengths[group_id]:
                raise IndexError(f"Index {idx} out of range for group {group_id}.")
            return self.samples[group_id][self.index_lists[group_id][idx]]


    class Package:
        """
        Represents a data package that contains samples from one or more buckets. Packages can either
        be mixed (containing samples from multiple buckets) or single (containing samples from only one bucket).
        The package is used to organize and handle data when processing batches during training.

        This class is an abstraction that allows for batching of data, ensuring that each batch either
        contains data from a single bucket or from a mixture of multiple buckets, depending on whether
        the `mixed` flag is set. It provides flexibility for data processing and shuffling during training.
        """
        def __init__(self, mixed: bool = False, num_groups: int = 1):
            self.samples = []
            self.mixed = mixed
            self.num_groups = num_groups
            self.bucket_range = (0, 0)

        def __str__(self):
            return f"Package(mixed={self.mixed}, bucket_range={self.bucket_range}, number_of_samples={len(self.samples)})"

        def add_samples_single_bucket(self, bucket_range: Tuple[int, int], samples: List[Tuple[int, int]]):
            """Add a single bucket of samples. This parameter is used only when mixed is set to False."""
            if self.mixed:
                raise ValueError("Cannot use this method when mixed=True.")
            if not samples:
                raise ValueError("Samples cannot be empty.")
            self.bucket_range = bucket_range
            for sample in samples:
                self.samples.append((bucket_range, sample))

        def add_mixed_samples(self, samples: List[Tuple[Tuple[int, int], int, int]]):
            """Add a sample of mixed packets. This parameter is used only when mixed is set to True."""
            if not self.mixed:
                raise ValueError("Cannot use this method when mixed=False.")
            if not samples:
                raise ValueError("Samples cannot be empty.")
            for sample in samples:
                if isinstance(sample, tuple) and len(sample) == 3:
                    self.samples.append(sample)
                else:
                    raise ValueError(f"Each sample must be a tuple (bucket_range, group_id, idx) when mixed=True. sample: {sample} ")

        def get_samples(self, buckets) -> List[List[int]]:
            """Obtains all samples in the packet based on the bucket information and allocates samples by group."""
            if not self.samples:
                return []
            sample_data = [[] for _ in range(self.num_groups)]
            if self.mixed:
                # Traverse samples to obtain data in the corresponding bucket.
                for bucket_range, group_id, idx in self.samples:
                    for bucket in buckets:
                        if bucket.bucket_range == bucket_range:
                            sample = bucket.get_sample_by_idx(group_id, idx)
                            sample_data[group_id].append(sample)
            else:
                for bucket_range, cursample in self.samples:
                    for bucket in buckets:
                        if bucket.bucket_range == bucket_range:
                            sample = bucket.get_sample_by_idx(cursample[0], cursample[1])
                            sample_data[cursample[0]].append(sample)
            return sample_data

    def __init__(
        self,
        image_size: int = 448,
        batch_size: int = 128,
        sharding: bool = False,
        num_replicas: int = None,
        keep_remainder: bool = True,
        rank: int = 0,
        processes_num: int = 8,
        global_batch_size: int = 128,
        priority_mode: str = "data_bucketing_img"
    ):
        """
        Initializes the BucketManager class, which is responsible for organizing image samples into
        buckets based on their aspect bucket range. The class can operate in both sharding and non-sharding
        modes, and it efficiently distributes data samples into batches or packages.

        This is the entry point for setting up the bucket management system, where it configures how
        data will be grouped, batched, and potentially distributed across multiple workers.
        """
        if num_replicas is None:
            raise ValueError("num_groups is required.")
        self.image_size = image_size
        self.sharding = sharding
        self.keep_remainder = keep_remainder
        self.rank = rank
        self.processes_num = processes_num
        self.num_replicas = num_replicas
        self.global_batch_size = global_batch_size
        self.priority_mode = priority_mode
        if sharding:
            self.batch_size = batch_size
            self.num_groups = num_replicas
        else:
            self.batch_size = batch_size * num_replicas
            self.num_groups = 1
        self.image_info = {}
        self.bucket_info = {}
        self.total_packages = []
        self.final_results_dict = {}
        self.buckets = None

    @abstractmethod
    def create_buckets(self) -> List[Bucket]:
        """Generate buckets based on the target aspect bucket range."""
        return []

    @abstractmethod
    def get_img_fullpath(self, idx: int, condataset):
        """Obtains the full path of the image based on the index."""
        return ""

    @abstractmethod
    def load_image_and_get_dimensions(self, image_fullpath):
        """Loads an image and returns its width and height"""
        return (0, 0)

    @abstractmethod
    def create_sorting_dictionary(self, dataset):
        """Sort by image data size and return to the dictionary {idx, image_token}."""
        return {}

    @abstractmethod
    def allocate_data_to_bucket(self, dataset):
        """All data is allocated to different buckets based on rules."""
        return ""

    @staticmethod
    def suggest_thread_count(dataset):
        cpu_count = os.cpu_count()
        data_size = len(dataset)
        processes_num = 8
        data_size_threshold = 50000
        if cpu_count is None:
            return processes_num

        # Rules for setting the number of threads:
        # 1. Use 1/12 of the total CPU cores as the number of threads when the data volume is less than 50,000;
        # 2. Use 1/6 of the total CPU cores as the number of threads when the data volume is 50,000 or more;
        # 3. The minimum number of threads is 8.
        if data_size < data_size_threshold:
            scale_factor = 1 / 12
        else:
            scale_factor = 1 / 6
        processes_num = max(8, int(cpu_count * scale_factor))
        return processes_num

    def handle_data_reordering_img(self, condataset):
        sorting_dict = self.create_sorting_dictionary(condataset)
        if sorting_dict is None:
            raise ValueError("Sorting dictionary creation failed")
        return sorting_dict

    def handle_data_bucketing_img(self, condataset):
        self.buckets = self.create_buckets()
        if not self.buckets:
            raise ValueError("Bucket creation failed")

        try:
            self.allocate_data_to_bucket(condataset)
        except Exception as e:
            raise ValueError(f"Data allocation to buckets failed: {str(e)}") from e
        return self.buckets

    def group_by_bucket(self, condataset):
        priority_handlers = {
            "data_reordering_img": self.handle_data_reordering_img,
            "data_bucketing_img": self.handle_data_bucketing_img,
        }

        handler = priority_handlers.get(self.priority_mode)
        if handler is None:
            raise ValueError(f"Unknown priority mode: {self.priority_mode}")

        return handler(condataset)

    @staticmethod
    def generate_index_by_gbs(idx_range, final_results_dict, global_batch_size, num_replicas):
        # Store the sorted index
        sorted_indices = []

        sort_batch_size = int(global_batch_size / num_replicas)
        for i in range(0, len(idx_range), sort_batch_size):
            batch_indices = idx_range[i:i + sort_batch_size]

            # Retrieve the corresponding values from result_dict based on batch_indices,
            # combine them into tuples in the form of (idx, value).
            batch_data = [(idx, final_results_dict.get(idx)) for idx in batch_indices]

            batch_data.sort(key=lambda x: x[1])

            # Extract the sorted indices and add them to sorted_indices
            sorted_batch_indices = [idx for idx, _ in batch_data]

            sorted_indices.extend(sorted_batch_indices)  # Merge into a one-dimensional list
        return sorted_indices

    def print_buckets(self):
        """Print the distribution of samples in the barrel, and print the number of samples in each group by group."""
        if self.rank != 0:
            return
        total = 0
        for bucket in self.buckets:
            print(f"Bucket (bucket_range {bucket.bucket_range}): ")
            bucket_num = 0
            for group_id in range(bucket.num_groups):
                group_sample_count = len(bucket.samples[group_id])
                print(f"  Group {group_id}: {group_sample_count} samples", end="  |  ")
                bucket_num += group_sample_count
                total += group_sample_count
            print(f"total num: {bucket_num}\n")
        print(f"Total samples across all buckets: {total}")

    def get_package_by_bucket(self, cur_bucket) -> List[Package]:
        """Generate data packets based on the group information in each bucket based on the specified bucket (cur_bucket)."""
        cur_packages = []
        min_length = min(cur_bucket.lengths)
        num_package = min_length // self.batch_size

        for package_index in range(num_package):
            package = BucketManager.Package(mixed=False, num_groups=self.num_groups)
            for group_id in range(cur_bucket.num_groups):
                startX = package_index * self.batch_size
                samples_range = list(range(startX, startX + self.batch_size))
                samples = [(group_id, sample) for sample in samples_range]
                package.add_samples_single_bucket(cur_bucket.bucket_range, samples)
            cur_packages.append(package)
        return cur_packages

    def create_package_list(self) -> List[Package]:
        """Create a data packet list based on the group size in the bucket."""
        total_packages = []
        remainder_data = [[] for _ in range(self.num_groups)]

        for bucket in self.buckets:
            cur_packages = self.get_package_by_bucket(bucket)
            total_packages.extend(cur_packages)

        for bucket in self.buckets:
            start_batch = min(bucket.lengths) // self.batch_size
            for group_id in range(bucket.num_groups):
                startX = self.batch_size * start_batch
                remaining_samples = list(range(startX, bucket.lengths[group_id]))
                for sample in remaining_samples:
                    remainder_data[group_id].append((bucket.bucket_range, group_id, sample))

        min_length = min(len(remainder_data[i]) for i in range(self.num_groups))

        cur_packages = []
        num_package = min_length // self.batch_size
        for package_index in range(num_package):
            package = BucketManager.Package(mixed=True, num_groups=self.num_groups)
            samples = []
            for group_id in range(self.num_groups):
                startX = package_index * self.batch_size
                sample = remainder_data[group_id][startX:startX + self.batch_size]
                samples.extend(sample)
            package.add_mixed_samples(samples)
            cur_packages.append(package)
        total_packages.extend(cur_packages)
        return total_packages

    def generate_index(self, shuffle=True, seed=42) -> List[int]:
        """Regenerate the index of the data group from the package."""
        total_packages = self.total_packages
        index_packages = list(range(len(total_packages)))

        if shuffle:
            random.seed(seed)
            random.shuffle(index_packages)

        for bucket in self.buckets:
            bucket.refresh_index(shuffle, seed)

        index_list = []
        if self.sharding:
            group_list = [[] for _ in range(self.num_groups)]
            for idx in index_packages:
                cur_list = total_packages[idx].get_samples(self.buckets)
                for group_id in range(self.num_groups):
                    group_list[group_id].extend(cur_list[group_id])
            for idx in range(self.num_groups):
                index_list.extend(group_list[idx])
        else:
            for idx in index_packages:
                cur_list = total_packages[idx].get_samples(self.buckets)
                index_list.extend(cur_list[0])

        return index_list


class BucketManager_qwen2vl(BucketManager):
    def __init__(
        self,
        image_size: int = 512,  # Initialize bucket parameters, default values from config file and can be modified on demand.
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
        patch_size: int = 14,
        merge_size: int = 2,
        batch_size: int = 1,
        sharding: bool = False,
        num_replicas: int = 1,
        keep_remainder: bool = False,
        rank: int = 1,
        bucket_interval: int = 200,  # Token interval for bucket
        global_batch_size: int = 128,
        priority_mode: str = "data_bucketing_img"
    ):
        self.bucket_interval = bucket_interval
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.merge_size = merge_size

        self.factor = self.patch_size * self.merge_size

        super().__init__(
            image_size=image_size,
            batch_size=batch_size,
            sharding=sharding,
            num_replicas=num_replicas,
            keep_remainder=keep_remainder,
            rank=rank,
            global_batch_size=global_batch_size,
            priority_mode=priority_mode,
        )

    def create_buckets(self):
        merged_buckets = {}
        # Calculate tokens by image size
        resize_image_resolution = round(self.image_size / self.factor) * self.factor
        max_tokens = int(resize_image_resolution / self.patch_size) ** 2

        # Determine the bucket range based on bucket_interval
        bucket_range = tuple(range(0, max_tokens + 1, self.bucket_interval))

        # Ensure max_tokens is included in bucket_range
        if max_tokens % self.bucket_interval != 0:
            bucket_range = bucket_range + (max_tokens,)

        # Convert bucket_range to (start, end) interval pairs
        result_bucket_range = list(zip(bucket_range[:-1], bucket_range[1:]))

        for bucket_range in result_bucket_range:
            sorted_bucket_range = tuple(sorted(bucket_range))
            if bucket_range not in merged_buckets:
                merged_buckets[sorted_bucket_range] = BucketManager.Bucket(sorted_bucket_range, self.num_groups)
        return list(merged_buckets.values())

    def get_img_fullpath(self, idx, condataset):
        datasets = condataset
        sample = datasets[idx]
        image_path = sample['images']
        if image_path is None:
            return []
        return image_path

    def calculated_w_h(self, width, height):
        image_resolution = self.image_size ** 2
        if (width * height) > image_resolution:
            resize_factor = math.sqrt(image_resolution / (width * height))
            width, height = int(width * resize_factor), int(height * resize_factor)

        if min(width, height) < self.factor:
            width, height = max(width, self.factor), max(height, self.factor)

        if width / height > 200:
            width, height = height * 180, height

        if height / width > 200:
            width, height = width, width * 180

        return width, height

    def load_image_and_get_dimensions(self, image_fullpath):
        # Reuse Qwen2VL's image processing logic
        try:
            with Image.open(image_fullpath) as img:
                width, height = img.width, img.height
                preprocessed_width, preprocessed_height = self.calculated_w_h(width, height)
                resize_height, resize_width = smart_resize(preprocessed_height, preprocessed_width, self.factor, self.min_pixels, self.max_pixels)
                return resize_height, resize_width
        except Exception as e:
            print(f"Error loading image {image_fullpath}: {e}")
            return (0, 0)

    def process_bucket_data(self, idx, condataset):
        full_image_path = self.get_img_fullpath(idx, condataset)
        infos = []
        total_tokens = 0
        for image_path in full_image_path:
            try:
                width, height = self.load_image_and_get_dimensions(image_path)
                width_patch = width / self.patch_size
                height_patch = height / self.patch_size
                tokens = width_patch * height_patch
                total_tokens += tokens
                infos.append({
                    'path': image_path,
                    'width': width,
                    'height': height,
                    'width_patch': width_patch,
                    'height_patch': height_patch,
                    'tokens': tokens
                })
            except Exception as e:
                print(f"Error processing sample {idx}, image {image_path}: {e}")
                infos.append({
                    'path': image_path,
                    'tokens': 0
                })
        return idx, total_tokens, infos

    def process_calculate_images_token(self, idx, condataset):
        full_image_path = self.get_img_fullpath(idx, condataset)
        result_dict = {}
        total_tokens = 0
        for path in full_image_path:
            try:
                width, height = self.load_image_and_get_dimensions(path)
                width_patch = width / self.patch_size
                height_patch = height / self.patch_size
                tokens = width_patch * height_patch
                total_tokens += tokens
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
        result_dict[idx] = total_tokens
        return result_dict

    def create_sorting_dictionary(self, dataset):
        indices = [i for i in range(len(dataset))]
        self.processes_num = self.suggest_thread_count(dataset)
        with Pool(processes=self.processes_num) as pool:
            results = pool.starmap(self.process_calculate_images_token, [(idx, dataset) for idx in indices])
        # Merge all returned dictionaries
        for result in results:
            self.final_results_dict.update(result)
        return self.final_results_dict

    def allocate_data_to_bucket(self, condataset):
        group_length = len(condataset) // self.num_groups
        last_patch = len(condataset) % group_length
        indices = [i for i in range(len(condataset) - last_patch)]
        self.processes_num = self.suggest_thread_count(condataset)
        with Pool(processes=self.processes_num) as pool:
            results = pool.starmap(self.process_bucket_data, [(idx, condataset) for idx in indices])

        for idx, total_tokens, _ in results:
            bfind = False
            for bucket in self.buckets:
                if total_tokens >= bucket.bucket_range[0] and total_tokens < bucket.bucket_range[1]:
                    group_id = idx // group_length
                    bucket.add_sample(group_id, idx)
                    self.image_info[idx] = total_tokens
                    self.bucket_info[idx] = bucket.bucket_range
                    bfind = True
                    break

            # Add to the last bucket if no matching bucket found
            if not bfind:
                last_bucket = self.buckets[-1]
                group_id = idx // group_length
                last_bucket.add_sample(group_id, idx)
                self.image_info[idx] = total_tokens
                self.bucket_info[idx] = last_bucket.bucket_range

        self.total_packages = self.create_package_list()
        self.print_buckets()


class BucketManager_internvl2(BucketManager):
    def __init__(
        self,
        image_size: int = 512,
        min_num: int = 1,
        max_num: int = 6,
        batch_size: int = 1,
        sharding: bool = False,
        num_replicas: int = 1,
        keep_remainder: bool = False,
        rank: int = 1,
        global_batch_size: int = 128,
        priority_mode: str = "data_bucketing_img"
    ):
        self.min_num = min_num
        self.max_num = max_num
        self.target_ratios = set()

        super().__init__(
            image_size=image_size,
            batch_size=batch_size,
            sharding=sharding,
            num_replicas=num_replicas,
            keep_remainder=keep_remainder,
            rank=rank,
            global_batch_size=global_batch_size,
            priority_mode=priority_mode,
        )

    def create_buckets(self):
        # Calculate all possible target aspect ratios (target_ratios) within the range [min_num, max_num]
        for n in range(self.min_num, self.max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if self.min_num <= i * j <= self.max_num:
                        self.target_ratios.add((i, j))

        self.target_ratios = sorted(self.target_ratios, key=lambda x: x[0] * x[1])
        sorted_target_ratios = sorted(self.target_ratios, key=lambda ratio: (ratio[0], ratio[1]))

        merged_buckets = {}
        for ratio in sorted_target_ratios:
            # Ensure consistent order, e.g., (3,4) and (4,3) are treated as the same bucket
            sorted_ratio = tuple(sorted(ratio))
            if sorted_ratio not in merged_buckets:
                merged_buckets[sorted_ratio] = BucketManager.Bucket(ratio, self.num_groups)
        return list(merged_buckets.values())

    def get_dataset_info(self, idx: int, datasets):
        """Obtains the index of a dataset and its location in the dataset."""
        dataset_idx = 0
        index_in_dataset = idx
        for i, sub_dataset in enumerate(datasets):
            if index_in_dataset < len(sub_dataset):
                dataset_idx = i
                break
            index_in_dataset -= len(sub_dataset)
        return dataset_idx, index_in_dataset

    def get_img_fullpath(self, idx: int, condataset):
        datasets = condataset.datasets
        dataset_idx, index_in_dataset = self.get_dataset_info(idx, datasets)
        sample = datasets[dataset_idx].data_samples[index_in_dataset]
        image_path = sample['image']
        if image_path is None:
            return None
        return os.path.join(datasets[dataset_idx].data_folder, image_path)

    def load_image_and_get_dimensions(self, image_fullpath):
        try:
            with Image.open(image_fullpath) as img:
                return img.size
        except (OSError, ValueError) as e:
            return None

    def process_bucket_data(self, idx, condataset):
        full_image_path = self.get_img_fullpath(idx, condataset)
        try:
            width, height = self.load_image_and_get_dimensions(full_image_path)
            aspect_ratio = width / height
            closest_ratio = find_closest_aspect_ratio(aspect_ratio, self.target_ratios, width, height, self.image_size)
            sorted_ratio = tuple(sorted(closest_ratio))
            return idx, width, height, closest_ratio, sorted_ratio
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return idx, None, None, None, None

    def process_calculate_images_num(self, idx, condataset):
        full_image_path = self.get_img_fullpath(idx, condataset)
        result_dict = {}
        try:
            width, height = self.load_image_and_get_dimensions(full_image_path)
            aspect_ratio = width / height
            closest_ratio = find_closest_aspect_ratio(aspect_ratio, self.target_ratios, width, height, self.image_size)
            result_dict[idx] = closest_ratio[0] * closest_ratio[1]
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            result_dict[idx] = None
        return result_dict

    def create_sorting_dictionary(self, dataset):
        # Calculate all possible target aspect ratios (target_ratios) within the range [min_num, max_num]
        for n in range(self.min_num, self.max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if self.min_num <= i * j <= self.max_num:
                        self.target_ratios.add((i, j))
        indices = [i for i in range(len(dataset))]
        self.processes_num = self.suggest_thread_count(dataset)
        with Pool(processes=self.processes_num) as pool:
            results = pool.starmap(self.process_calculate_images_num, [(idx, dataset) for idx in indices])

        # Merge all returned dictionaries
        for result in results:
            self.final_results_dict.update(result)
        return self.final_results_dict

    def allocate_data_to_bucket(self, condataset):
        group_length = len(condataset) // self.num_groups
        last_patch = len(condataset) % group_length
        indices = [i for i in range(len(condataset) - last_patch)]
        self.processes_num = self.suggest_thread_count(condataset)
        with Pool(processes=self.processes_num) as pool:
            results = pool.starmap(self.process_bucket_data, [(idx, condataset) for idx in indices])

        for idx, width, height, closest_ratio, sorted_ratio in results:
            if width is None or height is None:
                continue

            bfind = False
            for bucket in self.buckets:
                if bucket.bucket_range == sorted_ratio:
                    group_id = idx // group_length
                    bucket.add_sample(group_id, idx)
                    self.image_info[idx] = (width, height)
                    self.bucket_info[idx] = closest_ratio
                    bfind = True
                    break
            if not bfind:
                print(f"Warning: Could not find a suitable bucket for sample {idx}")

        self.total_packages = self.create_package_list()
        self.print_buckets()
