import os
import math
import random
from collections import Counter
from typing import Dict, Optional, Type, List
from abc import ABC, abstractmethod

import numpy as np
import torch
import torchvision.transforms as TT
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from megatron.core import mpu
from mindspeed_mm.data.data_utils.data_transform import (
    calculate_centered_alignment,
    TemporalRandomCrop,
    maxhwresize
)
from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.utils import (
    get_value_from_args,
    cal_gradient_accumulation_size,
    DataStats,
    VID_EXTENSIONS,
    IMG_EXTENSIONS,
    TENSOR_EXTENSIONS
)
from mindspeed_mm.utils.utils import Registry


class VideoProcessor:
    """
    Factory class for creating video processor instances
    """
    @staticmethod
    def create(video_processor_type=None, **kwargs) -> "AbstractVideoProcessor":
        """
        Initialize with specified video processor type

        Args:
            video_processor_type: Registered video backend type (e.g., 'opensora_video_processor', 'cogvideox_video_processor', 'opensoraplan_video_processor')
        """
        processor_cls = Registry.get_class(video_processor_type)
        return processor_cls(**kwargs)


class AbstractVideoProcessor(ABC):
    """Base class for video processing pipelines

    Attributes:
        num_frames (int): Number of frames to sample from video
        frame_interval (int): Interval between sampled frames
        train_pipeline (callable): Data augmentation pipeline
    """

    def __init__(
        self,
        num_frames: int = 16,
        frame_interval: int = 1,
        train_pipeline: callable = None,
    ):
        """Initialize common parameters for all processors"""
        # Core sampling parameters
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.train_pipeline = train_pipeline

        # Shared components
        self.video_transforms = None  # Will be initialized per video
        self.temporal_sample = TemporalRandomCrop(num_frames * frame_interval)


    @abstractmethod
    def __call__(self, vframes, **kwargs):
        """Process video frames.

        Args:
            vframes: Input video frames
            kwargs: Additional processing parameters

        Returns:
            Processed video data
        """
        ...

    @abstractmethod
    def select_valid_data(self, data_samples):
        """Filter valid data samples from input

        Args:
            data_samples: Input data samples to be filtered

        Returns:
            Filtered data samples. Default implementation returns original input.
        """
        return data_samples


@Registry.register
class OpensoraVideoProcessor(AbstractVideoProcessor):
    """Opensora video processing pipeline with temporal sampling and spatial transforms"""

    def __call__(self, vframes, num_frames=None, frame_interval=None, image_size=None, **kwargs):
        """Process video frames through standard pipeline

        Args:
            vframes: Input video frames container
            num_frames: Override default number of frames
            frame_interval: Override default frame interval
            image_size: Target output dimensions

        Returns:
            torch.Tensor: Processed tensor in CTHW format
        """
        # Initialize transforms based on input size
        if image_size:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                   image_size=image_size)
        else:
            self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline)

        # Temporal sampling logic
        total_frames = vframes.get_len()
        if num_frames:  # Dynamic parameter adjustment
            self.num_frames = num_frames
            self.temporal_sample = TemporalRandomCrop(num_frames * (frame_interval or self.frame_interval))

        # Generate sampling window
        start, end = self.temporal_sample(total_frames)
        if end - start < self.num_frames:
            raise ValueError(f"Insufficient frames: {end-start} < {self.num_frames}")

        # Linear sampling within window
        indices = np.linspace(start, end - 1, self.num_frames, dtype=int)
        video = vframes.get_batch(indices)  # TCHW format

        # Apply transforms and permute dimensions
        video = self.video_transforms(video)
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return video

    def select_valid_data(self, data_samples):
        return super().select_valid_data(data_samples)


@Registry.register
class CogVideoXProcessor(AbstractVideoProcessor):
    """Specialized processor for CogVideoX model

    Args:
        skip_frame_num (int): Number of initial frames to skip (default: 0)
        train_fps (float): Target frames per second for processing
        max_height (int): Maximum allowed frame height (default: 480)
        max_width (int): Maximum allowed frame width (default: 640)
        **base_args: Inherited parameters from AbstractVideoProcessor
    """

    def __init__(
        self,
        skip_frame_num: int = 0,
        train_fps: float = None,
        max_height: int = 480,
        max_width: int = 640,
        **base_args
    ):
        """Initialize CogVideoX specific parameters"""
        super().__init__(**base_args)
        self.skip_frame_num = skip_frame_num
        self.train_fps = train_fps
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, vframes, **kwargs):
        """Process video following CogVideoX's temporal specifications"""

        # Calculate actual frame characteristics
        actual_fps = vframes.get_video_fps()
        ori_video_len = vframes.get_len()

        # Adaptive sampling strategy
        if ori_video_len / actual_fps * self.train_fps > self.num_frames:
            # High FPS video processing
            num_frames = self.num_frames
            start = int(self.skip_frame_num)
            end = int(start + num_frames / self.train_fps * actual_fps)
            end_safety = min(int(start + num_frames / self.train_fps * actual_fps), int(ori_video_len))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            tensor_frames = vframes.get_batch(np.arange(start, end_safety)) # T C H W
            tensor_frames = tensor_frames[torch.tensor((indices - start).tolist())]
        else:
            if ori_video_len > self.num_frames:
                num_frames = self.num_frames
                start = int(self.skip_frame_num)
                end = int(ori_video_len - self.skip_frame_num)
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                tensor_frames = vframes.get_batch(np.arange(start, end)) # T C H W
                tensor_frames = tensor_frames[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frame_num)
                end = int(ori_video_len - self.skip_frame_num)
                # 3D VAE requires the number of frames to be 4k+1
                num_frames = nearest_smaller_4k_plus_1(end - start)
                end = int(start + num_frames)
                tensor_frames = vframes.get_batch(np.arange(start, end))  # T C H W

        # the len of indices may be less than num_frames, due to round error
        tensor_frames = self._pad_last_frame(
            tensor_frames, self.num_frames
        )
        tensor_frames = self._resize_for_rectangle_crop(tensor_frames, [self.max_height, self.max_width],
                                                  reshape_mode="center")
        # Normalization to [-1, 1] range
        tensor_frames = (tensor_frames - 127.5) / 127.5
        return tensor_frames

    def _pad_last_frame(self, tensor, num_frames):
        # T, H, W, C
        if len(tensor) < num_frames:
            pad_length = num_frames - len(tensor)
            # Use the last frame to pad instead of zero
            last_frame = tensor[-1]
            pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
            padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
            return padded_tensor
        else:
            return tensor[:num_frames]

    def _resize_for_rectangle_crop(self, arr, image_size, reshape_mode="random"):
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            arr = resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
                antialias=None
            )
        else:
            arr = resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
                antialias=None
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        if reshape_mode == "random" or reshape_mode == "none":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def select_valid_data(self, data_samples):
        return super().select_valid_data(data_samples)


@Registry.register
class OpensoraplanVideoProcessor(AbstractVideoProcessor):
    """Specialized processor for Opensoraplan model

    Args:
        min_num_frames (int): Minimum required frames (default: 29)
        train_fps (float): Target frames per second for processing
        auto_interval (bool): Auto-calculate frame interval (default: True)
        speed_factor (float): Playback speed modifier (default: 1.0)
        drop_short_ratio (float): Ratio to drop short clips (default: 1.0)
        force_resolution (bool): Enforce resolution constraints (default: True)
        max_height (int): Maximum processing height (default: 480)
        max_width (int): Maximum processing width (default: 640)
        max_hxw (int): Maximum height×width product
        min_hxw (int): Minimum height×width product
        hw_stride (int): Height/width alignment stride (default: 32)
        hw_aspect_thr (float): Aspect ratio threshold (default: 1.5)
        truncate_t_by_sp (bool): Whether to truncate dimension t to a multiple of sp_size (default: True)
        vae_scale_factor (list): VAE down sample scale factor (default: [4, 8, 8]])
        train_sp_batch_size (int): Sequence parallel batch size (default: 1)
        seed (int): Random seed (default: 42)
        **base_args: Inherited parameters from AbstractVideoProcessor
    """

    def __init__(
        self,
        min_num_frames: int = 29,
        train_fps: float = 24,
        auto_interval: bool = True,
        speed_factor: float = 1.0,
        drop_short_ratio: float = 1.0,
        force_resolution: bool = True,
        max_height: int = 480,
        max_width: int = 640,
        max_hxw: int = None,
        min_hxw: int = None,
        hw_stride: int = 32,
        hw_aspect_thr: float = 1.5,
        truncate_t_by_sp: bool = True,
        vae_scale_factor: Optional[List[int]] = None,
        train_sp_batch_size: int = 1,
        seed: int = 42,
        **base_args
    ):
        """Initialize OpenSoraPlan specific parameters"""
        super().__init__(**base_args)
        if vae_scale_factor is None:
            vae_scale_factor = [4, 8, 8]
        self.train_fps = train_fps
        self.auto_interval = auto_interval
        self.speed_factor = speed_factor
        self.drop_short_ratio = drop_short_ratio

        # Spatial parameters
        self.force_resolution = force_resolution
        self.max_height = max_height
        self.max_width = max_width
        self.max_hxw = max_hxw
        self.min_hxw = min_hxw
        self.hw_stride = hw_stride
        self.hw_aspect_thr = hw_aspect_thr
        self.hw_aspect_thr = 1.5 if self.hw_aspect_thr == 0 else self.hw_aspect_thr
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.transform_size = {
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_hxw": self.max_hxw,
            "min_hxw": self.min_hxw
        }

        # Training configuration
        self.ae_stride_t = vae_scale_factor[0]
        from mindspeed_mm.utils.dpcp_utils import get_max_cp_size
        self.sp_size = get_max_cp_size() if truncate_t_by_sp else 1  # For sequence parallel
        self.train_sp_batch_size = train_sp_batch_size
        self.gradient_accumulation_size = cal_gradient_accumulation_size()
        self.batch_size = get_value_from_args("micro_batch_size")
        self.global_batch_size = get_value_from_args("global_batch_size")
        self.min_num_frames = min_num_frames

        # Randomness control
        self.generator = torch.Generator().manual_seed(seed)

        self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                                                    transform_size=self.transform_size)

    def __call__(
        self,
        vframes,
        sample_num_frames=13,
        start_frame_idx=0,
        num_frames=-1,
        crop=(None, None, None, None),
        **kwargs
    ):
        """Process video frames with temporal speed adjustment and spatial validation.

        Args:
            vframes: Video frames container object with frame access methods
            sample_num_frames: Expected number of output frames for validation
            start_frame_idx: Starting index for frame sampling
            num_frames: Total available frames (-1 = auto-detect from vframes)
            crop: Spatial crop coordinates (start_x, end_x, start_y, end_y)

        Returns:
            torch.Tensor: Processed video tensor in CTHW format

        Raises:
            IndexError: When video is too short for required processing
            ValueError: When sampled frames mismatch predefined count
            AssertionError: When aspect ratio validation fails
        """
        # Frame count and FPS initialization
        total_frames = vframes.get_len() if num_frames == -1 else num_frames
        fps = vframes.get_video_fps() if vframes.get_video_fps() > 0 else 30.0
        s_x, e_x, s_y, e_y = crop

        # Temporal sampling interval calculation
        if self.auto_interval:
            # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
            frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        else:
            frame_interval = self.frame_interval

        # Generate initial frame indices
        frame_indices = np.arange(start_frame_idx, start_frame_idx + total_frames, frame_interval).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx + total_frames]

        # speed up through temporal subsampling
        max_speed_factor = len(frame_indices) / self.num_frames
        if self.speed_factor > 1 and max_speed_factor > 1:
            speed_factor = min(self.speed_factor, max_speed_factor)
            target_frame_count = int(len(frame_indices) / speed_factor)
            speed_frame_idx = np.linspace(
                0, len(frame_indices) - 1, target_frame_count, dtype=int
            )
            frame_indices = frame_indices[speed_frame_idx]

        # Random temporal cropping for long sequences
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )
        if end_frame_idx == -1:  # too short that can not be encoded exactly by videovae
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )
        frame_indices = frame_indices[:end_frame_idx]

        # Frame validation
        if sample_num_frames != len(frame_indices):
            raise ValueError(
                f"sample_num_frames ({sample_num_frames}) is not equal with frame_indices ({len(frame_indices)})"
            )
        if len(frame_indices) < self.num_frames and self.drop_short_ratio >= 1:
            raise IndexError(
                f"video has {total_frames} frames, but need to sample {len(frame_indices)} frames ({frame_indices})"
            )

        # Frame extraction and processing
        video = vframes.get_batch(frame_indices)  # T C H W
        if s_y is not None:
            video = video[:, :, s_y: e_y, s_x: e_x]

        # Resolution validation
        h, w = video.shape[-2:]
        if self.force_resolution:
            if h / w > 17 / 16 or h / w < 8 / 16:
                raise AssertionError(
                    f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But the video found ratio is {round(h / w, 2)} with the shape of {video.shape}"
                )
        # TCHW -> TCHW
        video = self.video_transforms(video)
        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        return video

    def find_closest_y(self, x, vae_stride_t=4, model_ds_t=1):
        if x < self.min_num_frames:
            return -1
        for y in range(x, self.min_num_frames - 1, -1):
            if (y - 1) % vae_stride_t == 0 and ((y - 1) // vae_stride_t + 1) % model_ds_t == 0:
                # 4, 8: y in [29, 61, 93, 125, 157, 189, 221, 253, 285, 317, 349, 381, 413, 445, 477, 509, ...]
                # 4, 4: y in [29, 45, 61, 77, 93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253, 269, 285, 301, 317, 333, 349, 365, 381, 397, 413, 429, 445, 461, 477, 493, 509, ...]
                # 8, 1: y in [33, 41, 49, 57, 65, 73, 81, 89, 97, 105]
                # 8, 2: y in [41, 57, 73, 89, 105]
                # 8, 4: y in [57, 89]
                # 8, 8: y in [57]
                return y
        return -1

    def select_valid_data(self, data_samples):
        """data filtering

        Args:
            data_samples: List of video caption dictionaries

        Returns:
            valid_samples

        Processing Steps:
        1. Filter invalid entries (missing captions/resolution)
        2. Validate resolution constraints
        3. Calculate temporal sampling indices
        4. Apply quality filters
        5. Collect statistics
        """
        stats = DataStats()
        valid_samples = []
        sample_sizes = []

        for sample in data_samples:
            stats.increment('total_processed')

            if not self._validate_caption(sample, stats):
                continue

            if not self._process_resolution(sample, stats):
                continue

            if not self._process_temporal(sample, stats):
                continue

            self._validate_aesthetic(sample, stats)

            # sample update
            sample_size = f'{len(sample["sample_frame_index"])}x{sample["resolution"]["sample_height"]}x{sample["resolution"]["sample_width"]}'
            sample["sample_size"] = sample_size
            sample_sizes.append(sample_size)
            valid_samples.append(sample)

        valid_samples, sample_sizes = self._apply_final_filters(valid_samples, sample_sizes, stats)

        return valid_samples

    def _validate_caption(self, sample, stats):
        cap = sample.get("cap", None)
        if cap is None:
            stats.increment("no_caption")
            return False
        else:
            return True

    def _process_resolution(self, sample, stats):
        """Handle resolution validation and processing"""
        res_info = sample.get("resolution", {})
        height, width = res_info.get("height", -1), res_info.get("width", -1)
        if height <= 0 or width <= 0:
            stats.increment("no_resolution")
            return False

        # Process resolution
        if not self.force_resolution:
            # Dynamic resolution
            tr_h, tr_w = maxhwresize(height, width, self.max_hxw)
            _, _, sample_h, sample_w = calculate_centered_alignment(tr_h, tr_w, self.hw_stride)

            if sample_h <= 0 or sample_w <= 0:
                stats.increment("resolution_mismatch")
                return False
            if sample_h * sample_w < self.min_hxw:
                stats.increment("resolution_too_small")
                return False

            is_pick = self._filter_resolution(
                sample_h,
                sample_w,
                max_h_div_w_ratio=self.hw_aspect_thr,
                min_h_div_w_ratio=1 / self.hw_aspect_thr
            )
        else:
            # Static resolution
            aspect = self.max_height / self.max_width
            is_pick = self._filter_resolution(
                height,
                width,
                max_h_div_w_ratio=self.hw_aspect_thr * aspect,
                min_h_div_w_ratio=1 / self.hw_aspect_thr * aspect
            )
            sample_h, sample_w = self.max_height, self.max_width

        if not is_pick:
            stats.increment("aspect_mismatch")
            return False

        # Update resolution
        sample["resolution"].update(dict(sample_height=sample_h, sample_width=sample_w))
        return True

    def _filter_resolution(self, h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
        if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
            return True
        return False

    def _process_temporal(self, sample, stats):
        """Handle temporal sampling and frame indices"""
        path = sample["path"]
        ext = os.path.splitext(path)[-1].lower()

        if ext.lower() in VID_EXTENSIONS:  # video
            return self._process_video_temporal(sample, stats)
        elif ext.lower() in IMG_EXTENSIONS:  # image
            sample["sample_frame_index"] = [0]
            sample["sample_num_frames"] = 1
            return True
        elif ext.lower() in TENSOR_EXTENSIONS:  # tensor
            return True
        else:
            raise NameError(
                f"Unknown file extension {path.split('.')[-1]}"
            )

    def _process_video_temporal(self, sample, stats):
        # no fps and duration
        duration = sample.get("duration", None)
        fps = sample.get("fps", None)
        num_frames = sample.get("num_frames", None)
        if fps is None or (duration is None and num_frames is None):
            return False

        sample["num_frames"] = round(fps * duration) if num_frames is None else num_frames
        num_frames = sample["num_frames"]

        if self.auto_interval:
            # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
            frame_interval = 1.0 if abs(fps - self.train_fps) < 0.1 else fps / self.train_fps
        else:
            frame_interval = 1.0

        start_frame_idx = sample.get("cut", [0])[0]
        sample["start_frame_idx"] = start_frame_idx
        frame_indices = np.arange(
            start_frame_idx, start_frame_idx + num_frames, frame_interval
        ).astype(int)
        frame_indices = frame_indices[frame_indices < start_frame_idx + num_frames]

        # comment out it to enable dynamic frames training
        if (
                len(frame_indices) < self.num_frames
                and torch.rand(1, generator=self.generator).item() < self.drop_short_ratio
        ):
            stats.increment('too_short')
            return False

        # too long video will be temporal-crop randomly
        if len(frame_indices) > self.num_frames:
            begin_index, end_index = self.temporal_sample(len(frame_indices))
            frame_indices = frame_indices[begin_index:end_index]

        # to find a suitable end_frame_idx, to ensure we do not need pad video
        end_frame_idx = self.find_closest_y(
            len(frame_indices), vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
        )

        # too short that can not be encoded exactly by videovae
        if end_frame_idx == -1:
            stats.increment('too_short')
            return False

        frame_indices = frame_indices[:end_frame_idx]
        sample["sample_frame_index"] = frame_indices.tolist()
        sample["sample_num_frames"] = len(sample["sample_frame_index"])
        return True

    def _validate_aesthetic(self, sample, stats):
        # no aesthetic
        if sample.get("aesthetic", None) is None or sample.get("aes", None) is None:
            stats.increment("no_aesthetic")
        else:
            stats.collect("aesthetic_score", sample.get("aesthetic", None) or sample.get("aes", None))

    def _apply_final_filters(self, data_samples, sample_sizes, stats):
        """Apply final filters"""
        counter = Counter(sample_sizes)
        filter_major_num = 4 * self.global_batch_size
        data_samples, sample_sizes = zip(*[[i, j] for i, j in zip(data_samples, sample_sizes) if counter[j] >= filter_major_num])
        stats.print_report()
        print(f"{'After filter':<25}: {len(data_samples)}")

        return data_samples, sample_sizes


@Registry.register
class RewardVideoProcessor(AbstractVideoProcessor):
    def __init__(
        self,
        sample_type: str = "uniform",
        sample_nframe: int = None,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 768,
        video_min_pixels: int = 100352,
        video_max_pixels: int = None,
        resized_height: int = None,
        resized_width: int = None,
        **base_args
    ):
        """Initialize Reward specific parameters"""
        super().__init__(**base_args)
        self.sample_type = sample_type
        self.sample_nframe = sample_nframe
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.frame_factor = 2
        self.image_factor = 28

        if sample_type not in ["uniform", "multi_pts"]:
            print("Warning: No valid video sample-type is offering. Whole frames will be used for model input")

    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor


    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor


    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def get_sample_nframes(
        self,
        total_frames: int,
        video_fps: int | float,
    ) -> int:
        """calculate the number of frames for video used for model inputs.
        Args:
            - sample_nframe: the number of frames to extract for model inputs.
            - fps: the fps to extract frames for model inputs.
                - min_frames: the minimum number of frames of the video, only used when fps is provided.
                - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.
        """
        if self.sample_nframe:
            nframes = self.round_by_factor(min(self.sample_nframe, total_frames), self.frame_factor)
        else:
            min_frames = self.ceil_by_factor(self.min_frames, self.frame_factor)
            max_frames = self.floor_by_factor(min(self.max_frames, total_frames), self.frame_factor)
            nframes = total_frames / video_fps * self.fps
            nframes = min(max(nframes, min_frames), max_frames)
            nframes = self.round_by_factor(nframes, self.frame_factor)
        nframes = min(nframes, total_frames)
        if not (self.frame_factor <= nframes and nframes <= total_frames):
            print(f"Warning: nframes should in interval [{self.frame_factor}, {total_frames}], but got {nframes}.")
        return nframes

    def sample_frames(self, vframes):
        total_frames, video_fps = vframes.get_len(), vframes.get_video_fps()
        if self.sample_type == "uniform":
            nframes = self.get_sample_nframes(total_frames=total_frames, video_fps=video_fps)
            idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
            video = vframes.get_batch(idx)
        elif self.sample_type == "multi_pts":
            frames_each_pts = 6
            num_pts = 4
            fps = 8
            nframes = int(total_frames * self.fps // video_fps)
            frames_idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

            start_pt = int(frames_each_pts // 2)
            end_pt = int(nframes - frames_each_pts // 2 - 1)
            pts = torch.linspace(start_pt, end_pt, num_pts).round().long().tolist()
            idx = []
            for pt in pts:
                idx.extend(frames_idx[pt - frames_each_pts // 2: pt + frames_each_pts // 2])
            video = vframes.get_batch(idx)
        else:
            video = vframes.get_batch(np.arange(0, total_frames))
        return video

    def get_frame_size(self, nframes: int, height: int, width: int):
        max_pixels_limit = 768 * 28 * 28
        max_total_pixels = 24576 * 28 * 28
        max_aspect_ratio = 200

        if self.resized_height and self.resized_width:
            height, width = self.resized_height, self.resized_width
            max_pixels = 16384 * 28 * 28
            min_pixels = 4 * 28 * 28
        else:
            min_pixels = self.video_min_pixels
            max_pixels = self.video_max_pixels if self.video_max_pixels else max(min(max_pixels_limit, max_total_pixels / nframes * self.frame_factor), int(self.video_min_pixels * 1.05))

        if max(height, width) / min(height, width) > max_aspect_ratio:
            print(f"Warning: absolute aspect ratio must be smaller than {max_aspect_ratio}, got {max(height, width) / min(height, width)}")

        resized_height = max(self.image_factor, self.round_by_factor(height, self.image_factor))
        resized_width = max(self.image_factor, self.round_by_factor(width, self.image_factor))
        if resized_height * resized_width > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            resized_height = self.floor_by_factor(height / beta, self.image_factor)
            resized_width = self.floor_by_factor(width / beta, self.image_factor)
        elif resized_height * resized_width < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_height = self.ceil_by_factor(height * beta, self.image_factor)
            resized_width = self.ceil_by_factor(width * beta, self.image_factor)
        return resized_height, resized_width

    def __call__(
        self,
        vframes,
        **kwargs
    ):
        video = self.sample_frames(vframes)
        nframes, _, height, width = video.shape
        resized_height, resized_width = self.get_frame_size(nframes, height, width)
        self.transform_size = {
            "max_height": resized_height,
            "max_width": resized_width,
        }

        self.video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline, transform_size=self.transform_size)

        video = self.video_transforms(video)

        return video

    def select_valid_data(self, data_samples):
        return super().select_valid_data(data_samples)


@Registry.register
class VACEVideoProcessor(AbstractVideoProcessor):
    def __init__(self, num_frames, auto_interval, max_height, max_width, max_hxw, train_fps, speed_factor, force_resolution,
                  vae_stride=None, vae_patch_size=None, zero_start=True, keep_last=True, **kwargs):
        super().__init__(**kwargs)
        if (num_frames - 1) % 4 != 0:
            raise AssertionError("The length of the frame must be the 4x+1")
        if vae_patch_size is None:
            vae_patch_size = [1, 2, 2]
        if vae_stride is None:
            vae_stride = [4, 8, 8]
        self.downsample = tuple([x * y for x, y in zip(vae_stride, vae_patch_size)])
        self.auto_interval = auto_interval
        self.max_height = max_height
        self.max_width = max_width
        self.speed_factor = speed_factor
        self.force_resolution = force_resolution
        self.max_hxw = max_hxw
        self.min_hxw = max_hxw
        self.train_fps = train_fps
        self.zero_start = zero_start
        self.keep_last = keep_last
        if self.max_hxw == 480 * 832:
            self.seq_len = (480 * 832 / (self.downsample[1] * self.downsample[2])) * (1 + (num_frames - 1) / 4)
        elif self.max_hxw == 720 * 1280:
            self.seq_len = (720 * 1280 / (self.downsample[1] * self.downsample[2])) * (1 + (num_frames - 1) / 4)
        else:
            raise NotImplementedError(f'image_size {self.max_hxw} is not supported')
        if self.seq_len < self.min_hxw / (self.downsample[1] * self.downsample[2]):
            raise AssertionError("seq_len is too short")
        self.rng = np.random.default_rng()


    def __call__(
            self,
            *vframes,
            crop_box=None,
            **kwargs
    ):
        fps = vframes[0].get_video_fps()
        length = min([r.get_len() for r in vframes])
        frame_timestamps = [vframes[0].get_frame_timestamp(i) for i in range(length)]
        frame_timestamps = np.array(frame_timestamps, dtype=np.float32)
        h, w = list(vframes[0].get_batch((0,)).shape[2:])
        # If a crop_box exists, x1, x2, y1, y2 are set to the crop_box values; otherwise, they are set to (0, w, 0, h).
        frame_ids, (x1, x2, y1, y2), (target_height, target_weight), fps = self._get_frameid_bbox(fps, frame_timestamps, h, w, crop_box)

        # preprocess video
        videos = [reader.get_batch(frame_ids)[:, y1:y2, x1:x2, :] for reader in vframes]

        self.image_size = (target_height, target_weight)
        video_transforms = get_transforms(is_video=True, train_pipeline=self.train_pipeline,
                            image_size=self.image_size)
        videos = [video_transforms(video) for video in videos]
        return *videos, frame_ids, (target_height, target_weight), fps

    def _get_frameid_bbox(self, fps, frame_timestamps, h, w, crop_box):
        if self.keep_last:
            return self._get_frameid_bbox_adjust_last(fps, frame_timestamps, h, w, crop_box)
        else:
            return self._get_frameid_bbox_default(fps, frame_timestamps, h, w, crop_box)

    # return the x previous frames
    def _get_frameid_bbox_fixed(self, fps, frame_timestamps, h, w, crop_box):
        target_fps = min(fps, self.train_fps)
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (int(duration * target_fps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = target_frame / target_fps
        begin = 0. if self.zero_start else random.randint(0, duration - target_duration)
        timestamps = np.linspace(begin, begin + target_duration, target_frame)
        frame_ids = list(range(0, target_frame))
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    # extrace a video from the target_duration and evenly capture the target_frame form it
    def _get_frameid_bbox_default(self, fps, frame_timestamps, h, w, crop_box):
        # Extract a number of frames from a specific segment of the video.
        target_fps = min(fps, self.train_fps)
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (int(duration * target_fps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = target_frame / target_fps
        begin = 0. if self.zero_start else random.randint(0, duration - target_duration)
        timestamps = np.linspace(begin, begin + target_duration, target_frame)
        frame_ids = np.argmax(np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] < frame_timestamps[None, :, 1]
        ), axis=1).tolist()
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    # evenly capture the target_frame form the video
    def _get_frameid_bbox_adjust_last(self, fps, frame_timestamps, h, w, crop_box):
        duration = frame_timestamps[-1].mean()
        x1, x2, y1, y2 = [0, w, 0, h] if crop_box is None else crop_box
        h, w = y2 - y1, x2 - x1
        ratio = h / w
        downsample_frame, downsample_height, downsample_weight = self.downsample

        area_z = min(self.seq_len, self.max_hxw / (downsample_height * downsample_weight), (h // downsample_height) * (w // downsample_weight))
        target_frame = min(
            (len(frame_timestamps) - 1) // downsample_frame + 1,
            int(self.seq_len / area_z)
        )

        # deduce target shape of the [latent video]
        target_area_z = min(area_z, int(self.seq_len / target_frame))
        target_height = round(np.sqrt(target_area_z * ratio))
        target_weight = int(target_area_z / target_height)
        target_frame = (target_frame - 1) * downsample_frame + 1
        target_height *= downsample_height
        target_weight *= downsample_weight

        # sample frame ids
        target_duration = duration
        target_fps = target_frame / target_duration
        timestamps = np.linspace(0., target_duration, target_frame)
        frame_ids = np.argmax(np.logical_and(
            timestamps[:, None] >= frame_timestamps[None, :, 0],
            timestamps[:, None] <= frame_timestamps[None, :, 1]
        ), axis=1).tolist()
        return frame_ids, (x1, x2, y1, y2), (target_height, target_weight), target_fps

    def select_valid_data(self, data_samples):
        return super().select_valid_data(data_samples)
