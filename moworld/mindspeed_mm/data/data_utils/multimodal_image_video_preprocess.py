from abc import ABC, abstractmethod
import copy
import math
import random

import torch
import numpy as np
from PIL import Image
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from torchvision.datasets.folder import pil_loader

from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.data_transform import Expand2Square
from mindspeed_mm.data.data_utils.video_reader import VideoReader


class MultiModalImageVideoPreprocessBase(ABC):
    def __init__(
            self,
            train_pipeline=None,
            image_reader_type="torchvision",
            tokenizer=None,
            dynamic_image_size=False,
            patch_size=14,
            image_size=224,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_thumbnail=False,
            transform_size=None,
            min_num_frame: int = 4,
            max_num_frame: int = 12,
            sampling_method: str = "rand",
            **kwargs,
    ):
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline, transform_size=transform_size
        )
        self.train_pipeline = train_pipeline
        self.image_reader_type = image_reader_type
        self.tokenizer = tokenizer
        self.dynamic_image_size = dynamic_image_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.min_num_frame = min_num_frame
        self.max_num_frame = max_num_frame
        self.sampling_method = sampling_method

    @abstractmethod
    def __call__(self, image_path=None, video_path=None, **kwargs):
        pass

    def image_reader(self, image_path):
        if self.image_reader_type in ["torchvision", "CLIPImageProcessor"]:
            image = pil_loader(image_path)
        elif self.image_reader_type == "Image":
            image = Image.open(image_path).convert("RGB")  # [h, w, c]
        else:
            raise NotImplementedError(
                f"Unsupported image reader type: {self.image_reader_type}"
            )
        return image


class InternvlImageVideoPreprocess(MultiModalImageVideoPreprocessBase):
    def __init__(self, **kwargs,):
        super().__init__(**kwargs)

    def __call__(self, image_path=None, video_path=None, mode="", num_image=1, clip=None, **kwargs):
        if image_path:
            return self.image_to_pixel_values(image_path, mode, num_image)
        elif video_path:
            return self.video_to_pixel_values(video_path, clip)
        else:
            raise ValueError("Either image_path or video_path must be provided")


    def image_to_pixel_values(self, image_path, mode="", num_image=1):
        image = self.image_reader(image_path)
        max_num = self.max_dynamic_patch // num_image if mode == "multi_image" else self.max_dynamic_patch
        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=max_num,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [self.image_transforms(image) for image in images]
        pixel_values = pixel_values if mode == "multi_image" else torch.stack(pixel_values)

        return {"pixel_values": pixel_values}

    def video_to_pixel_values(self, video_path, clip):
        image_list = read_frames_decord(
            video_path=video_path,
            num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=clip)

        # Transform each frame image and stack them into a tensor
        pixel_values = [self.image_transforms(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)

        return {"pixel_values": pixel_values, "image_list": image_list}


def get_multimodal_image_video_preprocessor(template_name, **kwargs):
    if template_name in ("internvl2_5", "internvit_qwen3"):
        return InternvlImageVideoPreprocess(**kwargs)
    else:
        raise ValueError(f"Unsupported template_name: {template_name}")


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    This function finds the closest aspect ratio from a set of target aspect ratios based on the input
    image's aspect ratio.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """
    This function dynamically preprocesses an input image by resizing it to match a closest target
    aspect ratio and then splitting the resized image into smaller blocks.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def slice_image(
        image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
            (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
        original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num, new_schema=False):
    if new_schema:
        image_placeholder = (
                tokenizer.slice_start + tokenizer.unk_token * query_num + tokenizer.slice_end
        )
    else:
        image_placeholder = (
                tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
        )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for _ in range(rows):
        lines = []
        for _ in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    if new_schema:
        slice_placeholder = '\n'.join(slices)
    else:
        slice_placeholder = tokenizer.slice_start + \
                            "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, client=None, clip=None, min_num_frames=4
):
    video_reader = VideoReader(video_reader_type="DecordVideo")(video_path, layout="THWC", array_type="numpy")
    vlen = video_reader.get_len()
    fps = video_reader.get_video_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]  # little different
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
    else:
        raise ValueError
    return frame_indices
