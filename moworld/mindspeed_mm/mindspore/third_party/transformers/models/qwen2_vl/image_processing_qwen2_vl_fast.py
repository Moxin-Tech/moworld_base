# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Fast Image processor class for Qwen2-VL."""

from typing import Optional, Union
import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import group_images_by_shape, reorder_images
from transformers.image_utils import SizeDict

from transformers.utils import TensorType
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
):
    # Group images by size for batched resizing
    grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
    resized_images_grouped = {}
    for shape, stacked_images in grouped_images.items():
        height, width = stacked_images.shape[-2:]
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            stacked_images = self.resize(
                image=stacked_images,
                size=SizeDict(height=resized_height, width=resized_width),
                interpolation=interpolation,
            )
        resized_images_grouped[shape] = stacked_images
    resized_images = reorder_images(resized_images_grouped, grouped_images_index)

    # Group images by size for further processing
    # Needed in case do_resize is False, or resize returns images with different sizes
    grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
    processed_images_grouped = {}
    processed_grids = {}
    for shape, stacked_images in grouped_images.items():
        resized_height, resized_width = stacked_images.shape[-2:]
        # Fused rescale and normalize
        patches = self.rescale_and_normalize(
            stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
        )
        # MSAdapter: Convert tensor to NumPy array
        patches = patches.asnumpy()
        if patches.ndim == 4:
            # add a temporal dimension if we have images
            patches = np.expand_dims(patches, axis=1)  # MSAdapter: torch.unsqueeze(1)

        # MSAdapter: use np.pad instead of torch.repeat and torch.cat
        if patches.shape[1] % temporal_patch_size != 0:
            pad_num = temporal_patch_size - (patches.shape[1] % temporal_patch_size)
            patches = np.pad(patches, ((0, 0), (0, pad_num), (0, 0), (0, 0), (0, 0)), mode='edge')
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        # MSAdapter: use np.reshape instead of torch.view
        patches = patches.reshape(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # MSAdapter: use np.transpose instead of torch.permute)
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        # MSAdapter: Convert NumPy back to tensor
        flatten_patches = torch.from_numpy(flatten_patches)

        processed_images_grouped[shape] = flatten_patches
        processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

    processed_images = reorder_images(processed_images_grouped, grouped_images_index)
    processed_grids = reorder_images(processed_grids, grouped_images_index)
    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids)

    return BatchFeature(
        data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
    )
