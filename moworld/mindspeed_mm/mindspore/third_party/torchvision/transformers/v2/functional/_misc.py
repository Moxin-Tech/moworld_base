# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Soumith Chintala 2016;
from typing import List
import torch
import numpy as np

from torchvision.transforms.v2.functional._misc import normalize
from torchvision.transforms.v2.functional._utils import _KERNEL_REGISTRY, _register_kernel_internal
from torchvision import tv_tensors


def normalize_image(image: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {image.dtype}.")

    if image.ndim < 3:
        raise ValueError(f"Expected tensor to be a tensor image of size (..., C, H, W). Got {image.shape}.")

    if isinstance(std, (tuple, list)):
        divzero = not all(std)
    elif isinstance(std, (int, float)):
        divzero = std == 0
    else:
        divzero = False
    if divzero:
        raise ValueError("std evaluated to zero, leading to division by zero.")

    dtype = image.dtype
    # MSAdapt: Convert tensor to numpy array
    image = image.asnumpy()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # MSAdapt: Use numpy reshape (original: torch view)
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1, 1)
    if std.ndim == 1:
        std = std.reshape(-1, 1, 1)

    # MSAdapt: Numpy in-place subtraction (equivalent to torch.sub_), Numpy non-in-place subtraction (equivalent to torch.sub)
    image = image - mean
    image /= std

    # MSAdapt: Convert numpy array back to torch tensor
    result = torch.from_numpy(image).to(dtype=dtype)

    return result


def patch_normalize_image(aspm):
    normalize_registry = _KERNEL_REGISTRY.get(normalize, {})
    normalize_registry.pop(torch.Tensor, None)
    normalize_registry.pop(tv_tensors.Image, None)
    _register_kernel_internal(normalize, torch.Tensor)(normalize_image)
    _register_kernel_internal(normalize, tv_tensors.Image)(normalize_image)
    aspm.register_patch('torchvision.transforms.v2.functional._misc.normalize_image', normalize_image)
