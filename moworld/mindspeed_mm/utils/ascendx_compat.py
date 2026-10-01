"""
Compatibility shim for ascendx_video.base -> Mindspeed context parallel utilities.

The original camera-control implementation imports from `ascendx_video.base`:
    from ascendx_video.base import ParallelManager, get_pad, set_pad, split_sequence, gather_sequence

In Mindspeed-MM, the equivalent functionality is provided by:
    - ParallelManager -> mpu context parallel groups
    - split_sequence -> split_forward_gather_backward
    - gather_sequence -> gather_forward_split_backward
    - get_pad / set_pad -> not needed (Mindspeed handles padding internally)

This module provides drop-in replacements so that code referencing ascendx_video
can work within the Mindspeed-MM framework.
"""

import torch
import torch.distributed as dist

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core import mpu


class ParallelManager:
    """Replacement for ascendx_video.base.ParallelManager.

    Manages data parallel (dp), context parallel (cp), and sequence parallel (sp)
    process groups, matching the `ascendx_video.base` interface.
    """

    def __init__(self, dp_size, cp_size, sp_size):
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.sp_size = sp_size
        self.dp_rank = 0
        self.cp_rank = 0
        self.sp_rank = 0

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Build sp_group: consecutive ranks of size sp_size
        # Build dp_group: ranks at same sp position across different sp groups
        self.sp_group = None
        self.dp_group = None

        if dist.is_initialized() and world_size > 1:
            # SP group: ranks within the same DP group slice
            dp_group_idx = rank // sp_size
            sp_start = dp_group_idx * sp_size
            sp_ranks = list(range(sp_start, sp_start + sp_size))
            self.sp_group = dist.new_group(sp_ranks)
            self.sp_rank = rank % sp_size

            # DP group: same SP rank across DP groups
            dp_ranks = list(range(rank % sp_size, world_size, sp_size))
            self.dp_group = dist.new_group(dp_ranks)
            self.dp_rank = dp_group_idx

    def sp_group_size(self):
        return self.sp_size


# Pad storage for sequence parallel operations
_pad_store = {}


def get_pad(name):
    """Get padding value for a named tensor split operation.

    In ascendx_video, this returns the padding size needed to evenly divide
    a tensor across the SP group. In Mindspeed-MM, the split/gather functions
    handle this internally.
    """
    return _pad_store.get(name, 0)


def set_pad(name, total_len, group):
    """Set padding value so that total_len is divisible by group size.

    This is needed to match ascendx_video behavior where explicit padding
    tracking is used for sequence parallel splits.
    """
    if isinstance(group, int):
        world_size = group
    elif hasattr(group, 'size'):
        world_size = group.size()
    else:
        world_size = 1
    remainder = total_len % world_size
    _pad_store[name] = (world_size - remainder) % world_size


def split_sequence(tensor, group, dim=1, pad=0):
    """Split a tensor along a dimension for sequence parallel distribution.

    Drop-in replacement for ascendx_video.base.split_sequence.
    Uses Mindspeed's split_forward_gather_backward.
    """
    if pad > 0:
        # Pad the tensor to make it evenly divisible
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)

    if isinstance(group, int):
        # Simple chunk-based split
        chunks = group
        return tensor.chunk(chunks, dim=dim)[mpu.get_context_parallel_rank()] if mpu.get_context_parallel_world_size() > 1 else tensor

    return split_forward_gather_backward(tensor, group, dim=dim, grad_scale="down")


def gather_sequence(tensor, group, dim=1, pad=0):
    """Gather a tensor along a dimension from sequence parallel distribution.

    Drop-in replacement for ascendx_video.base.gather_sequence.
    Uses Mindspeed's gather_forward_split_backward.
    """
    if isinstance(group, int):
        # Simple all-gather
        if not dist.is_initialized():
            return tensor
        tensor_list = [torch.zeros_like(tensor) for _ in range(group)]
        dist.all_gather(tensor_list, tensor)
        result = torch.cat(tensor_list, dim=dim)
    else:
        result = gather_forward_split_backward(tensor, group, dim=dim, grad_scale="up")

    # Remove padding if it was added
    if pad > 0:
        slices = [slice(None)] * result.ndim
        slices[dim] = slice(None, -pad if pad > 0 else None)
        result = result[tuple(slices)]

    return result
