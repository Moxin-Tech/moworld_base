from typing import Dict, Any
import time

import torch
from torch.distributed.tensor import DTensor, Replicate

from .constants import AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM
from .device import get_device_type, get_torch_device

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.device import (
    get_memory_reserved,
    get_max_memory_reserved,
    get_memory_allocated,
    get_max_memory_allocated,
)


def to_empty_if_needed(model, device: torch.device | str | int | None, recurse: bool = True):
    """Move the parameters and buffers to the specified device without copying storage if they are not already on that device.

    Args:
        module: The module whose parameters and buffers to (maybe) move.
        device: The desired device of the parameters and buffers in the module. If `None`, the default device is used.
        recurse: Whether parameters and buffers of submodules should be recursively moved to the specified device.

    Behavior Scenarios:
        Scenario 1: Meta initialization + CPU offload (e.g., FSDP2 with offload_to_cpu=True)
        -------------------------------------------------------------------------
          - Parameters:               Meta => CPU
          - Buffers:                  CUDA => CUDA
          - Tensors(eg. inv_freq):    CPU => CUDA

        Scenario 2: Meta initialization only (no CPU offload)
        -------------------------------------------------------------------------
          - Parameters:               Meta => CUDA
          - Buffers:                  CUDA => CUDA
          - Tensors(eg. inv_freq):    CPU => CUDA
    """
    device = torch.empty((), device=device).device

    def _replace_tensor(t):
        # Case 1: This is a trainable parameter (subclass of torch.Tensor with requires_grad)
        if isinstance(t, torch.nn.Parameter):# meta or cpu
            return torch.empty_like(t, device=device) if t.device != device else t
        else:
            # Case 2: This is a buffer or regular tensor (non-parameter)
            # we do not offload buffer to cpu when enable FSDP2 offload_to_cpu function.
            return t.to(device=get_device_type()) if t.device == torch.device('cpu') else t

    return model._apply(_replace_tensor, recurse=recurse)


def tensor_to_dtensor(t: torch.Tensor, device_mesh, placements):
    replicate = [Replicate() for _ in range(device_mesh.ndim)]
    ori_dtensor = DTensor.from_local(local_tensor=t, device_mesh=device_mesh, placements=replicate)
    new_dtensor = ori_dtensor.redistribute(device_mesh=device_mesh, placements=placements)
    return new_dtensor


def init_model_weights(model):
    post_init_modules = []

    def _pre_init_weights():
        # Find the parameters that cannot be initialized with Dtensor type, restore full_tensor, and then shard after initialization is complete
        for name, module in model.named_modules():
            setattr(module, "_is_initialized", False)
            if getattr(module, "_is_hf_initialized", False):
                module._is_hf_initialized = False
            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                post_init_modules.append([name, module.weight.data.device_mesh, module.weight.data.placements])
                full_weight = torch.empty(module.weight.data.shape, device=module.weight.device)
                module.weight = torch.nn.Parameter(full_weight, requires_grad=module.weight.requires_grad)

    def _post_init_weights():
        if not post_init_modules:
            return

        for post_init_name, device_mesh, placements in post_init_modules:
            for name, module in model.named_modules():
                if name != post_init_name:
                    continue
                if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                    dtensor = tensor_to_dtensor(module.weight.data, device_mesh, placements)
                    module.weight = torch.nn.Parameter(dtensor, requires_grad=module.weight.requires_grad)

    _pre_init_weights()
    model.init_weights()
    _post_init_weights()


def move_to_device(batch: Dict[str, Any], float_dtype: str = None, non_blocking=False):
    new_batch = dict()
    for k, v in batch.items():
        if k in [AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM]:
            new_batch[k] = v.to(device=get_device_type(), non_blocking=non_blocking)
        elif isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            new_batch[k] = v.to(device=get_device_type(), dtype=dtype, non_blocking=non_blocking)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            new_batch[k] = [t.to(device=get_device_type(),
                            dtype=float_dtype if torch.is_floating_point(t) else None, non_blocking=non_blocking)
                        for t in v]
        elif isinstance(v, (bool, int, float, str)) or v is None:
            new_batch[k] = v
    return new_batch


def get_time(barrier=False):
    if barrier:
        torch.distributed.barrier()
    get_torch_device().synchronize()
    return time.time()


def is_npu_available():
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False


def configure_hsdp_gradient_sync(model, is_last_step: bool):
    """
    Configure gradient synchronization strategy for HSDP (Hierarchical Sharded Data Parallel).

    In HSDP sharding, by default, gradients are AllReduced across different FSDP domains
    during every backward pass. However, this is redundant as synchronization is only
    required once before `optimizer.step`.

    This function optimizes communication overhead by controlling:
    1. set_requires_all_reduce: Sets if the module should all-reduce gradients.
        This can be used to implement gradient accumulation with only reduce-scatter but not all-reduce for HSDP.
    2. set_is_last_backward: Sets whether the next backward is the last one. On the last backward,
        FSDP waits on pending gradient reduction and clears internal data data structures for backward prefetching.
        This can be useful for microbatching.

    Args:
        model: The model wrapped with fully_shard (FSDP2).
        is_last_step (bool): Whether the current step is the last in the gradient accumulation cycle.
    """
    model.set_is_last_backward(is_last_step)
    model.set_requires_all_reduce(is_last_step)


def report_memory(name):
    """Simple memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        get_memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        get_max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        get_memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        get_max_memory_reserved() / mega_bytes)
    if get_parallel_state().get_dp_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
            flush=True)
