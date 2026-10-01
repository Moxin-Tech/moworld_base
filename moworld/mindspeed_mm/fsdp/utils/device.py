# Copyright 2025 Bytedance Ltd. and/or its affiliates
import sys
import types
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = False
try:
    import torch_npu

    IS_NPU_AVAILABLE = True
except Exception as e:
    IS_NPU_AVAILABLE = False

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False


def accelerator_getattr(module, fallback_module):
    def __getattr__(name):
        if hasattr(fallback_module, name):
            attr = getattr(fallback_module, name)
            setattr(module, name, attr)
            return attr
        else:
            raise AttributeError(f'module {module} and {fallback_module} has no attribute {name}.')

    return __getattr__


def set_accelerator_compatible(fallback_module=None):
    accelerator_module = types.ModuleType('torch.accelerator')
    accelerator_module.__doc__ = f'Fallback accelerator module that delegates to {get_device_type()}'
    for attr in dir(torch.accelerator):
        if attr.startswith('__'):
            continue
        setattr(accelerator_module, attr, getattr(torch.accelerator, attr))

    accelerator_module.__getattr__ = accelerator_getattr(accelerator_module, fallback_module)
    torch.accelerator = accelerator_module
    sys.modules['torch.accelerator'] = accelerator_module


def get_dist_comm_backend(cpu: bool = False) -> str:
    """Return distributed communication backend type based on device type."""
    if cpu:
        if IS_CUDA_AVAILABLE:
            return "cpu:gloo,cuda:nccl"
        elif IS_NPU_AVAILABLE:
            return "cpu:gloo,npu:hccl"
    if IS_CUDA_AVAILABLE:
        return "nccl"
    elif IS_NPU_AVAILABLE:
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_name() -> str:
    """Get real device name, e.g. A100, H100"""
    return get_torch_device().get_device_name()


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def create_stream(device: Optional[torch.device] = None, priority: int = 0) -> Any:
    "Create custom stream."
    return get_torch_device().Stream(device=device, priority=priority)


def create_event(enable_timing: bool = False, blocking: bool = False) -> Any:
    "Create empty event."
    return get_torch_device().Event(enable_timing=enable_timing, blocking=blocking)


def get_current_stream() -> Any:
    return get_torch_device().current_stream()


def switch_to_specified_stream(stream) -> Any:
    return get_torch_device().stream(stream)


def get_memory_reserved():
    if IS_NPU_AVAILABLE:
        return torch.npu.memory_reserved()
    else:
        return torch.cuda.memory.memory_reserved()


def get_max_memory_reserved():
    if IS_NPU_AVAILABLE:
        return torch.npu.max_memory_reserved()
    else:
        return torch.cuda.memory.max_memory_reserved()


def get_memory_allocated():
    if IS_NPU_AVAILABLE:
        return torch.npu.memory_allocated()
    else:
        return torch.cuda.memory.memory_allocated()


def get_max_memory_allocated():
    if IS_NPU_AVAILABLE:
        return torch.npu.max_memory_allocated()
    else:
        return torch.cuda.memory.max_memory_allocated()


def reset_peak_memory_stats():
    if IS_NPU_AVAILABLE:
        return torch.npu.reset_peak_memory_stats()
    else:
        return torch.cuda.memory.reset_peak_memory_stats()


def set_allow_hf32(allow_hf32=None) -> None:
    """Set allow_hf32/allow_tf32 attribute based on device type."""
    if allow_hf32 is None:
        return

    if IS_NPU_AVAILABLE:
        torch.npu.aclnn.allow_hf32 = allow_hf32
    else:
        torch.backends.cudnn.allow_tf32 = allow_hf32
