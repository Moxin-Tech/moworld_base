import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


def eager_swiglu(inputs, dim=-1):
    if dim < 0:
        dim = inputs.dim() + dim
    x1, x2 = torch.chunk(inputs, 2, dim=dim)
    return torch.nn.functional.silu(x1) * x2


def fused_swiglu(inputs, dim=-1):
    return torch_npu.npu_swiglu(inputs, dim=dim)


def swiglu(inputs, dim=-1, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_swiglu(inputs, dim=dim)
    else:
        return eager_swiglu(inputs, dim=dim)

def clamp_swiglu(inputs, dim=-1, fused=True, limit=0.0):
    y1, y2 = torch.chunk(inputs, 2, dim=-1)
    if limit > 0:
        y1 = y1.clamp(min=None, max=limit)
        y2 = y2.clamp(min=-limit, max=limit)
    return swiglu(torch.cat([y1, y2], dim=-1), dim=dim, fused=fused)
