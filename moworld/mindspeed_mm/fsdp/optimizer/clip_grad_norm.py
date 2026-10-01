import math
from typing import List
import logging

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.device import get_device_type


logger = logging.getLogger(__name__)


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    Clip gradients by their norm for distributed training with support for Expert Parallelism (EP).

    Args:
        model: The model containing parameters to clip.
        max_norm: Maximum norm of gradients. If 0, only compute norm without clipping.
        norm_type: Type of norm to compute (p-norm). Default is 2.0 (L2 norm).
        error_if_nonfinite: If True, raise error when gradients are non-finite.
        foreach: Whether to use foreach implementation for gradient clipping.

    Returns:
        Total norm of gradients before clipping (or after clipping if max_norm > 0).
    """
    # EP-aware path (FSDP2 + EP)
    # _ep_param_groups is constructed in the optimizer
    # Check if model is configured for Expert Parallelism
    if hasattr(model, "_ep_param_groups"):
        return ep_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    ps = get_parallel_state()
    fsdp_group = ps.get_fsdp_group()
    params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.grad is not None]
    # Compute and reduce non-EP
    total_norm = _fsdp2_reduce_group(
        params=params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
    )
    if math.isinf(norm_type):
        total_norm = total_norm
    else:
        total_norm = total_norm ** (1.0 / float(norm_type))
    # Special case: max_norm = 0 means only compute gradient norm without clipping
    if max_norm > 0.:
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm, foreach=foreach)
    return total_norm


@torch.no_grad()
def ep_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    EP-aware gradient clipping for composable FSDP2 with reductions mirroring FSDP1:

    - Compute local norms for non-EP and EP parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-EP: all-reduce over FSDP group.
      • EP: all-reduce over EP-FSDP group, then over EP group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """

    ps = get_parallel_state()
    fsdp_group = ps.get_fsdp_group()
    ep_group = ps.get_ep_group() if ps.is_ep_enable() else None
    # For EP params sharded by FSDP2 along hidden dimension
    ep_fsdp_group = ps.get_efsdp_group()

    # Build param groups (filter out params without grads)
    ep_params: List[torch.nn.Parameter] = [p for p in model._ep_param_groups.get("ep", []) if p.grad is not None]
    non_ep_params: List[torch.nn.Parameter] = [p for p in model._ep_param_groups.get("non_ep", []) if p.grad is not None]
    # Compute and reduce non-EP
    non_ep_total = _fsdp2_reduce_group(
        params=non_ep_params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
    )
    # Compute and reduce EP: first across ep_fsdp, then across ep
    ep_total = _fsdp2_reduce_group(
        params=ep_params,
        norm_type=norm_type,
        reduce_groups=[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)],
    )

    if math.isinf(norm_type):
        total_norm = torch.maximum(non_ep_total, ep_total)
    else:
        total_norm = (non_ep_total + ep_total) ** (1.0 / float(norm_type))

    if max_norm == 0.:
        return total_norm
    # Apply the same clip coefficient to both groups
    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach=foreach)

    return total_norm


# compute local sum of param gard norm
def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor) else g.detach().to(torch.float32)
        for g in grads
    ]
    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device(get_device_type())
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    with torch.no_grad():
        grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads_local], _) in grouped_grads_local.items():
            if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads_local, p), p)
                res += torch.sum(torch.stack(out)).to(default_device)
            else:
                for grad_local in device_grads_local:
                    gn = torch.norm(grad_local, p=p)
                    res = res + (gn**p).to(default_device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val
