# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
import math
from typing import Tuple, Dict

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard


def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if len(G.shape) != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 expects a 2-D tensor, got shape {G.shape}"
        )

    # Coefficients a,b,c are used to ensure convergence
    # values are from the source code of Keller Jordan
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


def normalize_range(value_range: Tuple[int, int], start: int):
    return (value_range[0] - start, value_range[1] - start)


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD.
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough).
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self,
                 param_groups,
                 lr: float = 2e-2,
                 weight_decay: float = 0.1,
                 matched_adamw_rms: float = 0.2,
                 momentum: float = 0.95,
                 nesterov: bool = True,
                 ns_steps: int = 5,
                 adamw_betas: Tuple[float, float] = (0.95, 0.95),
                 adamw_eps: float = 1e-8):

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):

        # First pass: update muon momentum buffers
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize momentum buffer
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(p.grad)

                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(p.grad)

                # Prepare for Newton-Schulz iteration
                if nesterov:
                    g = p.grad.add(buf, alpha=momentum)
                else:
                    g = buf

                state["ns_input"] = g.to(torch.bfloat16)

        # Second pass: apply updates
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                ns_input = state.pop("ns_input", None)
                if ns_input is None:
                    continue

                # Handle FSDP sharding
                if hasattr(p, 'device_mesh'):
                    device_mesh = ns_input.device_mesh

                    # Gather to replica for computation
                    new_placements = [Replicate() if isinstance(placement, Shard)
                                     else placement for placement in ns_input.placements]

                    ns_input_full = ns_input.redistribute(device_mesh, new_placements)
                    ns_input_local = ns_input_full.to_local()
                else:
                    ns_input_local = ns_input

                # Apply Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(ns_input_local, steps=ns_steps)

                # Handle FSDP sharding for output
                if hasattr(p, 'device_mesh'):
                    update_dtensor = DTensor.from_local(update, p.device_mesh, new_placements)
                    update_sharded = update_dtensor.redistribute(p.device_mesh, p.placements)
                    update = update_sharded

                # Apply weight decay
                p.mul_(1 - lr * weight_decay)

                # Adjust LR and apply update
                adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, ns_input.shape)
                p.data.add_(update, alpha=-adjusted_lr)

        # Use AdamW for non-Muon parameters
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            # init step
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            step = group['step']
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=- lr / scale)
