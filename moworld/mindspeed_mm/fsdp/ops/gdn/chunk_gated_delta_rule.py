# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch

from mindspeed_mm.fsdp.utils.import_utils import IS_TRITON_AVAILABLE
from mindspeed_mm.fsdp.train.training_context import TrainingContext, TrainingStage
from mindspeed_mm.fsdp.features.memory.async_offload import OffloadManager, SwapTensor
from mindspeed_mm.fsdp.utils.device import get_current_stream

if IS_TRITON_AVAILABLE:
    from .triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from .triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from .triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from .triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from .triton.solve_tril import solve_tril
    from .triton.cumsum import chunk_local_cumsum
    from .triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
else:
    def _identity_decorator(fn):
        return fn
    input_guard = _identity_decorator
    autocast_custom_fwd = _identity_decorator
    autocast_custom_bwd = _identity_decorator


def chunk_gated_delta_rule_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state


def chunk_gated_delta_rule_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(
            f"dg current type is {dg.dtype} , should be float32"
        )
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: Optional[torch.LongTensor] = None,
            use_qk_l2norm_in_kernel: bool = False,
            chunk_size: int = 64,
            skip_recompute: bool = False,
    ):
        q_rstd, k_rstd = None, None

        training_stage = TrainingContext().get_training_stage()
        layer_idx, depth = TrainingContext().get_layer_index(), TrainingContext().get_model_depth()
        h2d_stream = OffloadManager().swap_stream
        d2h_stream = OffloadManager().swap_stream

        if skip_recompute and training_stage == TrainingStage.BACKWARD:
            if layer_idx == depth - 1:
                if output_final_state:
                    final_state = OffloadManager().pop_npu_tensor()
                A = OffloadManager().pop_npu_tensor()
                o = OffloadManager().pop_npu_tensor()
                g = OffloadManager().pop_npu_tensor()

            else:
                layer_items_keys = OffloadManager().get_layer_items_keys(layer_idx)

                swap_tensor_nums = 4 if output_final_state else 3
                swap_tensors = []

                for swap_key in reversed(layer_items_keys[-swap_tensor_nums:]):
                    swap_tensor = OffloadManager().get(swap_key)
                    swap_tensor.launch_h2d(h2d_stream)
                    get_current_stream().wait_event(swap_tensor.h2d_event)

                    swap_tensors.append(swap_tensor.tensor)
                    OffloadManager().clear(swap_key)

                if output_final_state:
                    final_state, A, o, g = swap_tensors
                else:
                    A, o, g = swap_tensors
                    final_state = None
        else:
            g, o, A, final_state = chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                chunk_size=chunk_size
            )

        if skip_recompute and training_stage == TrainingStage.FORWARD:
            swap_tensors = [g, o, A]
            if output_final_state:
                swap_tensors.append(final_state)

            for swap_tensor in swap_tensors:
                key, after_block = OffloadManager().get_cnt(layer_idx)
                if after_block:
                    OffloadManager().del_npu_tensor("{}_".format(layer_idx - 1))

                if layer_idx == depth - 1:
                    OffloadManager().put_npu_tensor(SwapTensor(swap_tensor, key))
                else:
                    swap_tensor = SwapTensor(swap_tensor, key)
                    swap_tensor.launch_d2h(d2h_stream)
                    OffloadManager().put(key, swap_tensor)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor
    ):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        head_first: bool = False,
        skip_recompute: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
        skip_recompute (bool):
            Whether skip recomupte and async offload the outputs to cpu.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype} , k current type is {k.dtype} ,v current type is {v.dtype} , they should are equal"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        """This function is intended to align with the l2norm implementation in the FLA library."""
        original_dtype = x.dtype
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        # Counteract verl's autocast promotion (bf16 -> fp32) by restoring original dtype
        return (x * inv_norm).to(original_dtype)

    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        False,
        chunk_size,
        skip_recompute,
    )
    return o, final_state
