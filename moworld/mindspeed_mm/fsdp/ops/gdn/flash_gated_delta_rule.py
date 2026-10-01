import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# Large default smoke shapes can exceed Triton-NPU's default launch-grid limit.
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch_npu
import fla_npu

from mindspeed_mm.fsdp.ops.gdn.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from mindspeed_mm.fsdp.ops.gdn.triton_core.cumsum import chunk_local_cumsum
from mindspeed_mm.fsdp.ops.gdn.triton_core.l2norm import l2norm_bwd, l2norm_fwd
from mindspeed_mm.fsdp.ops.gdn.triton_core.solve_tril_fast import solve_tril_npu as solve_tril
from mindspeed_mm.fsdp.ops.gdn.triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from mindspeed_mm.fsdp.train.training_context import TrainingContext, TrainingStage
from mindspeed_mm.fsdp.features.memory.async_offload import OffloadManager, SwapTensor
from mindspeed_mm.fsdp.utils.device import get_current_stream

_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)
_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


def cdiv_torch(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat(
        [torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_list(cu_seqlens: list[int] | torch.LongTensor, chunk_size: int) -> list[int]:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _as_int_list(value: Optional[list[int] | torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _as_chunk_dict(
    value: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_size: int,
) -> Dict[str, Optional[torch.LongTensor]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> Dict[str, Optional[list[int]]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: torch.Tensor, chunk_size: int) -> int:
    # Keep this aligned with mindspeed_mm.fsdp.ops.gdn.triton_core.cumsum.chunk_local_cumsum_scalar.
    h = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, h * int(chunk_size)))


def _ensure_varlen_metadata(
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> tuple[
    Optional[torch.LongTensor],
    Optional[list[int]],
    Optional[Dict[str, Optional[torch.LongTensor]]],
    Optional[Dict[str, Optional[list[int]]]],
]:
    if cu_seqlens is None:
        return None, None, None, None

    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens_list) or _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None

    tensor_indices = _as_chunk_dict(chunk_indices, chunk_size)
    list_indices = _as_chunk_list_dict(chunk_indices_list, chunk_size)

    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(_cumsum_block_t(g, chunk_size))

    for size in required_sizes:
        key = str(size)
        if key not in tensor_indices or tensor_indices[key] is None:
            tensor_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        if key not in list_indices or list_indices[key] is None:
            list_indices[key] = prepare_chunk_indices_list(cu_seqlens_list, size)

    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]],
    chunk_size: int,
) -> Optional[torch.LongTensor]:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    # A is the WY lower-triangular representation before inversion.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        output_dtype=k.dtype,
    )

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    if not output_final_state:
        final_state = None

    o = torch.ops.npu.npu_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta,
        A,
        chunk_size,
        g=g,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do = do.transpose(1, 2).contiguous()

    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    dh0 = None

    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

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
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
        chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        skip_recompute: bool = False,
    ):

        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
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
            g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=cu_seqlens_list,
                chunk_indices=chunk_indices,
                chunk_indices_list=chunk_indices_list,
                chunk_size=chunk_size,
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

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@_disable_compile
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
    skip_recompute: bool = False,
):
    r"""
    Flash-linear-attention NPU port of xtuner's GDN entry.

    This port keeps the xtuner NPU layout:
        q, k: [B, H, T, K]
        v:    [B, H, T, V]
        g:    [B, T, H]
        beta: [B, T, H]

    It returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] when output_final_state=True, else None
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}; they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != q.shape[0] or g.shape[1] != q.shape[2] or g.shape[2] != q.shape[1]:
        raise ValueError(
            "Expected q/k/v in [B, H, T, D] and g/beta in [B, T, H]; "
            f"got q={tuple(q.shape)}, g={tuple(g.shape)}."
        )

    if head_first:
        warnings.warn(
            "head_first is kept only for API compatibility. This NPU port always expects q/k/v as [B, H, T, D].",
            stacklevel=2,
        )
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    if cu_seqlens is not None:
        cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
            g=g,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens_list) - 1:
            raise ValueError(
                "The number of initial states is expected to match the number of input sequences, "
                f"got initial_state.shape[0]={initial_state.shape[0]} and sequences={len(cu_seqlens_list) - 1}."
            )
    else:
        cu_seqlens_list = None
        chunk_indices = None
        chunk_indices_list = None

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
        skip_recompute,
    )
    return o, final_state


__all__ = [
    "flash_gated_delta_rule",
    "flash_chunk_gated_delta_rule_fwd",
    "flash_chunk_gated_delta_rule_bwd",
    "prepare_chunk_indices",
    "prepare_chunk_indices_list",
]


def _build_mean_1k_cu_seqlens(
    total_tokens: int,
    chunk_size: int,
    device: str,
    mean_len: int = 1024,
) -> torch.LongTensor:
    num_seqs = max(1, round(total_tokens / mean_len))
    if num_seqs == 1:
        lengths = [total_tokens]
    else:
        target = max(1, total_tokens // num_seqs)
        delta = max(chunk_size, (target // 4 // chunk_size) * chunk_size)
        low = max(1, target - delta)
        high = max(1, target + delta)
        lengths = [low if i % 2 == 0 else high for i in range(num_seqs)]

        diff = total_tokens - sum(lengths)
        i = len(lengths) - 1
        while diff != 0:
            if diff > 0:
                add = min(diff, max(1, delta))
                lengths[i] += add
                diff -= add
            else:
                sub = min(-diff, max(0, lengths[i] - 1))
                if sub == 0:
                    i = (i - 1) % len(lengths)
                    continue
                lengths[i] -= sub
                diff += sub
            i = (i - 1) % len(lengths)

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return torch.tensor(offsets, dtype=torch.int64, device=device)
