# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Single module: NPU causal depthwise conv1d (Triton) + xtuner-compatible autograd API.
# Inlined kernel support + Triton kernels (ex-mojo_opset) + xtuner `causal_conv1d` API.

from __future__ import annotations

import contextlib
import functools
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version


# Compatibility shim: in newer triton-ascend, slice ops live in
# `triton.language.extra.cann.extension` instead of `triton.language`.
# Re-expose them on `tl` so kernel code works on both versions.
try:
    from triton.language.extra.cann.extension import extract_slice, insert_slice

    if not hasattr(tl, "extract_slice"):
        tl.extract_slice = extract_slice
    if not hasattr(tl, "insert_slice"):
        tl.insert_slice = insert_slice
except ImportError:
    pass


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(
    *,
    make_contiguous: bool = True,
    auto_to_device: bool = True,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if make_contiguous:
                new_args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
                new_kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            else:
                new_args = args
                new_kwargs = kwargs

            tensor = None
            for a in new_args:
                if isinstance(a, torch.Tensor):
                    tensor = a
                    break
            if tensor is None:
                for v in new_kwargs.values():
                    if isinstance(v, torch.Tensor):
                        tensor = v
                        break

            if auto_to_device and tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*new_args, **new_kwargs)

        return wrapper

    return decorator


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = "2.4") -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        return "meta_device"


device = get_available_device()
device_torch_lib = getattr(torch, device)


if check_pytorch_version("2.4"):

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)


else:

    def custom_device_ctx(index: int):
        return torch.npu.device(index)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    lens = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    total = lens.sum()
    flat = torch.arange(total, device=cu_seqlens.device)
    seq_ids = torch.repeat_interleave(torch.arange(lens.numel(), device=cu_seqlens.device), lens)
    offsets = torch.cumsum(lens, 0) - lens
    indices = flat - offsets[seq_ids]
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], dim=1)


@lru_cache(maxsize=1)
def get_num_cores(op_type="vector"):
    assert op_type in ["vector", "cube", "mix"], f"op_type {op_type} must in ['vector', 'cube', 'mix']."
    return (
        triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        if op_type == "vector"
        else triton.runtime.driver.active.utils.get_device_properties("npu")["num_aicore"]
    )


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_fwd_kernel_old(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

        b_y = tl.zeros((BT, BD), dtype=tl.float32)

        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]

                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # We keep intra loop load because preloading will cause ub overflow under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])

                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w

                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d

                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                    tl.float32
                )

                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(
                residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
            )
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["weight"] is not None,
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_CHKS: tl.int32,
    NUM_BLKS_D: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_BLKS_D * NUM_CHKS

    for task_id in range(pid, total_tasks, num_programs):
        i_d_blk = task_id % NUM_BLKS_D
        i_chk = task_id // NUM_BLKS_D

        i_d = i_d_blk

        if IS_VARLEN:
            idx_ptr = chunk_indices + i_chk * 2
            i_n = tl.load(idx_ptr).to(tl.int32)
            i_t = tl.load(idx_ptr + 1).to(tl.int32)

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)
            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

        D_SUB: tl.constexpr = D // H
        HEADS_PER_BLOCK: tl.constexpr = BD // D_SUB
        b_y = tl.zeros((BT, BD), dtype=tl.float32)

        yi_offset_1 = i_d * BD + tl.arange(0, BD)[None, :]

        if not USE_INITIAL_STATE:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]

                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                # We keep intra loop load because preloading will cause ub overflow under certain tiling.
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])

                b_y += b_yi
        elif i_t * BT >= W:
            for i_w in tl.static_range(-W + 1, 1):
                yi_offset_0 = i_t * BT + i_w + tl.arange(0, BT)[:, None]
                mask = (yi_offset_0 < T_len) & (yi_offset_1 < D) & (yi_offset_0 >= 0)
                b_yi = tl.load(x + bos * D + yi_offset_0 * D + yi_offset_1, mask=mask, other=0.0).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w

                m_x = ((o_x >= 0) & (o_x < T_len))[:, None] & m_d

                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D * W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(
                    tl.float32
                )

                if HAS_WEIGHT:
                    b_yi *= tl.extract_slice(b_w, [i_w + W - 1, 0], [1, BD], [1, 1])
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == "swish" or ACTIVATION == "silu":
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(
                residual + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
            )
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        head_start = (i_d * BD) // D_SUB
        batch_off = 0 if IS_VARLEN else i_b * T * D
        bos_off = bos if IS_VARLEN else 0
        p_y = tl.make_block_ptr(
            y + batch_off + head_start * D_SUB * T + bos_off * D_SUB,
            (T_len, HEADS_PER_BLOCK, D_SUB),
            (D_SUB, D_SUB * T, 1),
            (i_t * BT, 0, 0),
            (BT, HEADS_PER_BLOCK, D_SUB),
            (2, 1, 0),
        )
        b_y = tl.reshape(b_y, (BT, HEADS_PER_BLOCK, D_SUB))  # for better readability, no actual reshape
        tl.store(
            p_y,
            tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1, 2),
        )


@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_Blk_D: tl.int32,
    NUM_CHKS: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_CHKS * NUM_Blk_D

    for task_id in range(pid, total_tasks, num_programs):
        i_d = task_id % NUM_Blk_D
        i_chk = task_id // NUM_Blk_D

        if IS_VARLEN:
            i_t = i_chk

            idx_chk = i_chk

            i_tg = idx_chk

            ptr = chunk_indices + idx_chk * 2
            i_n = tl.load(ptr).to(tl.int32)
            i_t_offset = tl.load(ptr + 1).to(tl.int32)

            i_t = i_t_offset

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)

            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_tg = i_chk

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_x = tl.make_block_ptr(x + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_x = tl.load(p_x, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")

        b_dx = tl.zeros((BT, BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)

        if not USE_FINAL_STATE:
            b_dw = tl.zeros((W, BD), dtype=tl.float32)

            K: tl.constexpr = D // H
            HEADS_PER_BLOCK: tl.constexpr = BD // K
            head_start = (i_d * BD) // K
            # B T H K
            # B H T K
            p_dy = tl.make_block_ptr(
                dy + bos * K  + head_start * T * K,
                (T_len, HEADS_PER_BLOCK, K),
                (K, T * K, 1),
                (i_t * BT, 0, 0),
                (BT * W, HEADS_PER_BLOCK, K),
                (2, 1, 0),
            )
            b_dy = tl.load(p_dy, boundary_check=(0, 1, 2)).to(tl.float32)
            b_dy = tl.reshape(b_dy, (BT * W, BD))  # for better readability, no actual reshape

            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT * W, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)

            for i_w in tl.static_range(0, W):
                b_dy_sub = tl.extract_slice(b_dy, [i_w, 0], [BT, BD], [1, 1])

                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    b_y_sub = tl.extract_slice(b_y, [i_w, 0], [BT, BD], [1, 1])
                    b_ys = tl.sigmoid(b_y_sub)
                    b_dy_sub = b_dy_sub * b_ys * (1 + b_y_sub * (1 - b_ys))

                b_wdy = b_dy_sub
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw_sub = tl.sum(b_dy_sub * b_x, 0)  # [BT, BD] * [BT, BD] --> sum(0) = [BD]
                    b_dw = tl.insert_slice(b_dw, b_dw_sub[None, :], [W - i_w - 1, 0], [1, BD], [1, 1])

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_sub, 0)
                b_dx += b_wdy

            p_dw = tl.make_block_ptr(dw + i_tg * W * D, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            tl.store(p_dw, b_dw.to(dw.dtype.element_ty))
        elif i_t * BT >= W:
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))

                b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                if HAS_WEIGHT:
                    b_dw = tl.sum(b_dy_shift * b_x, 0)

                    if USE_INITIAL_STATE:
                        mask_head_rows = o_t < i_w

                        b_dy_head = tl.load(
                            dy + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)
                        if ACTIVATION == "swish" or ACTIVATION == "silu":
                            b_y_head = tl.load(
                                y + bos * D + o_t[:, None] * D + o_d,
                                mask=(mask_head_rows[:, None] & m_d[None, :]),
                                other=0.0,
                            ).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t

                        mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                        b_xc = tl.load(
                            initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                            mask=(mask_c[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)

                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                b_wdy = (
                    b_dy_shift
                    if not HAS_WEIGHT
                    else (b_dy_shift * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1]))
                )
                b_dx += b_wdy

            if USE_INITIAL_STATE:
                p_dy0 = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_dy0 = tl.load(p_dy0, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y0 = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                    b_y0 = tl.load(p_y0, boundary_check=(0, 1)).to(tl.float32)
                    b_ys0 = tl.sigmoid(b_y0)
                    b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))

                for i_w in tl.static_range(1, W):
                    m_rows = o_t < i_w
                    if HAS_WEIGHT:
                        w_idx_rows = i_w - 1 - o_t

                        w_mask = o_w[None, :] == w_idx_rows[:, None]
                        w_pick = tl.sum(tl.trans(b_w)[None, :, :] * w_mask[:, None, :], 2)
                    else:
                        w_pick = 1.0
                    contrib = (b_dy0 * w_pick).to(tl.float32)
                    contrib = tl.where(m_rows[:, None] & m_d[None, :], contrib, 0.0)

                    b_dh0_s = tl.sum(contrib, 0)

                    tl.store(
                        dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                        b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
                        mask=m_d,
                    )

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if i_t * BT + BT >= T_len - W:
                start_tok = max(0, T_len - (W - 1))
                offset = i_t * BT + tl.arange(0, BT)
                tok_idx = offset - start_tok
                mask = (offset >= start_tok) & (offset < T_len)
                w_idx = 1 + tok_idx
                dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
                b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
                b_dx += b_dht

        p_dx = tl.make_block_ptr(dx + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = eos - BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - BW + tl.arange(0, BW)
    m_t = o_t >= tl.maximum(bos, eos - W)
    m_d = o_d < D
    m_w = (o_w >= 0) & (o_w < W)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
    if USE_INITIAL_STATE:
        if T < BW:
            o_c = W - (BW - T) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(initial_state + i_n * D * W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
            b_x += b_cache

    tl.store(final_state + i_n * D * W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, T, D, W = *x.shape, state_len
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 256)
    BW = W
    grid = (triton.cdiv(D, BD), N)
    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        BW=BW,
        BD=BD,
    )
    return final_state


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_fwd_impl_old(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."
    B, T, D, W = *x.shape, weight.shape[0]

    NUM_CORES = get_num_cores()

    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    BD = 256
    assert D % BD == 0, "D must be divisible by BD."
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None

        NUM_CHKS = triton.cdiv(T, BT) * B

    y = torch.empty_like(x)

    grid = (NUM_CORES,)

    causal_conv1d_fwd_kernel_old[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
    )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    return y.view(shape), final_state


@input_guard(make_contiguous=True, auto_to_device=True)
def causal_conv1d_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."
    B, T, D, W = *x.shape, weight.shape[0]

    NUM_CORES = get_num_cores()

    BT = min(32, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    BD = 256
    assert D % BD == 0, "D must be divisible by BD."
    NUM_BLKS_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NUM_CHKS = len(chunk_indices)
    else:
        chunk_indices = None

        NUM_CHKS = triton.cdiv(T, BT) * B

    y = torch.empty_like(x)

    grid = (NUM_CORES,)

    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        D=D,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_CHKS=NUM_CHKS,
        NUM_BLKS_D=NUM_BLKS_D,
    )

    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    return y.reshape(B, H, T, D // H), final_state


def causal_conv1d_bwd_impl(
    x: torch.Tensor,
    dy: torch.Tensor,
    H: int,
    dht: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
):

    shape = x.shape
    assert x.shape[-1] == weight.shape[-1], "x [B, T, D], weight [W, D], please check."

    B, T, D = x.shape
    _,H,_,_ = dy.shape
    W = weight.shape[0] if weight is not None else None

    NUM_CORES = get_num_cores()
    BT = min(4, triton.next_power_of_2(triton.cdiv(max(16, B * T), NUM_CORES)))

    BD = 512

    assert D % BD == 0, "D must be divisible by BD."
    NUM_Blk_D = triton.cdiv(D, BD)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NUM_CHKS = len(chunk_indices)

        NT = len(chunk_indices)
    else:
        chunk_indices = None

        NT = triton.cdiv(T, BT)
        NUM_CHKS = NT * B

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd_impl_old(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B * NT, W, D, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B * NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None

    if initial_state is not None:
        if cu_seqlens is not None:
            eff_NT = len(chunk_indices)
        else:
            eff_NT = triton.cdiv(T, BT)

        dh0 = initial_state.new_zeros(min(eff_NT, triton.cdiv(W, BT)), *initial_state.shape)
    else:
        dh0 = None

    grid = (NUM_CORES,)

    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dh0=dh0,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        H=H,
        W=W,
        BT=BT,
        BD=BD,
        ACTIVATION=activation,
        NUM_Blk_D=NUM_Blk_D,
        NUM_CHKS=NUM_CHKS,
        multibuffer=False,
    )

    if weight is not None:
        dw = dw.sum(0).contiguous().to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    if initial_state is not None:
        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)

    return dx.view(shape), dw, db, dr, dh0


# === PyTorch autograd entry (vendored from xtuner causal_conv1d.py) ===

__all__ = ["causal_conv1d_triton", "CausalConv1dFunction"]


class CausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: str = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ):
        weight = weight.transpose(-1, -2).contiguous()
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.H = H

        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=weight,
            H=H,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
        )
        ctx.final_state = final_state

        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        activation = ctx.activation
        cu_seqlens = ctx.cu_seqlens
        H = ctx.H

        dx, dw, db, dr, dh0 = causal_conv1d_bwd_impl(
            x=x,
            dy=dy,
            H=H,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
        )

        return dx, dw.transpose(0, 1).contiguous(), None, db, dr, dh0, None, None, None


def causal_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: str = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Causal 1D convolution (NPU Triton) with forward/backward."""
    return CausalConv1dFunction.apply(
        x,
        weight,
        H,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )
