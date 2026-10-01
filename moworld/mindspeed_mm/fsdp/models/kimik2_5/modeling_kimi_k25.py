# pylint: disable=too-many-lines,duplicate-code
# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for Kimi-K2.5.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import activations

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh
from transformers.activations import PytorchGELUTanh
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available

from mindspeed_mm.fsdp.loss.loss_func import build_loss_func
from mindspeed_mm.fsdp.models.base_model import WeightInitMixin
from mindspeed_mm.fsdp.models.kimik2_5.configuration_kimi_k25 import KimiK25Config
from mindspeed_mm.fsdp.models.kimik2_5.modeling_deepseek import DeepseekV3ForCausalLM, set_seq_len, get_seq_len
from mindspeed_mm.fsdp.utils.register import model_register
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
    all_to_all,
    gather_forward_split_backward,
    packed_data_split_forward_gather_backward_with_cp,
)
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu

_SIN = None
_COS = None


def set_global_param(param_name: str = None, param: Optional[torch.Tensor] = None) -> None:
    if param_name == "sin":
        global _SIN
        _SIN = param
    elif param_name == "cos":
        global _COS
        _COS = param
    else:
        raise ValueError(f"Invalid param type: '{param_name}'.")


def get_global_param(param_name: str = None) -> Optional[torch.Tensor]:
    if param_name == "sin":
        return _SIN
    elif param_name == "cos":
        return _COS
    else:
        raise ValueError(f"Invalid param type: '{param_name}'.")


# Flash attention imports
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


def multihead_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Union[tuple, torch.Tensor] | None = None,
    k_cu_seqlens: Union[tuple, torch.Tensor] | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    deterministic: bool = False,
):
    """Multi-head attention using flash attention 2.

    Args:
        q, k, v: tensor of shape (batch_size, seqlen, num_heads, head_dim),
            or (tot_seqlens, num_heads, head_dim) if packing.
        q_cu_seqlens (Union[tuple, torch.Tensor]): cumulative sequence lengths of q.
        k_cu_seqlens (Union[tuple, torch.Tensor]): cumulative sequence lengths of k.

    Returns:
        output: shape (batch_size, seqlen, dim) or (tot_seqlens, dim) if packing,
            where dim = num_heads * head_dim
    """

    # Modification start, ulysses cp
    ps = get_parallel_state()
    is_ulysses_enabled = ps.is_ulysses_enable()
    total_seq_len = get_seq_len("visual")
    head_num = q.shape[1]
    kv_head_num = k.shape[1]

    # ulysses validation
    if is_ulysses_enabled:
        ulysses_size = ps.get_ulysses_group_size()
        if head_num % ulysses_size != 0:
            raise ValueError(f"num_query_heads ({head_num}) must be divisible by ulysses_size ({ulysses_size})")
        if ulysses_size > kv_head_num:
            if ulysses_size % kv_head_num != 0:
                raise ValueError(
                    f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
                )
            n_repeat = ulysses_size // kv_head_num
            # Shape before: (total_seq_len, kv_head_num, head_dim)
            # This repeats the K/V heads (dim 1) to match the ulysses_size (SP world size)
            # Shape after: (total_seq_len, kv_head_num * n_repeat, head_dim) where (kv_head_num * n_repeat) == ulysses_size
            k = torch.repeat_interleave(k, dim=1, repeats=n_repeat)
            v = torch.repeat_interleave(v, dim=1, repeats=n_repeat)

    if is_ulysses_enabled:
        q = all_to_all(q, ps.get_ulysses_group(), scatter_dim=1, gather_dim=0, gather_size=total_seq_len)
        k = all_to_all(k, ps.get_ulysses_group(), scatter_dim=1, gather_dim=0, gather_size=total_seq_len)
        v = all_to_all(v, ps.get_ulysses_group(), scatter_dim=1, gather_dim=0, gather_size=total_seq_len)

    # Modification end, ulysses cp

    if IS_NPU_AVAILABLE:
        # Modification start
        attn_out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=q.shape[1],
            pse=None,
            atten_mask=None,
            scale=1.0 / math.sqrt(q.shape[-1]),
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=q_cu_seqlens,
            actual_seq_kvlen=k_cu_seqlens
        )[0]
        # Modification end
    else:
        attn_out = flash_attn_varlen_func(
            q,
            k,
            v,
            q_cu_seqlens,
            k_cu_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            causal=False,
            deterministic=deterministic,
        )

    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]

    if is_ulysses_enabled:
        attn_out = all_to_all(attn_out, ps.get_ulysses_group(), scatter_dim=0, gather_dim=1)

    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[
            ...,
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
            q_cu_seqlens[i - 1] : q_cu_seqlens[i],
        ] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "eager": eager_attention,
}


def _apply_rope_input_validation(x, freqs_cis):
    if x.ndim != freqs_cis.ndim + 1:
        raise AssertionError(f"x.ndim ({x.ndim}) should be equal to freqs_cis.ndim + 1 ({freqs_cis.ndim + 1})")

    if x.shape[:-2] != freqs_cis.shape[:-1]:
        raise AssertionError(
            f"x.shape[:-2] ({x.shape[:-2]}) should equal freqs_cis.shape[:-1] ({freqs_cis.shape[:-1]})"
        )

    if x.shape[-1] != 2 * freqs_cis.shape[-1]:
        raise AssertionError(
            f"x.shape[-1] ({x.shape[-1]}) should be twice freqs_cis.shape[-1] ({2 * freqs_cis.shape[-1]})"
        )

    if freqs_cis.dtype != torch.complex64:
        raise AssertionError(f"freqs_cis.dtype ({freqs_cis.dtype}) should be torch.complex64")


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    if IS_NPU_AVAILABLE:
        # Modification start
        cos = get_global_param("cos")
        sin = get_global_param("sin")
        xq_out = torch_npu.npu_rotary_mul(xq.float(), cos, sin, rotary_mode="interleave")
        xk_out = torch_npu.npu_rotary_mul(xk.float(), cos, sin, rotary_mode="interleave")
        # Modification end
    else:
        freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
        # ..., num_heads, head_dim/2
        xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise AssertionError(f"Dimension mismatch: embed_dim ({embed_dim}) must be divisible by 2")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    def __init__(self, height: int, width: int, num_frames: int, dim: int, interpolation_mode: str = 'bicubic') -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            'time_weight',
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames)).float().unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor,
                grid_thws_list: list) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws_list:
            if t > self.num_frames:
                raise AssertionError(f't:{t} > self.num_frames:{self.num_frames}')
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = 'divided_fixed',
    ):
        super().__init__()
        if not isinstance(patch_size, int | Sequence):
            raise AssertionError(f'Invalid patch_size type: {type(patch_size)}')
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise AssertionError(f'Expected patch_size to be a tuple of 2, got {patch_size}')
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        if pos_emb_type == 'divided_fixed':
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height, width=pos_emb_width, num_frames=pos_emb_time, dim=out_dim
            )
        else:
            raise NotImplementedError(f'Not support pos_emb_type: {pos_emb_type}')

    def forward(self, x: torch.Tensor,
                grid_thws_list: list) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws_list)
        return x


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        if self.dim % 4 != 0:
            raise AssertionError('dim must be divisible by 4')
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f'dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}'

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_thws_list: list,
                      device: torch.device) -> torch.Tensor:
        """
        Args:
            grid_thws_list (list): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, 'freqs_cis'):
            self.register_buffer('freqs_cis', self._precompute_freqs_cis(device), persistent=False)

        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in grid_thws_list):
            raise AssertionError(
                "Some (h, w) values are out of bounds (1<=h<=self.max_height, 1<=w<=self.max_width)."
            )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in grid_thws_list
            ],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(self, dims: list[int], activation, bias=True):
        super().__init__()
        if len(dims) != 3:
            raise AssertionError(f"dims should be 3, but {len(dims)}")
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            nn.init.trunc_normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = 'flash_attention_2',
        activation=F.gelu,
        attn_bias: bool = False,
        use_deterministic_attn: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation
        self.use_deterministic_attn = use_deterministic_attn

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: Union[tuple, torch.Tensor],
        max_seqlen: torch.Tensor,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        """
        Args:
            x (torch.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (torch.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (total_seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(
            xq,
            xk,
            xv,
            q_cu_seqlens=cu_seqlens,
            k_cu_seqlens=cu_seqlens,
            max_seqlen_k=max_seqlen,
            max_seqlen_q=max_seqlen,
            deterministic=self.use_deterministic_attn,
        )

        attn_out = self.wo(attn_out)
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Union[tuple, torch.Tensor],
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attention_qkvpacked(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    def __init__(
        self, hidden_dim: int, num_layers: int, block_cfg: dict, video_attn_type: str = 'spatial_temporal'
    ) -> None:
        super().__init__()

        if video_attn_type != 'spatial_temporal':
            raise AssertionError(f'video_attn_type must be "spatial_temporal", got {video_attn_type}')
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(block_cfg['hidden_dim'] // block_cfg['num_heads'], 512, 512)
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(**block_cfg, use_deterministic_attn=False)  # Modification
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
        grid_thws_list: list,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws_list=grid_thws_list, device=hidden_states.device)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0,
                                                             dtype=torch.int32)
        if IS_NPU_AVAILABLE:
            cu_seqlens = tuple(cu_seqlens[1:].cpu().numpy().tolist())

        # Modification start: ulysses cp
        seq_len, _ = hidden_states.size()
        sequence_lengths = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cpu()
        set_seq_len("visual", seq_len)
        set_seq_len("per_visual", sequence_lengths)

        ps = get_parallel_state()
        # Split sequences across context parallel groups for distributed processing
        if ps.is_cp_enable():
            hidden_states = packed_data_split_forward_gather_backward_with_cp(
                hidden_states, dim=0, seq_lens=sequence_lengths
            )
            rope_freqs_cis = packed_data_split_forward_gather_backward_with_cp(
                rope_freqs_cis, dim=0, seq_lens=sequence_lengths
            )

        cos = rope_freqs_cis.unsqueeze(-2).real.to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        sin = rope_freqs_cis.unsqueeze(-2).imag.to(torch.float32).repeat_interleave(2, dim=-1).contiguous()
        set_global_param("cos", cos)
        set_global_param("sin", sin)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis)

        ps = get_parallel_state()
        if ps.is_cp_enable():
            gather_sizes = cal_split_sizes(get_seq_len("visual"), ps.get_ulysses_group_size())
            hidden_states = gather_forward_split_backward(
                hidden_states,
                ps.get_ulysses_group(),
                dim=0,
                grad_scale="up",
                gather_sizes=gather_sizes,
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
        x: torch.Tensor,
        grid_thws_list: list,
        merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws_list:
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViT3dPretrainedModel(PreTrainedModel):
    config_class = None
    model_type = 'moonvit3d'
    _no_split_modules = ['PackingTransformer']
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                'num_heads': config.num_attention_heads,
                'hidden_dim': config.hidden_size,
                'mlp_dim': config.intermediate_size,
                'activation': PytorchGELUTanh(),
                'attn_bias': True,
                'attn_implementation': config._attn_implementation,
            },
            video_attn_type=config.video_attn_type,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        if grid_thws.ndim != 2:
            raise AssertionError(f'grid_thws should be 2D, got {grid_thws.ndim}')
        if grid_thws.size(1) != 3:
            raise AssertionError(f'No support for thw: {grid_thws}')
        grid_thws_list = grid_thws.tolist()
        hidden_states = self.patch_embed(pixel_values, grid_thws_list)
        hidden_states = self.encoder(hidden_states, grid_thws, grid_thws_list)
        if self.merge_type == 'sd2_tpool':  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(
                hidden_states,
                grid_thws_list,
                merge_kernel_size=self.merge_kernel_size)
        else:
            raise NotImplementedError(f'Not support {self.merge_type}')

        return hidden_states


# ============================================================================
# MM Projector Helper Classes (from mm_projector/modeling_mm_projectors.py)
# ============================================================================


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # use faster LayerNorm
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if not isinstance(x, list | tuple):
            raise AssertionError(f'x is not a list or tuple: {type(x)}')
        lengths = [item.shape[0] for item in x]
        x = torch.cat(x, dim=0)
        x = self.pre_norm(x)
        x = self.proj(x)
        x = torch.split(x, lengths, dim=0)

        return x


class PatchMergerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        eps = config.projector_ln_eps
        self.hidden_size = config.mm_hidden_size * (config.merge_kernel_size[0] * config.merge_kernel_size[1])
        self.pre_norm = nn.LayerNorm(config.mm_hidden_size, eps=eps)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            x = [self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x]
        else:
            # B, N, N_k, C = x.shape
            B = x.shape[0]
            x = self.proj(self.pre_norm(x).view(B, -1, self.hidden_size))
        return x


class KimiK25PreTrainedModel(PreTrainedModel):
    config_class = KimiK25Config
    base_model_prefix = "model"
    _no_split_modules = [
        "MoonViT3dPretrainedModel",
        "MoonViTEncoderLayer",
        "DeepseekDecoderLayer",
        "PatchMergerMLP",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class VisionTowerConfig(PretrainedConfig):
    model_type = 'moonvit3d'

    def __init__(self, config: KimiK25Config, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = config.patch_size
        self.init_pos_emb_height = config.init_pos_emb_height
        self.init_pos_emb_width = config.init_pos_emb_width
        self.init_pos_emb_time = config.init_pos_emb_time
        self.pos_emb_type = config.pos_emb_type
        self.num_attention_heads = config.vt_num_attention_heads
        self.num_hidden_layers = config.vt_num_hidden_layers
        self.hidden_size = config.vt_hidden_size
        self.intermediate_size = config.vt_intermediate_size
        self.merge_kernel_size = config.merge_kernel_size
        self.video_attn_type = config.video_attn_type
        self.merge_type = config.merge_type
        self._attn_implementation = config._attn_implementation


class ProjectorConfig:
    def __init__(self, config: KimiK25Config):
        self.mm_projector_type = config.mm_projector_type
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.text_hidden_size
        self.merge_kernel_size = config.merge_kernel_size
        self.projector_hidden_act = config.projector_hidden_act
        self.projector_ln_eps = config.projector_ln_eps


# Modification
@model_register.register("kimi_k25")
class KimiK25ForConditionalGeneration(WeightInitMixin, KimiK25PreTrainedModel):
    def __init__(self, config: KimiK25Config):
        super().__init__(config)

        vt_config = VisionTowerConfig(config.vision_config)
        self.vision_tower = MoonViT3dPretrainedModel(vt_config)

        proj_config = ProjectorConfig(config.vision_config)
        if proj_config.mm_projector_type == 'identity':
            self.mm_projector = IdentityMap()
        elif proj_config.mm_projector_type == 'mlp':
            self.mm_projector = MLP(proj_config)
        elif proj_config.mm_projector_type == 'patchmerger':
            self.mm_projector = PatchMergerMLP(proj_config)
        else:
            raise ValueError(f"Unsupported mm_projector_type: {proj_config.mm_projector_type}")

        self.language_model = DeepseekV3ForCausalLM(config.text_config)
        self.vocab_size = None
        self.loss_function = None

        if hasattr(self.language_model, 'dtype'):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)

    def set_modules_to_prefetch(self, fsdp_plan, ep_plan):
        if fsdp_plan.num_to_forward_prefetch > 0:
            self.language_model.model.embed_tokens.set_modules_to_forward_prefetch([self.vision_tower])
            self.vision_tower.set_modules_to_forward_prefetch([self.vision_tower.encoder.blocks[0]])
            for idx in range(len(self.vision_tower.encoder.blocks)-1):
                self.vision_tower.encoder.blocks[idx].set_modules_to_forward_prefetch([self.vision_tower.encoder.blocks[idx+1]])
            self.vision_tower.encoder.blocks[-1].set_modules_to_forward_prefetch([self.mm_projector])
            self.mm_projector.set_modules_to_forward_prefetch([self.language_model.model.layers[0]])
            self.language_model.model.layers[0].set_modules_to_forward_prefetch([self.language_model.model.layers[1].mlp.experts, self.language_model.model.layers[1]])

            for idx in range(1, len(self.language_model.model.layers)-1):
                self.language_model.model.layers[idx].set_modules_to_forward_prefetch([self.language_model.model.layers[idx+1].mlp.experts, self.language_model.model.layers[idx+1]])
            self.language_model.model.layers[-1].set_modules_to_forward_prefetch([self.language_model.model, self.language_model.lm_head])

        if fsdp_plan.num_to_backward_prefetch > 0:
            self.language_model.lm_head.set_modules_to_backward_prefetch([self.language_model.model, self.language_model.model.layers[-1].mlp.experts, self.language_model.model.layers[-1]])

            for idx in range(len(self.language_model.model.layers)-2, 2, -1):
                self.language_model.model.layers[idx].set_modules_to_backward_prefetch([self.language_model.model.layers[idx-1].mlp.experts, self.language_model.model.layers[idx-1]])
            self.language_model.model.layers[1].set_modules_to_backward_prefetch([self.language_model.model.layers[0]])

            self.language_model.model.layers[0].set_modules_to_backward_prefetch([self.mm_projector])
            self.mm_projector.set_modules_to_backward_prefetch([self.vision_tower.encoder.blocks[-1]])
            for idx in range(len(self.vision_tower.encoder.blocks)-1, 0, -1):
                self.vision_tower.encoder.blocks[idx].set_modules_to_backward_prefetch([self.vision_tower.encoder.blocks[idx-1]])
            self.vision_tower.encoder.blocks[0].set_modules_to_backward_prefetch([self.vision_tower])
            self.vision_tower.set_modules_to_backward_prefetch([self.language_model.model.embed_tokens])
        return True

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """
        Args:
            image_features (:obj:`torch.Tensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The image features to merge with the input embeddings.
            inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, embed_dim)`):
                The input embeddings.
            input_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The attention mask.
            labels (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
                The labels.
        """
        image_token_index: int = self.config.media_placeholder_token_id
        pad_token_id: int = self.config.pad_token_id
        ignore_index: int = self.config.ignore_index

        if self.training:
            pad_mask = input_ids == pad_token_id
            image_mask = input_ids == image_token_index
            image_features = torch.cat(image_features, dim=0)

            n_image_tokens = image_mask.sum()
            if image_features is not None and n_image_tokens != image_features.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
                )

            inputs_embeds[image_mask] = image_features
            inputs_embeds *= ~pad_mask.unsqueeze(-1)
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(attention_mask == 0, 1)

            return inputs_embeds, attention_mask, labels, position_ids

        _, embed_dim = image_features[0].shape
        feature_lengths = [x.shape[0] for x in image_features]
        image_features = torch.cat(image_features, dim=0)

        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))

        # 1. Create a mask to know where special image tokens are
        _token_occupation_table = torch.ones_like(input_ids.flatten())
        _token_occupation_table[input_ids.flatten() == image_token_index] = torch.tensor(
            feature_lengths, dtype=torch.long, device=input_ids.device
        )
        _token_occupation_table = _token_occupation_table.reshape(input_ids.shape)

        max_embed_dim = _token_occupation_table.sum(-1).max().item()
        if max_embed_dim < sequence_length:
            raise AssertionError(f"The max_embed_dim({max_embed_dim}) is less than sequence_length({sequence_length})")
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        new_token_positions = torch.cumsum(_token_occupation_table, -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask.
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {image_to_overwrite.sum()} while"
                f" the number of image features given to the model is {image_features.shape[:-1].numel()}. "
                "This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _extract_image_features(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
                The pixel values of the images processed by image processor.
            grid_thws (:obj:`torch.Tensor` of shape :obj:`(batch_size, 3)`):
                The grid, height, width of the images.

        Returns:
            selected_image_feature (:obj:`torch.FloatTensor` of shape :obj:`(num_image_tokens, embed_dim)`):
                The selected image features to use as input to the projector head.

        """

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)

        image_features = self.vision_tower(pixel_values, grid_thws)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | list[torch.FloatTensor] | None = None,
        grid_thws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | LlavaCausalLMOutputWithPast:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        ```"""
        use_cache = False  # Modification
        if self.vision_tower is None:
            raise AssertionError("vision_tower is not loaded")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and len(pixel_values) > 0 and input_ids.shape[1] != 1:
                image_features = self._extract_image_features(pixel_values, grid_thws)
                if self.mm_projector:
                    image_features = self.mm_projector(image_features)

                inputs_embeds = inputs_embeds.to(image_features[0].dtype)  # num_tokens, embed_dim
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    labels,
                )

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Modification: start，kimi性能优化，loss计算优化，适配chunkloss
        hidden_states = outputs.hidden_states

        if getattr(self, "enable_chunk_loss", False):
            logits = None
            chunk_size = getattr(self, "chunk_size", 1024)
            self.loss_function = build_loss_func(loss_type="default", chunk_size=chunk_size, labels=labels)
            loss = self.language_model.lm_head(hidden_states, self.loss_function)
        else:
            logits = self.language_model.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[..., 1:]
                    shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1).to(shift_logits.device),
                )
        # Modification：end

        ps = get_parallel_state()
        if ps.is_cp_enable():
            loss = gather_forward_split_backward(loss.unsqueeze(0), ps.get_cp_group(), dim=0)
            loss = loss.sum()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        grid_thws=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = getattr(past_key_values, 'seen_tokens', cache_length)
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.media_placeholder_token_id in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "grid_thws": grid_thws,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
