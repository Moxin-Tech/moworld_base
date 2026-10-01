# Copyright 2024 The HuggingFace Team. All rights reserved.
#
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
from typing import Optional, Callable
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn

from diffusers.models.activations import GELU, GEGLU, ApproximateGELU
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed_mm.models.common.linear import MatmulAddLinear


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ===============================================
# General-purpose Layers
# ===============================================


class MatmulAddFeedForward(nn.Module):
    r"""
    A feed-forward layer with MatmulAddLinear.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = MatmulAddLinear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states



class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, (T, S), C]
        x = torch.lerp(masked_x, x, x_mask)
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero)
        x = self.linear(x)
        return x


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""
    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable,
        enable_tensor_parallel: bool = False,
        gather_tensor_parallel_output: bool = True,
        zero_initialize: bool = True,
    ):
        super().__init__()
        self.enable_tensor_parallel = enable_tensor_parallel
        self.gather_tensor_parallel_output = gather_tensor_parallel_output and enable_tensor_parallel
        self.act = act_layer()

        if self.enable_tensor_parallel:
            args = get_args()
            config = core_transformer_config_from_args(args)
            config.sequence_parallel = False
            self.linear = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                factor * hidden_size,
                bias=True,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
            self.sequence_parallel = args.sequence_parallel
        else:
            self.linear = nn.Linear(
                hidden_size, factor * hidden_size, bias=True
            )

        if zero_initialize:
            # Zero-initialize the modulation
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_tensor_parallel:
            output = self.linear(self.act(x))[0]
            if self.gather_tensor_parallel_output:
                if self.sequence_parallel:
                    output = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(output)
                else:
                    output = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(output)
            return output
        else:
            return self.linear(self.act(x))


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        enable_tensor_parallel=False,
        enable_tp_sp=True
    ):
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels

        bias = bias if isinstance(bias, tuple) else tuple(repeat(bias, 2))
        drop_probs = drop if isinstance(drop, tuple) else tuple(repeat(drop, 2))
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.enable_tensor_parallel = not use_conv and enable_tensor_parallel

        if self.enable_tensor_parallel:
            args = get_args()
            config = core_transformer_config_from_args(args)
            config.sequence_parallel = enable_tp_sp and args.sequence_parallel
            self.fc1 = tensor_parallel.ColumnParallelLinear(
                in_channels,
                hidden_channels,
                config=config,
                init_method=config.init_method,
                bias=bias[0],
                gather_output=False
            )
        else:
            self.fc1 = linear_layer(
                in_channels, hidden_channels, bias=bias[0]
            )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_channels)
            if norm_layer is not None
            else nn.Identity()
        )

        if norm_layer is not None:
            for param in self.norm.parameters():
                setattr(param, "sequence_parallel", enable_tp_sp and args.sequence_parallel)

        if self.enable_tensor_parallel:
            config.sequence_parallel = enable_tp_sp and args.sequence_parallel
            self.fc2 = tensor_parallel.RowParallelLinear(
                hidden_channels,
                out_features,
                config=config,
                init_method=config.init_method,
                bias=bias[1],
                input_is_parallel=True,
                skip_bias_add=False
            )
        else:
            self.fc2 = linear_layer(
                hidden_channels, out_features, bias=bias[1]
            )
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        if self.enable_tensor_parallel:
            x = self.fc1(x)[0]
        else:
            x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        if self.enable_tensor_parallel:
            x = self.fc2(x)[0]
        else:
            x = self.fc2(x)
        x = self.drop2(x)
        return x
