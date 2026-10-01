# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import random

import numpy as np
import torch_npu
import torch
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.attention import AdaLayerNorm, FeedForward
from diffusers.models.attention_processor import Attention

from diffusers.utils import BaseOutput, is_torch_version
from einops import rearrange, repeat
from torch import nn
from diffusers.utils.torch_utils import maybe_allow_in_graph

from mindspeed_mm.models.common.embeddings import PatchEmbed2D_3DsincosPE


try:
    from diffusers.models.embeddings import PixArtAlphaTextProjection
except ImportError:
    from diffusers.models.embeddings import \
        CaptionProjection as PixArtAlphaTextProjection


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


@maybe_allow_in_graph
class ProxyTokensTransformerBlock(nn.Module):
    r"""
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        shift_window: bool = False,
        final_dropout: bool = False,
        attention_type: str = "default",
        compress_ratios=None,
        proxy_compress_ratios=None,
    ):
        super().__init__()

        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.proxy_norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proxy_cross_norm = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.proxy_attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
        )


        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.norm_before = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.pvisual_attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None
            self.norm_before = None

        # 2.5 shift windeo attention
        self.shift_window = shift_window
        if self.shift_window:
            self.attn3 = Attention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    cross_attention_dim=None,
                    upcast_attention=upcast_attention,
                )
            self.shift_window_norm = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.linear_2 = zero_module(nn.Linear(dim, dim))

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)
        self.hid_dim = dim
        self.linear_1_visual = zero_module(nn.Linear(self.hid_dim, dim))
        if compress_ratios[0] == 1:
            self.scale_ratio = (1, compress_ratios[1] // proxy_compress_ratios[1], compress_ratios[2] // proxy_compress_ratios[2])
        else:
            self.scale_ratio = (compress_ratios[0] // compress_ratios[0], compress_ratios[1] // proxy_compress_ratios[1], compress_ratios[2] // proxy_compress_ratios[2])


        # 5. Scale-shift for PixArt-Alpha.
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
            if self.shift_window:
                self.proxy_scale_shift_table = nn.Parameter(torch.randn(3, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def window_shift(self, hidden_states, number_proxys, compress_ratios, num_frames, height, width):
        compress_scale = compress_ratios[0] * compress_ratios[1] * compress_ratios[2]
        dim = hidden_states.shape[-1]
        F_new, H_new, W_new = num_frames // compress_ratios[0], height // compress_ratios[1], width // compress_ratios[2]
        f_shift_size, h_shift_size, w_shift_size = compress_ratios[0] // 2, compress_ratios[1] // 2, compress_ratios[2] // 2

        hidden_states = rearrange(hidden_states, "(b p) n c -> b p n c", p=number_proxys)
        hidden_states = hidden_states.reshape(-1, F_new, H_new, W_new, compress_ratios[0], compress_ratios[1], compress_ratios[2], dim)
        hidden_states = rearrange(hidden_states, "b f h w x y z c -> b (f x) (h y) (w z) c")

        after_shift_hidden_states = torch.roll(hidden_states, shifts=(-f_shift_size, -h_shift_size, -w_shift_size), dims=(1, 2, 3))

        after_shift_hidden_states = rearrange(after_shift_hidden_states, "b (f x) (h y) (w z) c -> b f h w x y z c",
                                              f=F_new, h=H_new, w=W_new)

        after_shift_hidden_states = rearrange(after_shift_hidden_states, "b f h w x y z c -> (b f h w) (x y z) c")

        return after_shift_hidden_states

    def window_Ishift(self, after_shift_hidden_states, number_proxys, compress_ratios, num_frames, height, width):
        compress_scale = compress_ratios[0] * compress_ratios[1] * compress_ratios[2]
        dim = after_shift_hidden_states.shape[-1]
        F_new, H_new, W_new = num_frames // compress_ratios[0], height // compress_ratios[1], width // compress_ratios[2]
        f_shift_size, h_shift_size, w_shift_size = compress_ratios[0] // 2, compress_ratios[1] // 2, compress_ratios[2] // 2

        after_shift_hidden_states = rearrange(after_shift_hidden_states, "(b p) n c -> b p n c", p=number_proxys)
        after_shift_hidden_states = after_shift_hidden_states.reshape(-1, F_new, H_new, W_new, compress_ratios[0], compress_ratios[1], compress_ratios[2], dim)
        after_shift_hidden_states = rearrange(after_shift_hidden_states, "b f h w x y z c -> b (f x) (h y) (w z) c")

        hidden_states = torch.roll(after_shift_hidden_states, shifts=(f_shift_size, h_shift_size, w_shift_size), dims=(1, 2, 3))

        hidden_states = rearrange(hidden_states, "b (f x) (h y) (w z) c -> b f h w x y z c",
                                              f=F_new, h=H_new, w=W_new)

        hidden_states = rearrange(hidden_states, "b f h w x y z c -> (b f h w) (x y z) c")

        return hidden_states

    def get_proxy_token(self, hidden_states, compress_ratios, scale_ratio, window_number, mode='first'):
        number, dim = hidden_states.shape[1], hidden_states.shape[-1]
        f, h, w = compress_ratios[0] // scale_ratio[0], compress_ratios[1] // scale_ratio[1], compress_ratios[2] // scale_ratio[2]
        proxy_hidden_states = hidden_states.reshape(hidden_states.shape[0], *compress_ratios, dim)
        proxy_hidden_states = proxy_hidden_states.reshape(hidden_states.shape[0], f, scale_ratio[0], h, scale_ratio[1],
                                                            w, scale_ratio[2], dim)
        proxy_hidden_states = rearrange(proxy_hidden_states, "b f x h y w z c -> b (f h w) (x y z) c")
        if mode == 'first':
            proxy_hidden_states = proxy_hidden_states[:, 0, :, :]
        elif mode == 'mean':
            proxy_hidden_states = proxy_hidden_states.mean(1)
        proxy_hidden_states = rearrange(proxy_hidden_states, "(b p) n c -> b (p n) c", p=window_number)
        return proxy_hidden_states


    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        shift_window_attention_mask: Optional[torch.FloatTensor] = None,
        compress_ratios: Optional[torch.FloatTensor] = None,
        proxy_compress_ratios: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        num_frames: int = 16,
        height: int = 32,
        width: int = 32,
    ) -> torch.FloatTensor:

        compress_f, compress_h, compress_w = num_frames // compress_ratios[0], height // compress_ratios[1], width // compress_ratios[2]
        window_number = compress_f * compress_h * compress_w

        p_compress_f, p_compress_h, p_compress_w = num_frames // proxy_compress_ratios[0], height // proxy_compress_ratios[1], width // proxy_compress_ratios[2]
        proxy_token_number = p_compress_f * p_compress_h * p_compress_w

        h_dtype = hidden_states.dtype
        batch_size, dim = hidden_states.shape[0] // window_number, hidden_states.shape[-1]  # hidden_states shape of b * p, n, c

        if self.use_ada_layer_norm_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1).repeat(window_number, 1, 1)
            ).chunk(6, dim=1)
            if self.shift_window:
                shift_sw_msa, scale_sw_msa, gate_sw_msa = (
                    self.proxy_scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)[:, :3, :].repeat(window_number, 1, 1)
                ).chunk(3, dim=1)
        else:
            raise ValueError("Incorrect norm used")

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)  # None

        #1.----------------------------------proxy token self attention-----------------------------------
        proxy_hidden_states = self.get_proxy_token(hidden_states, compress_ratios, self.scale_ratio, window_number, mode="mean")

        norm_proxy_hidden_states = self.proxy_norm1(proxy_hidden_states)
        proxy_attn_output = self.proxy_attn1(
            norm_proxy_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        proxy_hidden_states = proxy_attn_output + proxy_hidden_states
        if proxy_hidden_states.ndim == 4:
            proxy_hidden_states = proxy_hidden_states.squeeze(1)

        #2.------------------------------------proxy visual cross-attention-----------------------------------------
        if self.pvisual_attn2 is not None:
            a_hidden_states = rearrange(hidden_states, "(b w) n c -> b (w n) c", w=window_number)
            pvisual_hidden_states = self.proxy_cross_norm(proxy_hidden_states)
            a_hidden_states = self.norm_before(a_hidden_states)
            pvisual_attn_output = self.pvisual_attn2(
                a_hidden_states,
                encoder_hidden_states=pvisual_hidden_states,
                attention_mask=None,
                **cross_attention_kwargs,
            )
            pvisual_attn_output = rearrange(pvisual_attn_output, "b (w n) c -> (b w) n c", w=window_number)
            hidden_states = hidden_states + self.linear_1_visual(pvisual_attn_output)

        #3.--------------------------------window Self-Attention--------------------------------
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        #4.---------------------------shift window Self-Attention------------------------------------
        if self.shift_window:
            after_shift_hidden_states = self.window_shift(hidden_states, window_number,
                                                        compress_ratios,
                                                        num_frames, height, width)

            after_shift_norm_hidden_states = self.shift_window_norm(after_shift_hidden_states)
            after_shift_norm_hidden_states = after_shift_norm_hidden_states * (1 + scale_sw_msa) + shift_sw_msa

            after_shift_attn_output = self.attn3(
                after_shift_norm_hidden_states,
                encoder_hidden_states=None,
                attention_mask=shift_window_attention_mask,
                **cross_attention_kwargs,
            )
            attn_output = self.window_Ishift(after_shift_attn_output, window_number,
                                            compress_ratios,
                                            num_frames, height, width)

            attn_output = gate_sw_msa * attn_output
            hidden_states = self.linear_2(attn_output) + hidden_states

        #5.---------------------------text cross attention------------------------------------

        if self.attn2 is not None:
            hidden_states = rearrange(hidden_states, "(b p) n c -> b (p n) c", p=window_number)

            norm_hidden_states = hidden_states

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
            hidden_states = rearrange(hidden_states, "b (p n) c -> (b p) n c", p=window_number)

        #6.--------------------------------mlp------------------------------------
        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [
                    self.ff(hid_slice, scale=lora_scale)
                    for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                ],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

            self.resolution_embedder.linear_2 = zero_module(self.resolution_embedder.linear_2)
            self.aspect_ratio_embedder.linear_2 = zero_module(self.aspect_ratio_embedder.linear_2)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class PTDiTDiffuser(ModelMixin, ConfigMixin):
    """
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        frame: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        shift_window: bool = False,
        compress_ratios: Optional[list] = None,
        proxy_compress_ratios: Optional[list] = None,
        **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.norm_type = norm_type

        self.compress_ratios = [1, 8, 8] if compress_ratios is None else compress_ratios
        self.proxy_compress_ratios = [1, 4, 4] if proxy_compress_ratios is None else proxy_compress_ratios
        print(f'compress_ratios: {compress_ratios}')
        print(f'proxy_compress_ratios: {proxy_compress_ratios}')

        if sample_size is None:
            raise ValueError("PTDiTDiffuser over patched input must provide sample_size")

        self.height = sample_size
        self.width = sample_size
        self.sample_size = sample_size

        self.patch_size = patch_size
        interpolation_scale = self.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)

        self.pos_embed = PatchEmbed2D_3DsincosPE(
            height=sample_size,
            width=sample_size,
            frame=frame,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                ProxyTokensTransformerBlock(
                    inner_dim,
                    num_heads,
                    head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    shift_window=shift_window,
                    attention_type=attention_type,
                    compress_ratios=self.compress_ratios,
                    proxy_compress_ratios=self.proxy_compress_ratios,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.compress_scale = self.compress_ratios[0] * self.compress_ratios[1] * self.compress_ratios[2]
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            self.use_additional_conditions = self.sample_size == 128
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)


        self.gradient_checkpointing = True
        self.shift_window_attention_mask = None


    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_shift_window_attention_mask(self, video_length, height, weight, compress_ratios, c_dtype, c_device):

        img_mask = torch.zeros((1, video_length, height, weight))

        shift_size = [compress_ratios[0] // 2, compress_ratios[1] // 2, compress_ratios[2] // 2]
        h_slices = (slice(0, -compress_ratios[1]),
                    slice(-compress_ratios[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        w_slices = (slice(0, -compress_ratios[2]),
                    slice(-compress_ratios[2], -shift_size[2]),
                    slice(-shift_size[2], None))
        f_slices = (slice(0, -compress_ratios[0]),
                    slice(-compress_ratios[0], -shift_size[0]),
                    slice(-shift_size[0], None))

        cnt = 0
        for f in f_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, f, h, w] = cnt
                    cnt += 1

        img_mask = img_mask.view(1, video_length // compress_ratios[0], compress_ratios[0],
                                    height // compress_ratios[1], compress_ratios[1],
                                    weight // compress_ratios[2], compress_ratios[2])
        mask_windows = rearrange(img_mask, "b f x h y w z -> (b f h w) (x y z)")
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1.0)).masked_fill(attn_mask == 0, float(0.0))
        self.shift_window_attention_mask = attn_mask.to(device=c_device, dtype=c_dtype)



    def extract_proxy_tokens(self, hidden_states, compress_ratios):
        b, c, f, h, w = hidden_states.shape
        n_f, n_h, n_w = f // compress_ratios[0], h // compress_ratios[1], w // compress_ratios[2]
        hidden_states = hidden_states.reshape(b, c, n_f, compress_ratios[0], n_h, compress_ratios[1], n_w, compress_ratios[2])
        hidden_states = rearrange(hidden_states, "b c f x h y w z -> (b f h w) c x y z")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """
        Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func
        """
        self.input_tensor = input_tensor

    def zero_init_shift_window_attention_linear(self):
        for block in self.transformer_blocks:
            try:
                zero_module(block.linear_2)
            except ValueError as e:
                print(f"An error occurred while zzeroing module: {e}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.

        Returns:
            sample tensor.
        """
        dtype = self.pos_embed.proj.weight.dtype

        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        if added_cond_kwargs is None:
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(encoder_hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)


        # 1. Input new
        video_length, height, width = hidden_states.shape[-3], hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        compress_f, compress_h, compress_w = video_length // self.compress_ratios[0], height // self.compress_ratios[1], width // self.compress_ratios[2]
        number_proxy_tokens = compress_f * compress_h * compress_w
        hidden_states = rearrange(hidden_states, "b c f h w -> b f c h w")

        hidden_states = self.pos_embed(hidden_states)
        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            batch_size = hidden_states.shape[0] // video_length
            t = timestep
            timestep, embedded_timestep = self.adaln_single(
                t, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width)

        hidden_states = self.extract_proxy_tokens(hidden_states, self.compress_ratios)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # 1.5 Shift Window Attention Mask
        if self.shift_window_attention_mask is None:
            self.get_shift_window_attention_mask(video_length, height, width, self.compress_ratios,
                                                  hidden_states.dtype, hidden_states.device)

        if self.shift_window_attention_mask is not None and self.shift_window_attention_mask.ndim == 3:
            shift_window_attention_mask = self.shift_window_attention_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            shift_window_attention_mask = shift_window_attention_mask.reshape(batch_size * number_proxy_tokens, self.compress_scale, self.compress_scale) * 10000.0

        # 2. Blocks
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])


        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                args = [video_length, height, width]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    shift_window_attention_mask,
                    self.compress_ratios,
                    self.proxy_compress_ratios,
                    cross_attention_kwargs,
                    *args,
                    **ckpt_kwargs,
                )
            else:
                kwargs = {"num_frames": video_length, "height": height, "width": width}
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    shift_window_attention_mask=shift_window_attention_mask,
                    compress_ratios=self.compress_ratios,
                    proxy_compress_ratios=self.proxy_compress_ratios,
                    cross_attention_kwargs=cross_attention_kwargs,
                    **kwargs
                )

        # 3. Outputs
        if self.norm_type == "ada_norm_single":
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)

            hidden_states = rearrange(hidden_states, "(b p) n c -> b (p n) c", p=number_proxy_tokens)
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

        hidden_states = rearrange(hidden_states, "b (p n) c -> b p n c", p=number_proxy_tokens)

        hidden_states = hidden_states.reshape(-1, compress_f, compress_h, compress_w, self.compress_ratios[0],
                                                self.compress_ratios[1], self.compress_ratios[2],
                                                self.patch_size * self.patch_size * self.out_channels)

        hidden_states = rearrange(hidden_states, "b f h w x y z c -> b (f x) (h y) (w z) c")

        hidden_states = hidden_states.reshape(
            shape=(-1, video_length, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, video_length, height * self.patch_size, width * self.patch_size)
        )
        output = output.to(torch.float32)
        return output
