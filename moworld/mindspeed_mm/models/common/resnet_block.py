from typing import Optional, Dict

import torch
from torch import nn
from diffusers.models.activations import get_activation
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm

from mindspeed_mm.utils.utils import video_to_image
from mindspeed_mm.models.common.conv import (
    CausalConv3d,
    SafeConv3d,
    ContextParallelCausalConv3d,
    WfCausalConv3d,
    TimePaddingCausalConv3d
)
from mindspeed_mm.models.common.updownsample import UpsampleCausal3D, DownsampleCausal3D
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.activations import Sigmoid, get_activation_layer


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        dropout,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_shortcut=False,
        num_groups=32,
        eps=1e-6,
        affine=True,
        norm_type="groupnorm",
        act_type="silu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.norm2 = normalize(out_channels, num_groups, eps, affine, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = get_activation_layer(act_type)()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    @video_to_image
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        x = x + h
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
        conv_shortcut=False,
        dropout=0,
        norm_type="groupnorm",
        conv_type="CausalConv3d",
        enable_vae_cp=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.enable_vae_cp = enable_vae_cp
        if conv_type == "WfCausalConv3d":
            ConvLayer = ContextParallelCausalConv3d if self.enable_vae_cp else WfCausalConv3d
        elif conv_type == "CausalConv3d":
            ConvLayer = CausalConv3d
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm2 = normalize(out_channels, num_groups, eps, affine, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.SiLU()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ConvLayer(in_channels, out_channels, kernel_size, padding=padding)
            else:
                self.nin_shortcut = ConvLayer(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h_conv1 = self.conv1(h)
        h = h_conv1[0] if isinstance(h_conv1, tuple) else h_conv1

        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h_conv2 = self.conv2(h)
        h = h_conv2[0] if isinstance(h_conv2, tuple) else h_conv2

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x_conv_shortcut = self.conv_shortcut(x)
                x = x_conv_shortcut[0] if isinstance(x_conv_shortcut, tuple) else x_conv_shortcut
            else:
                x_nin_shortcut = self.nin_shortcut(x)
                x = x_nin_shortcut[0] if isinstance(x_nin_shortcut, tuple) else x_nin_shortcut
        return x + h


class ContextParallelResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        gather_norm=False,
        normalization=normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )

        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
        self.act = Sigmoid()

    def forward(self, x, temb=None, zq=None, clear_fake_cp_cache=True, enable_cp=True, is_encode=True,
                           conv_cache: Optional[Dict[str, torch.Tensor]] = None, use_conv_cache=False):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        h = x

        if zq is not None:
            h = self.norm1(h, zq, clear_fake_cp_cache=clear_fake_cp_cache, enable_cp=enable_cp)
        else:
            h = self.norm1(h)

        h = self.act(h)
        # i2v train image encode need close vae-cp
        conv_enable_cp = enable_cp if is_encode else True
        h, new_conv_cache["conv1"] = self.conv1(h, clear_cache=clear_fake_cp_cache, enable_cp=conv_enable_cp,
                                                conv_cache=conv_cache.get("conv1"),
                                                use_conv_cache=use_conv_cache)

        if temb is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None, None]

        if zq is not None:
            h = self.norm2(h, zq, clear_fake_cp_cache=clear_fake_cp_cache, enable_cp=enable_cp)
        else:
            h = self.norm2(h)

        h = self.act(h)
        h = self.dropout(h)
        h, new_conv_cache["conv2"] = self.conv2(h, clear_cache=clear_fake_cp_cache, enable_cp=conv_enable_cp,
                                                conv_cache=conv_cache.get("conv2"),
                                                use_conv_cache=use_conv_cache)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x, new_conv_cache["conv_shortcut"] = self.conv_shortcut(x,
                                                                        clear_cache=clear_fake_cp_cache,
                                                                        enable_cp=enable_cp,
                                                                        conv_cache=conv_cache.get("conv_shortcut"),
                                                                        use_conv_cache=use_conv_cache)
            else:
                x = self.nin_shortcut(x)

        return x + h, new_conv_cache


class ResnetBlockCausal3D(nn.Module):
    r"""
    A Resnet block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = TimePaddingCausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"Unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = TimePaddingCausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleCausal3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = DownsampleCausal3D(in_channels, use_conv=False, name="op")

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = TimePaddingCausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor
