# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import List, Tuple

import torch
import torch_npu
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from mindspeed_mm.models.common.distrib import DiagonalGaussianDistribution
from mindspeed_mm.models.common.conv import CausalConv3dBase
from mindspeed_mm.models.common.updownsample import Upsample


def base_group_norm(x, norm_layer, act_silu=False):
    x_shape = x.shape
    x = x.flatten(0, 1)
    # Permute to NCHW format
    out = F.group_norm(x.permute(0, 3, 1, 2).contiguous(), norm_layer.num_groups,
                       norm_layer.weight, norm_layer.bias, norm_layer.eps)
    if act_silu:
        out = F.silu(out)

    # Permute back to NHWC format
    out = out.permute(0, 2, 3, 1)
    out = out.view(x_shape)
    return out


def base_conv2d(x, conv_layer, channel_last=False, residual=None):
    if channel_last:
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
    out = F.conv2d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding)
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 3, 1, 2)  # NHWC to NCHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 1)  # NCHW to NHWC
    return out


def base_conv3d(x, conv_layer, channel_last=False, residual=None, only_return_output=False):
    if only_return_output:
        size = cal_outsize(x.shape, conv_layer.weight.shape, conv_layer.stride, conv_layer.padding)
        return torch.empty(size, device=x.device, dtype=x.dtype)
    if channel_last:
        x = x.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
    out = F.conv3d(x, conv_layer.weight, conv_layer.bias, stride=conv_layer.stride, padding=conv_layer.padding)
    if residual is not None:
        if channel_last:
            residual = residual.permute(0, 4, 1, 2, 3)  # NDHWC to NCDHW
        out += residual
    if channel_last:
        out = out.permute(0, 2, 3, 4, 1)  # NCDHW to NDHWC
    return out


def cal_outsize(input_sizes, kernel_sizes, stride, padding):
    stride_d, stride_h, stride_w = stride
    padding_d, padding_h, padding_w = padding
    dilation_d, dilation_h, dilation_w = 1, 1, 1

    in_d = input_sizes[1]
    in_h = input_sizes[2]
    in_w = input_sizes[3]

    kernel_d = kernel_sizes[2]
    kernel_h = kernel_sizes[3]
    kernel_w = kernel_sizes[4]
    out_channels = kernel_sizes[0]

    out_d = calc_out_(in_d, padding_d, dilation_d, kernel_d, stride_d)
    out_h = calc_out_(in_h, padding_h, dilation_h, kernel_h, stride_h)
    out_w = calc_out_(in_w, padding_w, dilation_w, kernel_w, stride_w)
    size = [input_sizes[0], out_d, out_h, out_w, out_channels]
    return size


def calc_out_(in_size, padding, dilation, kernel, stride):
    return (in_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def base_conv3d_channel_last(x, conv_layer, residual=None):
    in_numel = x.numel()
    out_numel = int(x.numel() * conv_layer.out_channels / conv_layer.in_channels)
    if (in_numel >= 2**30) or (out_numel >= 2**30):
        if conv_layer.stride[0] != 1:
            raise Exception("time split asks time stride = 1")

        B, T, H, W, C = x.shape
        K = conv_layer.kernel_size[0]
        chunks = 4
        chunk_size = T // chunks

        if residual is None:
            out_nhwc = base_conv3d(x, conv_layer, channel_last=True, residual=residual, only_return_output=True)
        else:
            out_nhwc = residual

        for i in range(chunks):
            start = chunk_size * i
            if i == chunks - 1:
                xi = x[:1, start:]
                out_nhwci = out_nhwc[:1, start:]
            else:
                end = start + chunk_size + K - 1
                xi = x[:1, start:end]
                pos = start + chunk_size
                out_nhwci = out_nhwc[:1, start:pos]
            if residual is not None:
                if i == chunks - 1:
                    ri = residual[:1, start:]
                else:
                    pos = start + chunk_size
                    ri = residual[:1, start:pos]
            else:
                ri = None
            conv_result = base_conv3d(xi, conv_layer, channel_last=True, residual=ri)
            conv_result = conv_result.clone()
            out_nhwci.copy_(conv_result)

            if conv_result.shape != out_nhwci.shape or conv_result.dtype != out_nhwci.dtype:
                raise Exception(f"conv_result shape [{conv_result.shape}] must be the same as "
                                f"out_nhwci shape [{out_nhwci.shape}], and conv_result dtype [{conv_result.dtype}] "
                                f"must be the same as out_nhwci dtype [{out_nhwci.dtype}]")
    else:
        out_nhwc = base_conv3d(x, conv_layer, channel_last=True, residual=residual)
    return out_nhwc


def base_group_norm_with_zero_pad(x, norm_layer, act_silu=True, pad_size=2):
    out_shape = list(x.shape)
    out_shape[1] += pad_size
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    out[:, pad_size:] = base_group_norm(x, norm_layer, act_silu=act_silu)
    out[:, :pad_size] = 0
    return out


class Downsample2D(nn.Module):
    def __init__(self, channels, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.padding = padding
        stride = 2
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)

    def forward(self, x):
        if self.padding == 0:
            pad = (0, 0, 0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)

        x = base_conv2d(x, self.conv, channel_last=True)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.conv = CausalConv3dBase(in_channels, in_channels, kernel_size=3, stride=stride)

    def forward(self, x, is_init=True):
        x = self.conv(x, is_init)
        return x


class ConvPixelShuffleUpSampleLayer3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        self.conv = CausalConv3dBase(in_channels, out_channels * out_ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        batch_size, channels, depth, height, width = x.size()
        new_channels = channels // (self.factor ** 3)

        x = x.view(batch_size, new_channels, self.factor, self.factor, self.factor, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, new_channels, depth * self.factor, height * self.factor, width * self.factor)
        x = x[:, :, self.factor - 1:, :, :]
        return x


class ConvPixelUnshuffleDownSampleLayer3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**3
        if out_channels % out_ratio != 0:
            raise Exception(f"out_channels {out_channels} should be a multiple of out_ratio {out_ratio}.")
        self.conv = CausalConv3dBase(in_channels, out_channels // out_ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, is_init=True) -> torch.Tensor:
        x = self.conv(x, is_init)
        x = F.pad(x, (0, 0, 0, 0, self.factor - 1, 0))  # (left, right, top, bottom, front, back)
        batch_size, channels, depth, height, width = x.shape
        x = x.view(batch_size, channels, depth // self.factor, self.factor, height // self.factor,
                   self.factor, width // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(batch_size, channels * self.factor ** 3, depth // self.factor,
                   height // self.factor, width // self.factor)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        if out_channels * factor**3 % in_channels != 0:
            raise Exception(
                f"out_channels {out_channels} * factor {factor}**3 should be a multiple of in_channels {in_channels}.")
        self.repeats = out_channels * factor**3 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor, self.factor, self.factor, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor, x.size(4) * self.factor, x.size(6) * self.factor)
        x = x[:, :, self.factor - 1:, :, :]
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        if in_channels * factor ** 3 % out_channels != 0:
            raise Exception(
                f"in_channels {in_channels} * factor {factor}**3 should be a multiple of out_channels {out_channels}.")
        self.group_size = in_channels * factor**3 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, self.factor - 1, 0))  # (left, right, top, bottom, front, back)
        batch_size, channels, depth, height, width = x.shape

        x = x.view(batch_size, channels, depth // self.factor, self.factor, height // self.factor,
                   self.factor, width // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(batch_size, channels * self.factor**3, depth // self.factor,
                   height // self.factor, width // self.factor)
        x = x.view(batch_size, self.out_channels, self.group_size, depth // self.factor,
                   height // self.factor, width // self.factor)
        x = x.mean(dim=2)
        return x


class CausalConvChannelLast(CausalConv3dBase):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__(chan_in, chan_out, kernel_size, **kwargs)
        self.time_causal_padding = (0, 0) + self.time_causal_padding
        self.time_uncausal_padding = (0, 0) + self.time_uncausal_padding

    def forward(self, x, is_init=True, residual=None):
        x = F.pad(x, self.time_causal_padding if is_init else self.time_uncausal_padding)
        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class CausalConvAfterNorm(CausalConv3dBase):
    def forward(self, x, is_init=True, residual=None):
        if self.time_causal_padding != (1, 1, 1, 1, 2, 0):
            x = F.pad(x, self.time_causal_padding).contiguous()

        x = base_conv3d_channel_last(x, self.conv, residual=residual)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.q = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.k = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.v = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.proj_out = CausalConvChannelLast(in_channels, in_channels, kernel_size=1)
        self.attn_mask_npu = torch.triu(torch.ones([2048, 2048], device="npu"), diagonal=1).bool()

    def attention(self, x, is_init=True):
        x = base_group_norm(x, self.norm, act_silu=False)
        q = self.q(x, is_init)
        k = self.k(x, is_init)
        v = self.v(x, is_init)

        b, t, h, w, c = q.shape
        q, k, v = map(lambda x: rearrange(x, "b t h w c -> b 1 (t h w) c"), (q, k, v))
        x = torch_npu.npu_fusion_attention(q, k, v, q.size(1), input_layout="BNSD",
                                           atten_mask=self.attn_mask_npu, sparse_mode=2, scale=q.size(-1) ** (-0.5))[0]
        x = rearrange(x, "b 1 (t h w) c -> b t h w c", t=t, h=h, w=w)

        return x

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        h = self.attention(x)
        x = self.proj_out(h, residual=x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class Resnet3DBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels=None,
        temb_channels=512,
        conv_shortcut=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv1 = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = CausalConvAfterNorm(out_channels, out_channels, kernel_size=3, padding=1)

        self.use_conv_shortcut = conv_shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConvAfterNorm(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.nin_shortcut = CausalConvAfterNorm(in_channels, out_channels, kernel_size=1)

    def forward(self, x, temb=None, is_init=True):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        h = base_group_norm_with_zero_pad(x, self.norm1, act_silu=True, pad_size=2)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nn.functional.silu(temb))[:, :, None, None]

        x = self.nin_shortcut(x) if self.in_channels != self.out_channels else x
        h = base_group_norm_with_zero_pad(h, self.norm2, act_silu=True, pad_size=2)
        x = self.conv2(h, residual=x)

        x = x.permute(0, 4, 1, 2, 3)
        return x


class VideoEncoder(nn.Module):
    def __init__(self,
        ch: int = 32,
        ch_mult: Tuple[int] = None,
        num_res_blocks: int = 2,
        in_channels: int = 3,
        z_channels: int = 16,
        double_z: bool = True,
        down_sampling_layer: List[int] = None,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = (4, 8, 16, 16)
        if down_sampling_layer is None:
            down_sampling_layer = [1, 2]
        temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv3dBase(in_channels, ch, kernel_size=3)
        self.down_sampling_layer = down_sampling_layer

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    Resnet3DBlock(in_channels=block_in, out_channels=block_out, temb_channels=temb_ch))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level in self.down_sampling_layer:
                    down.downsample = Downsample3D(block_in, stride=(2, 2, 2))
                else:
                    down.downsample = Downsample2D(block_in, padding=0)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(in_channels=block_in, out_channels=block_in, temb_channels=temb_ch)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(in_channels=block_in, out_channels=block_in, temb_channels=temb_ch)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
        channels = 4 * z_channels * 2 ** 3
        self.conv_patchify = ConvPixelUnshuffleDownSampleLayer3D(block_in, channels, kernel_size=3, factor=2)
        self.shortcut_pathify = PixelUnshuffleChannelAveragingDownSampleLayer3D(block_in, channels, 2)
        self.shortcut_out = PixelUnshuffleChannelAveragingDownSampleLayer3D(channels, 2 * z_channels if double_z else z_channels, 1)
        self.conv_out = CausalConvChannelLast(channels, 2 * z_channels if double_z else z_channels, kernel_size=3)

    def forward(self, x, video_frame_num, is_init=True):
        # timestep embedding
        temb = None
        t = video_frame_num
        # downsampling
        h = self.conv_in(x, is_init)
        # make it real channel last, but behave like normal layout
        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb, is_init)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                if isinstance(self.down[i_level].downsample, Downsample2D):
                    _, _, t, _, _ = h.shape
                    h = rearrange(h, "b c t h w -> (b t) h w c", t=t)
                    h = self.down[i_level].downsample(h)
                    h = rearrange(h, "(b t) h w c -> b c t h w", t=t)
                else:
                    h = self.down[i_level].downsample(h, is_init)

        h = self.mid.block_1(h, temb, is_init)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, is_init)

        h = h.permute(0, 2, 3, 4, 1).contiguous()  # b c l h w -> b l h w c

        h = base_group_norm(h, self.norm_out, act_silu=True)
        h = h.permute(0, 4, 1, 2, 3).contiguous()
        shortcut = self.shortcut_pathify(h)
        h = self.conv_patchify(h, is_init)
        h = h.add_(shortcut)
        shortcut = self.shortcut_out(h).permute(0, 2, 3, 4, 1)
        h = self.conv_out(h.permute(0, 2, 3, 4, 1).contiguous(), is_init)
        h = h.add_(shortcut)

        h = h.permute(0, 4, 1, 2, 3)  # b l h w c -> b c l h w

        h = rearrange(h, "b c t h w -> b t c h w")
        return h


class Res3DBlockUpsample(nn.Module):
    def __init__(self,
        input_filters,
        num_filters,
        down_sampling_stride,
        down_sampling=False
    ):
        super().__init__()

        self.input_filters = input_filters
        self.num_filters = num_filters
        self.act_ = nn.SiLU(inplace=True)

        self.conv1 = CausalConvChannelLast(num_filters, num_filters, kernel_size=[3, 3, 3])
        self.norm1 = nn.GroupNorm(32, num_filters)
        self.conv2 = CausalConvChannelLast(num_filters, num_filters, kernel_size=[3, 3, 3])
        self.norm2 = nn.GroupNorm(32, num_filters)

        self.down_sampling = down_sampling
        if down_sampling:
            self.down_sampling_stride = down_sampling_stride
        else:
            self.down_sampling_stride = [1, 1, 1]

        if num_filters != input_filters or down_sampling:
            self.conv3 = CausalConvChannelLast(input_filters, num_filters, kernel_size=[1, 1, 1], stride=self.down_sampling_stride)
            self.norm3 = nn.GroupNorm(32, num_filters)

    def forward(self, x, is_init=False):
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        residual = x

        h = self.conv1(x, is_init)
        h = base_group_norm(h, self.norm1, act_silu=True)
        h = self.conv2(h, is_init)
        h = base_group_norm(h, self.norm2, act_silu=False)

        if self.down_sampling or self.num_filters != self.input_filters:
            x = self.conv3(x, is_init)
            x = base_group_norm(x, self.norm3, act_silu=False)

        h.add_(x)
        h = self.act_(h)
        if residual is not None:
            h.add_(residual)

        h = h.permute(0, 4, 1, 2, 3)
        return h


class Upsample3D(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
        self.conv3d = Res3DBlockUpsample(input_filters=in_channels,
                                         num_filters=in_channels,
                                         down_sampling_stride=(1, 1, 1),
                                         down_sampling=False)

    def forward(self, x, is_init=True, is_split=True):
        b, c, t, h, w = x.shape
        if is_split:
            split_size = c // 8
            x_slices = torch.split(x, split_size, dim=1)
            x = [nn.functional.interpolate(x, scale_factor=self.scale_factor) for x in x_slices]
            x = torch.cat(x, dim=1)
        else:
            x = nn.functional.interpolate(x, scale_factor=self.scale_factor)

        x = self.conv3d(x, is_init)
        return x


class VideoDecoder(nn.Module):
    def __init__(self,
        ch: int = 128,
        z_channels: int = 16,
        out_channels: int = 3,
        ch_mult: Tuple[int] = None,
        num_res_blocks: int = 2,
        temporal_up_layers: List[int] = None,
        temporal_downsample: int = 4,
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = (1, 2, 4, 4)
        if temporal_up_layers is None:
            temporal_up_layers = [2, 3]
        temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample

        block_in = ch * ch_mult[self.num_resolutions - 1]
        channels = 4 * z_channels * 2 ** 3
        self.conv_in = CausalConv3dBase(z_channels, channels, kernel_size=3)
        self.shortcut_in = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(z_channels, channels, 1)
        self.conv_unpatchify = ConvPixelShuffleUpSampleLayer3D(channels, block_in, kernel_size=3, factor=2)
        self.shortcut_unpathify = ChannelDuplicatingPixelUnshuffleUpSampleLayer3D(channels, block_in, 2)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Resnet3DBlock(in_channels=block_in, out_channels=block_in, temb_channels=temb_ch)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = Resnet3DBlock(in_channels=block_in, out_channels=block_in, temb_channels=temb_ch)

        # upsampling
        self.up_id = len(temporal_up_layers)
        self.video_frame_num = 1
        self.cur_video_frame_num = self.video_frame_num // 2 ** self.up_id + 1
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    Resnet3DBlock(in_channels=block_in, out_channels=block_out, temb_channels=temb_ch))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level in temporal_up_layers:
                    up.upsample = Upsample3D(block_in)
                    self.cur_video_frame_num = self.cur_video_frame_num * 2
                else:
                    up.upsample = Upsample(block_in, block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
        self.conv_out = CausalConvAfterNorm(block_in, out_channels, kernel_size=3, padding=1)

    def forward(self, z, is_init=True):
        z = rearrange(z, "b t c h w -> b c t h w")
        h = self.conv_in(z, is_init=is_init)
        shortcut = self.shortcut_in(z)
        h = h.add_(shortcut)
        shortcut = self.shortcut_unpathify(h)
        h = self.conv_unpatchify(h, is_init=is_init)
        h = h.add_(shortcut)

        temb = None
        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
        h = self.mid.block_1(h, temb, is_init=is_init)
        h = self.mid.attn_1(h)
        h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
        h = self.mid.block_2(h, temb, is_init=is_init)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = h.permute(0, 2, 3, 4, 1).contiguous().permute(0, 4, 1, 2, 3)
                h = self.up[i_level].block[i_block](h, temb, is_init=is_init)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                if isinstance(self.up[i_level].upsample, Upsample):
                    h = self.up[i_level].upsample(h)  # b c t h w
                else:
                    h = self.up[i_level].upsample(h, is_init=is_init)

        # end
        h = h.permute(0, 2, 3, 4, 1)  # b c l h w -> b l h w c
        h = base_group_norm_with_zero_pad(h, self.norm_out, act_silu=True, pad_size=2)
        h = self.conv_out(h)
        h = h.permute(0, 4, 1, 2, 3)

        if is_init:
            h = h[:, :, (self.temporal_downsample - 1):]
        return h


class StepVideoVae(nn.Module):
    def __init__(self,
        from_pretrained: str = None,
        in_channels: int = 3,
        out_channels: int = 3,
        z_channels: int = 16,
        num_res_blocks: int = 2,
        frame_len: int = 17,
        latent_len: int = 3,
        **kwargs
    ):
        super().__init__()

        self.frame_len = frame_len
        self.latent_len = latent_len

        self.encoder = VideoEncoder(
            in_channels=in_channels,
            z_channels=z_channels,
            num_res_blocks=num_res_blocks,
        )

        self.decoder = VideoDecoder(
            z_channels=z_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
        )
        self.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if from_pretrained is not None:
            weight_dict = self.init_from_ckpt(from_pretrained)
            if len(weight_dict) != 0:
                self.load_state_dict(weight_dict)

    def init_from_ckpt(self, model_path):
        from safetensors import safe_open
        p = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)
                if k.startswith("decoder.conv_out."):
                    k = k.replace("decoder.conv_out.", "decoder.conv_out.conv.")
                p[k] = tensor
        return p

    def naive_encode(self, x):
        length = x.size(1)
        x = rearrange(x, 'b l c h w -> b c l h w').contiguous()
        z = self.encoder(x, length, True)
        return z

    def encode(self, x):
        # b (nc cf) c h w -> (b nc) cf c h w -> encode -> (b nc) cf c h w -> b (nc cf) c h w
        chunks = list(x.split(self.frame_len, dim=1))
        for i, chunk in enumerate(chunks):
            chunks[i] = self.naive_encode(chunk)
        z = torch.cat(chunks, dim=1)

        posterior = DiagonalGaussianDistribution(z, dim=-3)
        return posterior.sample()

    def decode_naive(self, z, is_init=True):
        z = z.to(next(self.decoder.parameters()).dtype)
        dec = self.decoder(z, is_init)
        return dec

    def decode(self, z):
        # b (nc cf) c h w -> (b nc) cf c h w -> decode -> (b nc) c cf h w -> b (nc cf) c h w
        chunks = list(z.split(self.latent_len, dim=1))
        chunks_total_num = len(chunks)
        max_num_per_rank = (chunks_total_num + self.world_size - 1) // self.world_size

        if self.world_size > 1:
            rank = torch.distributed.get_rank()
            chunks_ = chunks[max_num_per_rank * rank: max_num_per_rank * (rank + 1)]
            if len(chunks_) < max_num_per_rank:
                chunks_.extend(chunks[:max_num_per_rank - len(chunks_)])
            chunks = chunks_

        for i, chunk in enumerate(chunks):
            chunks[i] = self.decode_naive(chunk, True).permute(0, 2, 1, 3, 4)
        x = torch.cat(chunks, dim=1)

        if self.world_size > 1:
            x_ = torch.empty([x.size(0), (self.world_size * max_num_per_rank) * self.frame_len, *x.shape[2:]],
                             dtype=x.dtype, device=x.device)
            torch.distributed.all_gather_into_tensor(x_, x)
            x = x_[:, : chunks_total_num * self.frame_len]

        x = self.mix(x)
        return x.transpose(1, 2)

    def mix(self, x):
        remain_scale = 0.6
        mix_scale = 1. - remain_scale
        front = slice(self.frame_len - 1, x.size(1) - 1, self.frame_len)
        back = slice(self.frame_len, x.size(1), self.frame_len)
        x[:, back] = x[:, back] * remain_scale + x[:, front] * mix_scale
        x[:, front] = x[:, front] * remain_scale + x[:, back] * mix_scale
        return x
