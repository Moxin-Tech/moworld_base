from typing import Union, Tuple, Optional
from collections import deque
from einops import rearrange

import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm

from mindspeed_mm.models.common.communications import collect_tensors_across_ranks, split_tensor
from mindspeed_mm.utils.utils import cast_tuple, video_to_image, get_context_parallel_group, get_context_parallel_rank
from mindspeed_mm.models.common.conv import CausalConv3d, WfCausalConv3d, TimePaddingCausalConv3d


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        **kwargs
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    @video_to_image
    def forward(self, x, scale_factor=2.0, mode="nearest"):
        x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        undown=False,
        **kwargs
    ):
        super().__init__()
        self.undown = undown
        # no asymmetric padding in torch conv, must do it ourselves
        if self.undown:
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=1
            )
        else:
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=0
            )

    @video_to_image
    def forward(self, x):
        if self.undown:
            x = self.conv(x)
        else:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        return x


class DownSample3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, with_conv=True, compress_time=False, **kwargs):
        super().__init__()
        self.with_conv = with_conv
        if out_channels is None:
            out_channels = in_channels
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.compress_time = compress_time

    def forward(self, x, enable_cp=True):
        if self.compress_time and x.shape[2] > 1:
            h, w = x.shape[-2:]
            x = rearrange(x, "b c t h w -> (b h w) c t")

            if self.enable_cp_or_tiling(enable_cp, x):
                # split first frame
                x_first, x_rest = x[..., 0], x[..., 1:]

                if x_rest.shape[-1] > 0:
                    splits = torch.split(x_rest, 32, dim=1)
                    interpolated_splits = [
                        torch.nn.functional.avg_pool1d(split, kernel_size=2, stride=2)
                        for split in splits
                    ]
                    x_rest = torch.cat(interpolated_splits, dim=1)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
            else:
                splits = torch.split(x, 32, dim=1)
                interpolated_splits = [
                    torch.nn.functional.avg_pool1d(split, kernel_size=2, stride=2)
                    for split in splits
                ]
                x = torch.cat(interpolated_splits, dim=1)
                x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)

        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        else:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x

    def enable_cp_or_tiling(self, enable_cp, x):
        return get_context_parallel_rank() == 0 and enable_cp or (not enable_cp and x.shape[-1] % 2 == 1)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False, **kwargs):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.compress_time = compress_time

    def forward(self, x, enable_cp=True):
        if self.compress_time and x.shape[2] > 1:
            if get_context_parallel_rank() == 0 and enable_cp:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")

                splits = torch.split(x_rest, 32, dim=1)
                interpolated_splits = [
                    torch.nn.functional.interpolate(split, scale_factor=2.0, mode="nearest")
                    for split in splits
                ]
                x_rest = torch.cat(interpolated_splits, dim=1)

                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                splits = torch.split(x, 32, dim=1)
                interpolated_splits = [
                    torch.nn.functional.interpolate(split, scale_factor=2.0, mode="nearest")
                    for split in splits
                ]
                x = torch.cat(interpolated_splits, dim=1)
        else:
            # only interpolate 2D
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")

            splits = torch.split(x, 32, dim=1)
            interpolated_splits = [
                torch.nn.functional.interpolate(split, scale_factor=2.0, mode="nearest")
                for split in splits
            ]
            x = torch.cat(interpolated_splits, dim=1)

            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.with_conv:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class SpatialDownsample2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (2, 2),
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 2)
        stride = cast_tuple(stride, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = CausalConv3d(
            self.in_channels,
            self.out_channels,
            (1,) + self.kernel_size,
            stride=(1,) + stride,
            padding=0
        )

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class SpatialUpsample2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int]] = (3, 3),
        stride: Union[int, Tuple[int]] = (1, 1),
        unup=False,
        **kwargs
    ):
        super().__init__()
        self.unup = unup
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            (1,) + kernel_size,
            stride=(1,) + stride,
            padding=1
        )

    def forward(self, x):
        if not self.unup:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> b (c t) h w")
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
            x = rearrange(x, "b (c t) h w -> b c t h w", t=t)
        x = self.conv(x)
        return x


class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 2,
        **kwargs
    ):
        super().__init__()
        # note: when kernel_size=(kernel_size, 1, 1), and stride=(stride, 1, 1), can be replaced by pool1d
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.kernel_size - 1, 1, 1))
        x = torch.concatenate((first_frame_pad, x), dim=2)
        n, c, d, h, w = x.shape
        x = torch_npu.npu_confusion_transpose(x, (0, 1, 3, 4, 2), (n, c * h * w, d), True)
        conv_res = self.conv(x)
        b, s, m = conv_res.shape
        conv_res = torch_npu.npu_confusion_transpose(conv_res, (0, 1, 4, 2, 3),
                                                     (n, c, h, w, (b * s * m) // (n * c * h * w)), False)
        return conv_res


class TimeUpsample2x(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        if x.size(2) > 1:
            x, y = x[:, :, :1], x[:, :, 1:]
            y = F.interpolate(y, scale_factor=(2, 1, 1), mode="trilinear")
            x = torch.concat([x, y], dim=2)
        return x


class TimeDownsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Tuple[int] = (2, 1, 1),
        padding: Tuple[int] = (0, 1, 1),
        mix_factor: float = 2.0,
        **kwargs
    ):
        super().__init__()
        # note: when kernel_size=(kernel_size, 1, 1), and stride=(stride, 1, 1), can be replaced by pool1d
        self.avg_pool = nn.AvgPool1d(kernel_size, stride[0])
        kernel_size = cast_tuple(kernel_size, 3)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.kernel_size[0] - 1, 1, 1))
        x = torch.concatenate((first_frame_pad, x), dim=2)
        n, c, d, h, w = x.shape
        x = torch_npu.npu_confusion_transpose(x, (0, 1, 3, 4, 2), (n, c * h * w, d), True)
        pool_res = self.avg_pool(x)
        b, s, m = pool_res.shape
        pool_res = torch_npu.npu_confusion_transpose(pool_res, (0, 1, 4, 2, 3),
                                                     (n, c, h, w, (b * s * m) // (n * c * h * w)), False)
        return alpha * pool_res + (1 - alpha) * self.conv(x)


class TimeUpsampleRes2x(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        padding: int = 1,
        mix_factor: float = 2.0,
        **kwargs
    ):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size, padding)
        self.mix_factor = torch.nn.Parameter(torch.Tensor([mix_factor]))

    def forward(self, x):
        alpha = torch.sigmoid(self.mix_factor)
        if x.size(2) > 1:
            x, y = x[:, :, :1], x[:, :, 1:]
            x, y = x[:, :, :1], x[:, :, 1:]
            y = F.interpolate(y.float(), scale_factor=(2, 1, 1), mode="trilinear")
            x = torch.concat([x, y], dim=2)
        return alpha * x + (1 - alpha) * self.conv(x)


class Spatial2xTime2x3DDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="CausalConv3d", enable_vae_cp=False, **kwargs):
        super().__init__()
        if conv_type == "WfCausalConv3d":
            ConvLayer = WfCausalConv3d
        elif conv_type == "CausalConv3d":
            ConvLayer = CausalConv3d
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=3, padding=0, stride=2, parallel_input=False)
        self.enable_vae_cp = enable_vae_cp

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 0)
        if self.enable_vae_cp:
            x = torch.concat(collect_tensors_across_ranks(x, get_context_parallel_group()), 2)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        if self.enable_vae_cp:
            x = split_tensor(x, get_context_parallel_group(), get_context_parallel_rank())
        return x


class Spatial2xTime2x3DUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        if x.size(2) > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(x_, scale_factor=(2, 2, 2), mode="trilinear")
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            x = torch.concat([x, x_], dim=2)
        else:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
        return self.conv(x)


class CachedCausal3DUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_interpolation="trilinear",
        enable_cached=False,
        **kwargs
    ):
        super().__init__()
        self.t_interpolation = t_interpolation
        self.conv = WfCausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.enable_cached = enable_cached
        self.causal_cached = deque()

    def forward(self, x):
        if x.size(2) > 1 or len(self.causal_cached) > 0:
            if self.enable_cached and len(self.causal_cached) > 0:
                x = torch.cat([self.causal_cached.popleft(), x], dim=2)
                self.causal_cached.append(x[:, :, -2:-1])
                x = F.interpolate(x, scale_factor=(2, 1, 1), mode=self.t_interpolation)
                x = x[:, :, 2:]
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
            else:
                if self.enable_cached:
                    self.causal_cached.append(x[:, :, -1:])
                x, x_ = x[:, :, :1], x[:, :, 1:]
                x_ = F.interpolate(
                    x_, scale_factor=(2, 1, 1), mode=self.t_interpolation
                )
                x_ = F.interpolate(x_, scale_factor=(1, 2, 2), mode="trilinear")
                x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")
                x = torch.concat([x, x_], dim=2)
        else:
            if self.enable_cached:
                self.causal_cached.append(x[:, :, -1:])
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear")

        return self.conv(x)


class DownsampleCausal3D(nn.Module):
    """
    A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        stride=2,
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = stride
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = TimePaddingCausalConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias
            )
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpsampleCausal3D(nn.Module):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        upsample_factor=(2, 2, 2),
        **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.upsample_factor = upsample_factor

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = TimePaddingCausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:

        if self.norm is not None:
            raise NotImplementedError

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            B, C, T, H, W = hidden_states.shape
            first_h, other_h = hidden_states.split((1, T - 1), dim=2)
            if output_size is None:
                if T > 1:
                    other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")

                first_h = first_h.squeeze(2)
                first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
                first_h = first_h.unsqueeze(2)
            else:
                raise NotImplementedError

            if T > 1:
                hidden_states = torch.cat((first_h, other_h), dim=2)
            else:
                hidden_states = first_h

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states
