from typing import Union, Tuple, Optional
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

from mindspeed_mm.utils.utils import (
    cast_tuple,
    video_to_image,
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_context_parallel_group_rank,
)


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    @video_to_image
    def forward(self, x):
        return super().forward(x)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


class SafeConv3d(torch.nn.Conv3d):
    def forward(self, input_x):
        memory_count = torch.prod(torch.tensor(input_x.shape)).item() * 2 / 1024 ** 3
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input_x, part_num, dim=2)  # NCTHW
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1:], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super(SafeConv3d, self).forward(input_x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        init_method: str = "random",
        **kwargs
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = kwargs.pop("stride", 1)
        padding = kwargs.pop("padding", 0)
        padding = list(cast_tuple(padding, 3))
        padding[0] = 0
        stride = cast_tuple(stride, 3)
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=stride, padding=padding)
        self.pad = nn.ReplicationPad2d((0, 0, self.time_kernel_size - 1, 0))
        if init_method:
            self._init_weights(init_method)

    def _init_weights(self, init_method):
        if init_method == "avg":
            if not (self.kernel_size[1] == 1 and self.kernel_size[2] == 1):
                raise AssertionError("only support temporal up/down sample")
            if self.in_channels != self.out_channels:
                raise AssertionError("in_channels must be equal to out_channels")
            weight = torch.zeros((self.out_channels, self.in_channels, *self.kernel_size))

            eyes = torch.concat(
                [
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                    torch.eye(self.in_channels).unsqueeze(-1) * 1 / 3,
                ],
                dim=-1,
            )
            weight[:, :, :, 0, 0] = eyes

            self.conv.weight = nn.Parameter(
                weight,
                requires_grad=True,
            )
        elif init_method == "zero":
            self.conv.weight = nn.Parameter(
                torch.zeros((self.out_channels, self.in_channels, *self.kernel_size)),
                requires_grad=True,
            )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_kernel_size - 1, 1, 1))  # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return self.conv(x)


class WfCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        enable_cached=False,
        bias=True,
        is_first_chunk=True,
        parallel_input=True,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size = self.kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = kwargs.pop("stride", 1)
        self.padding = kwargs.pop("padding", 0)
        self.padding = list(cast_tuple(self.padding, 3))
        self.padding[0] = 0
        self.stride = cast_tuple(self.stride, 3)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=bias
        )
        self.enable_cached = enable_cached
        self.is_first_chunk = is_first_chunk
        self.parallel_input = parallel_input
        self.causal_cached = deque()
        self.cache_offset = 0

    def forward(self, x):
        if self.is_first_chunk and (get_context_parallel_rank() == 0 or not self.parallel_input):
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x), dim=2)
        elif self.enable_cached:
            first_frame_pad = self.causal_cached.popleft()
            x = torch.concatenate((first_frame_pad, x), dim=2)

        if self.enable_cached and self.time_kernel_size != 1:
            if (self.time_kernel_size - 1) // self.stride[0] != 0:
                if self.cache_offset == 0 or self.is_first_chunk:
                    self.causal_cached.append(x[:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
                else:
                    self.causal_cached.append(
                        x[:, :, :-self.cache_offset][:, :, -(self.time_kernel_size - 1) // self.stride[0]:].clone())
            else:
                self.causal_cached.append(x[:, :, 0:0, :, :].clone())
        elif self.enable_cached:
            self.causal_cached.append(x[:, :, 0:0, :, :].clone())

        return self.conv(x)


def _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding=None):
    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    input_ = input_.transpose(0, dim)

    # pass from last rank
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    recv_buffer = torch.empty_like(input_[-kernel_size + 1:]).contiguous()
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1:].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    if cp_rank == 0:
        if cache_padding is not None:
            input_ = torch.cat([cache_padding.transpose(0, dim).to(input_.device), input_], dim=0)
        else:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    input_ = input_.transpose(0, dim).contiguous()
    return input_


class _FakeCPConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, cache_padding):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None, None


def fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding):
    return _FakeCPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, cache_padding)


def _drop_from_previous_rank(input_, dim, kernel_size):
    input_ = input_.transpose(0, dim)[kernel_size - 1:].transpose(0, dim)
    return input_


class ContextParallelCausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=1, **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        if not (is_odd(height_kernel_size) and is_odd(width_kernel_size)):
            raise AssertionError("Height and width must be odd.")

        kwargs.pop("padding", 0)
        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2

        stride = (stride, stride, stride)
        dilation = (1, 1, 1)
        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None

    def forward(self, input_, clear_cache=True, enable_cp=True, conv_cache: Optional[torch.Tensor] = None,
                use_conv_cache=False):
        if enable_cp:
            input_parallel = fake_cp_pass_from_previous_rank(
                input_, self.temporal_dim, self.time_kernel_size, self.cache_padding
            )
        else:
            kernel_size = self.time_kernel_size
            if kernel_size > 1 and use_conv_cache:
                cached_inputs = [conv_cache] if conv_cache is not None else [input_[:, :, :1]] * (kernel_size - 1)
                input_parallel = torch.cat(cached_inputs + [input_], dim=2)
            else:
                input_parallel = torch.cat([input_[:, :, 0:1]] * (self.time_kernel_size - 1) + [input_], dim=2)

        del self.cache_padding
        self.cache_padding = None
        if not clear_cache:
            cp_rank, cp_world_size = get_context_parallel_rank(), get_context_parallel_world_size()
            global_rank = torch.distributed.get_rank()
            if cp_world_size == 1:
                self.cache_padding = (
                    input_parallel[:, :, -self.time_kernel_size + 1:].contiguous().detach().clone().cpu()
                )
            else:
                if cp_rank == cp_world_size - 1:
                    torch.distributed.isend(
                        input_parallel[:, :, -self.time_kernel_size + 1:].contiguous(),
                        global_rank + 1 - cp_world_size,
                        group=get_context_parallel_group(),
                    )
                if cp_rank == 0:
                    recv_buffer = torch.empty_like(input_parallel[:, :, -self.time_kernel_size + 1:]).contiguous()
                    torch.distributed.recv(
                        recv_buffer, global_rank - 1 + cp_world_size, group=get_context_parallel_group()
                    )
                    self.cache_padding = recv_buffer.contiguous().detach().clone().cpu()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        if use_conv_cache:
            conv_cache = input_parallel[:, :, -self.time_kernel_size + 1:].clone()
        input_parallel = F.pad(input_parallel, padding_2d, mode="constant", value=0)

        output_parallel = self.conv(input_parallel)
        output = output_parallel
        return output, conv_cache


class TimePaddingCausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode='replicate',
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        dtype_org = x.dtype
        x = F.pad(x.float(), self.time_causal_padding, mode=self.pad_mode).to(dtype_org)
        return self.conv(x)


class CausalConv3dBase(nn.Module):
    def __init__(self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        bias=True,
        **kwargs
    ):
        super().__init__()

        self.kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = self.kernel_size

        self.dilation = dilation
        self.stride = cast_tuple(stride, 3)
        self.padding = list(cast_tuple(padding, 3))
        self.padding[0] = 0
        self.bias = bias

        time_pad = self.dilation * (self.time_kernel_size - 1) + max((1 - self.stride[0]), 0)
        self.time_pad = time_pad
        self.height_pad = self.height_kernel_size // 2
        self.width_pad = self.width_kernel_size // 2

        self.time_causal_padding = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        self.time_uncausal_padding = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, 0, 0)

        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=self.stride, dilation=self.dilation,
                              padding=self.padding, bias=self.bias, **kwargs)

    def forward(self, x, is_init=True, residual=None):
        x = F.pad(x, self.time_causal_padding if is_init else self.time_uncausal_padding)
        x = self.conv(x)
        if residual is not None:
            x.add_(residual)
        return x
