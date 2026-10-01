# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_npu
import torch.distributed as dist
from torch import Tensor
from megatron.training import get_args
from megatron.core import mpu
from einops import rearrange
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (
        all_to_all,
        split_forward_gather_backward,
        gather_forward_split_backward
)
from mindspeed_mm.models.common.attention import FlashAttention


def split_forward_gather_backward_FPDT_tensors(tensor, seq_dim=0, chunk_number=1, group=None, grad_scale="down"):
    world_size = torch.distributed.get_world_size(group)
    global_chunk_number = chunk_number * world_size

    chunks = list(torch.chunk(tensor, global_chunk_number, dim=seq_dim))
    indices = []
    for r in range(world_size):
        indices.extend(range(r, global_chunk_number, world_size))

    tensor = torch.cat([chunks[i] for i in indices], dim=seq_dim)
    tensor = split_forward_gather_backward(
        tensor, group, dim=seq_dim, grad_scale=grad_scale
    )
    return tensor


def gather_forward_split_backward_FPDT_tensors(tensor, seq_dim=0, chunk_number=1, group=None, grad_scale="up"):
    world_size = torch.distributed.get_world_size(group)
    tensor = gather_forward_split_backward(
        tensor, group, dim=seq_dim, grad_scale=grad_scale
    )
    global_chunk_number = chunk_number * world_size

    chunks = list(torch.chunk(tensor, global_chunk_number, dim=seq_dim))
    indices = [i // world_size + (i % world_size) * chunk_number for i in range(global_chunk_number)]

    tensor = torch.cat([chunks[i] for i in indices], dim=seq_dim)
    return tensor


def forward_update(prev_data, cur_data):

    prev_attn_out, prev_softmax_max, prev_softmax_sum = prev_data[0], prev_data[1], prev_data[2]
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_data[0], cur_data[1], cur_data[2]
    origin_dtype = prev_attn_out.dtype

    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)

    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    d = prev_attn_out.shape[-1]

    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    prev_out_scale = rearrange(prev_out_scale, 'b n s d -> b s n d').contiguous()

    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = rearrange(cur_out_scale, 'b n s d -> b s n d').contiguous()

    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    return attn_out, softmax_max, softmax_sum


def general_out_update(global_o, global_softmax_max, global_softmax_sum, cur_attn_outs):

    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]

    if global_o is None:
        global_o = cur_attn_out
        global_softmax_max = cur_softmax_max
        global_softmax_sum = cur_softmax_sum

    else:
        prev_data = [global_o, global_softmax_max, global_softmax_sum]
        cur_data = [cur_attn_out, cur_softmax_max, cur_softmax_sum]
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            prev_data, cur_data)

        global_o, global_softmax_max, global_softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

    return (global_o, global_softmax_max, global_softmax_sum)


class _FPDTGPUAttentionImpl_(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                attention_mask,
                cpg,
                scatter_idx,
                gather_idx,
                hidden_size,
                hidden_size_per_attention_head,
                dropout=0.0,
                num_chunks=8,
                cpu_offloading=False):

        do_save = q.requires_grad

        with torch.no_grad():

            per_gpu_seq_len = q.shape[0]
            chunk_size = per_gpu_seq_len // num_chunks
            if chunk_size * num_chunks != per_gpu_seq_len:
                raise ValueError(f'per_gpu_seq_len {per_gpu_seq_len} should be divided by num_chunks {num_chunks}')
            if attention_mask is not None:
                raise NotImplementedError('attention_mask is not supported now')
            ctx.num_chunks = num_chunks
            ctx.cpu_offloading = cpu_offloading
            ctx.cpg = cpg
            ctx.scatter_idx = scatter_idx
            ctx.gather_idx = gather_idx
            ctx.hidden_size_per_attention_head = hidden_size_per_attention_head

            device = 'npu:{}'.format(torch.npu.current_device())
            ctx.device = device
            ctx.dtype = q.dtype

            global_q = []
            global_k = []
            global_v = []

            ctx.softmax_scale = hidden_size_per_attention_head**(-0.5)

            ctx.keep_p = 1. - dropout
            n = hidden_size // hidden_size_per_attention_head // mpu.get_context_parallel_world_size()
            ctx.n = n
            batch_size = q.shape[1]

            global_o = [None for _ in range(num_chunks)]
            output = [None for _ in range(num_chunks)]
            global_softmax_max = [None for _ in range(num_chunks)]
            global_softmax_sum = [None for _ in range(num_chunks)]

            q_chunks = torch.chunk(q, chunks=num_chunks, dim=0)
            k_chunks = torch.chunk(k, chunks=num_chunks, dim=0)
            v_chunks = torch.chunk(v, chunks=num_chunks, dim=0)

            for i in range(num_chunks):

                q_chunk = q_chunks[i]
                k_chunk = k_chunks[i]
                v_chunk = v_chunks[i]

                q_chunk = q_chunk.reshape(q_chunk.shape[0], q_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
                k_chunk = k_chunk.reshape(k_chunk.shape[0], k_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
                v_chunk = v_chunk.reshape(v_chunk.shape[0], v_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()

                q_chunk = all_to_all(q_chunk, cpg, scatter_idx, gather_idx)
                k_chunk = all_to_all(k_chunk, cpg, scatter_idx, gather_idx)
                v_chunk = all_to_all(v_chunk, cpg, scatter_idx, gather_idx)

                global_q.append(q_chunk)
                global_k.append(k_chunk)
                global_v.append(v_chunk)


            for i in range(num_chunks):

                for k_i in range(num_chunks):

                    attn_outs = torch_npu.npu_fusion_attention(global_q[i],
                                                               global_k[k_i],
                                                               global_v[k_i],
                                                               n,
                                                               'BSND',
                                                               scale=ctx.softmax_scale,
                                                               keep_prob=ctx.keep_p)

                    global_o[i], global_softmax_max[i], global_softmax_sum[i] = general_out_update(global_o[i], global_softmax_max[i], \
                                                                                                   global_softmax_sum[i], attn_outs)

                global_o[i] = global_o[i].to(ctx.dtype).contiguous()

                output[i] = all_to_all(global_o[i], cpg, gather_idx, scatter_idx)

            output = torch.cat(output, dim=1)
            output = output.reshape(output.shape[0], output.shape[1], -1).permute(1, 0, 2).contiguous()
            head_dim = output.shape[-1]

            if do_save:
                ctx.global_q = global_q
                ctx.global_k = global_k
                ctx.global_v = global_v
                ctx.attn_output = global_o
                ctx.global_softmax_max = global_softmax_max
                ctx.global_softmax_sum = global_softmax_sum
                ctx.head_dim = head_dim
                ctx.batch_size = batch_size


        return output

    @staticmethod
    def backward(ctx, dout):
        num_chunks = ctx.num_chunks
        device = ctx.device
        dtype = ctx.dtype
        cpg = ctx.cpg
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        softmax_scale = ctx.softmax_scale
        keep_p = ctx.keep_p
        n = ctx.n
        hidden_size_per_attention_head = ctx.hidden_size_per_attention_head

        global_q = ctx.global_q
        global_k = ctx.global_k
        global_v = ctx.global_v
        attn_output = ctx.attn_output
        global_softmax_max = ctx.global_softmax_max
        global_softmax_sum = ctx.global_softmax_sum

        grad_global_dout = []
        chunk_size = dout.shape[1] // num_chunks

        dout_chunks = torch.chunk(dout, chunks=num_chunks, dim=0)

        for i in range(num_chunks):
            dout_chunk = dout_chunks[i]
            dout_chunk = dout_chunk.reshape(dout_chunk.shape[0], dout_chunk.shape[1], -1, \
                                            hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
            dout_chunk = all_to_all(dout_chunk, cpg, scatter_idx, gather_idx)

            grad_global_dout.append(dout_chunk)

        dq = [torch.zeros(global_q[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]
        dk = [torch.zeros(global_k[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]
        dv = [torch.zeros(global_v[0].shape, dtype=torch.float, device=device) for _ in range(num_chunks)]

        for i in range(num_chunks):
            k_chunk = global_k[i]
            v_chunk = global_v[i]

            for q_i in range(num_chunks):
                q_chunk = global_q[q_i]
                attn_output_chunk = attn_output[q_i]

                global_softmax_max_chunk = global_softmax_max[q_i]
                global_softmax_sum_chunk = global_softmax_sum[q_i]

                d_out = grad_global_dout[q_i]

                attn_grad_outs = torch_npu.npu_fusion_attention_grad(q_chunk,
                                                                     k_chunk,
                                                                     v_chunk,
                                                                     d_out,
                                                                     n,
                                                                     'BSND',
                                                                     softmax_max=global_softmax_max_chunk,
                                                                     softmax_sum=global_softmax_sum_chunk,
                                                                     attention_in=attn_output_chunk,
                                                                     scale_value=softmax_scale,
                                                                     keep_prob=keep_p)

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

                dq[q_i].add_(cur_dq.to(torch.float))
                dk[i].add_(cur_dk.to(torch.float))
                dv[i].add_(cur_dv.to(torch.float))

        for i in range(num_chunks):

            dq[i] = dq[i].to(dtype).contiguous()
            dk[i] = dk[i].to(dtype).contiguous()
            dv[i] = dv[i].to(dtype).contiguous()

            dq[i] = all_to_all(dq[i], cpg, gather_idx, scatter_idx)
            dk[i] = all_to_all(dk[i], cpg, gather_idx, scatter_idx)
            dv[i] = all_to_all(dv[i], cpg, gather_idx, scatter_idx)

        dq = torch.cat(dq, dim=1)
        dk = torch.cat(dk, dim=1)
        dv = torch.cat(dv, dim=1)

        dq = dq.reshape(dq.shape[0], dq.shape[1], -1).permute(1, 0, 2).contiguous()
        dk = dk.reshape(dk.shape[0], dk.shape[1], -1).permute(1, 0, 2).contiguous()
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1).permute(1, 0, 2).contiguous()

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


class ChunkManager:
    def __init__(self, chunk: torch.Tensor, device=None, onload=False) -> None:

        self.chunk_shape = chunk.shape
        self.chunk_dtype = chunk.dtype
        self.device = chunk.device if device is None else device

        cpu_chunk = torch.empty(chunk.shape, dtype=chunk.dtype, device='cpu', pin_memory=True)

        if chunk.is_npu:
            cpu_chunk.copy_(chunk, non_blocking=True)
        else:
            cpu_chunk = chunk

        self.cpu_chunk = cpu_chunk
        self.npu_chunk = chunk if onload else None

    def load_to_npu(self):
        if self.npu_chunk is not None:
            pass
        else:
            npu_chunk = torch.empty(self.chunk_shape, device=self.device, dtype=self.chunk_dtype)
            npu_chunk.copy_(self.cpu_chunk, non_blocking=True)
            self.npu_chunk = npu_chunk

    def get_npu_chunk(self):
        if self.npu_chunk is None or str(self.npu_chunk.device) != str(self.device):
            raise AttributeError()
        return self.npu_chunk

    def offload(self):
        if self.npu_chunk is None or str(self.npu_chunk.device) != str(self.device):
            raise AttributeError()
        del self.npu_chunk
        self.npu_chunk = None

    def overwrite_to_cpu(self):
        if self.npu_chunk is None or str(self.npu_chunk.device) != str(self.device):
            raise AttributeError()
        self.cpu_chunk.copy_(self.npu_chunk, non_blocking=True)


class _FPDTGPUAttentionOffloadImpl_(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                attention_mask,
                cpg,
                scatter_idx,
                gather_idx,
                hidden_size,
                hidden_size_per_attention_head,
                dropout=0.0,
                num_chunks=8):

        do_save = q.requires_grad

        with torch.no_grad():

            per_gpu_seq_len = q.shape[0]
            chunk_size = per_gpu_seq_len // num_chunks
            if chunk_size * num_chunks != per_gpu_seq_len:
                raise ValueError()
            if attention_mask is not None:
                raise ValueError()
            ctx.num_chunks = num_chunks
            ctx.cpg = cpg
            ctx.scatter_idx = scatter_idx
            ctx.gather_idx = gather_idx
            ctx.hidden_size_per_attention_head = hidden_size_per_attention_head

            device = 'npu:{}'.format(torch.npu.current_device())
            ctx.device = device
            ctx.dtype = q.dtype

            global_q = []
            global_k = []
            global_v = []

            ctx.softmax_scale = hidden_size_per_attention_head**(-0.5)

            ctx.keep_p = 1. - dropout

            n = hidden_size // hidden_size_per_attention_head // mpu.get_context_parallel_world_size()
            ctx.n = n
            batch_size = q.shape[1]

            general_offload_stream = torch_npu.npu.Stream()
            offload_stream = torch_npu.npu.Stream()
            compute_stream = torch_npu.npu.default_stream()

            q_chunks = torch.chunk(q, chunks=num_chunks, dim=0)
            k_chunks = torch.chunk(k, chunks=num_chunks, dim=0)
            v_chunks = torch.chunk(v, chunks=num_chunks, dim=0)

            for i in range(num_chunks):

                q_chunk = q_chunks[i]
                k_chunk = k_chunks[i]
                v_chunk = v_chunks[i]

                q_chunk = q_chunk.reshape(q_chunk.shape[0], q_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
                k_chunk = k_chunk.reshape(k_chunk.shape[0], k_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
                v_chunk = v_chunk.reshape(v_chunk.shape[0], v_chunk.shape[1], -1, \
                                          hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()

                q_chunk = all_to_all(q_chunk, cpg, scatter_idx, gather_idx)
                k_chunk = all_to_all(k_chunk, cpg, scatter_idx, gather_idx)
                v_chunk = all_to_all(v_chunk, cpg, scatter_idx, gather_idx)

                dist.barrier()

                compute_stream.wait_stream(offload_stream)
                compute_stream.synchronize()
                with torch_npu.npu.stream(offload_stream):
                    global_q.append(ChunkManager(q_chunk, onload=False))
                    global_k.append(ChunkManager(k_chunk, onload=False))
                    global_v.append(ChunkManager(v_chunk, onload=False))


            del q_chunks, k_chunks, v_chunks

            global_o = []
            global_softmax_max = []
            global_softmax_sum = []

            output = []

            q_compute_chunk_idx = 0
            kv_compute_chunk_idx = 0

            for i in range(num_chunks):
                with torch_npu.npu.stream(offload_stream):
                    global_q[i].load_to_npu()
                    global_k[0].load_to_npu()
                    global_v[0].load_to_npu()
                compute_stream.wait_stream(offload_stream)
                compute_stream.synchronize()

                cur_attn_output = None
                cur_softmax_max = None
                cur_softmax_sum = None

                for k_i in range(num_chunks):

                    with torch_npu.npu.stream(compute_stream):

                        attn_outs = torch_npu.npu_fusion_attention(global_q[q_compute_chunk_idx].get_npu_chunk(),
                                                                       global_k[kv_compute_chunk_idx].get_npu_chunk(),
                                                                       global_v[kv_compute_chunk_idx].get_npu_chunk(),
                                                                       n,
                                                                       'BSND',
                                                                       scale=ctx.softmax_scale,
                                                                       keep_prob=ctx.keep_p)

                        cur_attn_output, cur_softmax_max, cur_softmax_sum = general_out_update(cur_attn_output,
                                                                                               cur_softmax_max,
                                                                                               cur_softmax_sum,
                                                                                               attn_outs)

                    can_offload_kv = True

                    if k_i != (len(global_k) - 1) or i != (num_chunks - 1):
                        if k_i != (len(global_k) - 1):
                            next_kv_compute_chunk_idx = k_i + 1
                        else:
                            next_kv_compute_chunk_idx = 0

                        if next_kv_compute_chunk_idx == kv_compute_chunk_idx:
                            can_offload_kv = False

                        else:
                            with torch_npu.npu.stream(offload_stream):
                                global_k[next_kv_compute_chunk_idx].load_to_npu()
                                global_v[next_kv_compute_chunk_idx].load_to_npu()

                    if i == num_chunks - 1 and k_i == num_chunks - 1:
                        with torch_npu.npu.stream(offload_stream):
                            global_q[0].load_to_npu()
                            global_k[0].load_to_npu()
                            global_v[0].load_to_npu()
                            global_o[0].load_to_npu()
                            global_softmax_max[0].load_to_npu()
                            global_softmax_sum[0].load_to_npu()

                    compute_stream.wait_stream(offload_stream)
                    compute_stream.synchronize()

                    if can_offload_kv:
                        global_k[kv_compute_chunk_idx].offload()
                        global_v[kv_compute_chunk_idx].offload()
                    kv_compute_chunk_idx = next_kv_compute_chunk_idx

                global_q[q_compute_chunk_idx].offload()
                q_compute_chunk_idx += 1

                cur_attn_output = cur_attn_output.to(ctx.dtype).contiguous()

                all2all_output = all_to_all(cur_attn_output, cpg, gather_idx, scatter_idx)
                output.append(all2all_output)

                with torch_npu.npu.stream(general_offload_stream):
                    global_o.append(ChunkManager(cur_attn_output))
                    global_softmax_max.append(ChunkManager(cur_softmax_max.contiguous()))
                    global_softmax_sum.append(ChunkManager(cur_softmax_sum.contiguous()))
                general_offload_stream.synchronize()

            compute_stream.wait_stream(general_offload_stream)
            compute_stream.synchronize()

            output = torch.cat(output, dim=1)
            output = output.reshape(output.shape[0], output.shape[1], -1).permute(1, 0, 2).contiguous()
            head_dim = output.shape[-1]

        if do_save:
            ctx.global_q = global_q
            ctx.global_k = global_k
            ctx.global_v = global_v
            ctx.attn_output = global_o
            ctx.global_softmax_max = global_softmax_max
            ctx.global_softmax_sum = global_softmax_sum
            ctx.head_dim = head_dim
            ctx.batch_size = batch_size

        return output

    @staticmethod
    def backward(ctx, dout):
        num_chunks = ctx.num_chunks
        device = ctx.device
        dtype = ctx.dtype
        cpg = ctx.cpg
        scatter_idx = ctx.scatter_idx
        gather_idx = ctx.gather_idx
        softmax_scale = ctx.softmax_scale
        keep_p = ctx.keep_p
        n = ctx.n
        hidden_size_per_attention_head = ctx.hidden_size_per_attention_head

        global_q = ctx.global_q
        global_k = ctx.global_k
        global_v = ctx.global_v
        attn_output = ctx.attn_output
        global_softmax_max = ctx.global_softmax_max
        global_softmax_sum = ctx.global_softmax_sum

        offload_stream = torch_npu.npu.Stream()
        compute_stream = torch_npu.npu.default_stream()

        grad_global_dout = [None for _ in range(num_chunks)]

        dout = dout.reshape(dout.shape[0], dout.shape[1], -1, \
                            hidden_size_per_attention_head).permute(1, 0, 2, 3).contiguous()
        chunk_size = dout.shape[1] // num_chunks

        grad_global_attn_output_chunk = all_to_all(dout[:, :chunk_size].contiguous(), cpg, scatter_idx, gather_idx)

        dout = dout[:, chunk_size:]

        grad_global_dout[0] = ChunkManager(grad_global_attn_output_chunk, onload=True)

        dq = [ChunkManager(torch.zeros(global_q[0].chunk_shape, dtype=torch.float, device=device), onload=True)] + \
            [ChunkManager(torch.zeros(global_q[0].chunk_shape, dtype=torch.float, device='cpu', pin_memory=True), device=device) for _ in range(num_chunks - 1)]

        dk_accum = torch.zeros(global_k[0].chunk_shape, dtype=torch.float, device=device)
        dv_accum = torch.zeros(global_v[0].chunk_shape, dtype=torch.float, device=device)

        torch_npu.npu.synchronize()
        dq_final = []
        dk_final = []
        dv_final = []

        for i in range(num_chunks):
            for q_i in range(num_chunks):
                with torch_npu.npu.stream(compute_stream):
                    attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                                    global_q[q_i].get_npu_chunk(),
                                    global_k[i].get_npu_chunk(),
                                    global_v[i].get_npu_chunk(),
                                    grad_global_dout[q_i].get_npu_chunk(),
                                    n,
                                    'BSND',
                                    softmax_max=global_softmax_max[q_i].get_npu_chunk(),
                                    softmax_sum=global_softmax_sum[q_i].get_npu_chunk(),
                                    attention_in=attn_output[q_i].get_npu_chunk(),
                                    scale_value=softmax_scale,
                                    keep_prob=keep_p)

                    cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

                if q_i != (len(global_q) - 1):
                    next_q_compute_chunk_idx = q_i + 1
                else:
                    next_q_compute_chunk_idx = 0

                can_offload_q = True

                if next_q_compute_chunk_idx == q_i:
                    can_offload_q = False

                else:

                    with torch_npu.npu.stream(offload_stream):

                        dq[next_q_compute_chunk_idx].load_to_npu()
                        global_q[next_q_compute_chunk_idx].load_to_npu()
                        attn_output[next_q_compute_chunk_idx].load_to_npu()
                        global_softmax_max[next_q_compute_chunk_idx].load_to_npu()
                        global_softmax_sum[next_q_compute_chunk_idx].load_to_npu()
                        if grad_global_dout[next_q_compute_chunk_idx] is not None:
                            grad_global_dout[next_q_compute_chunk_idx].load_to_npu()

                        else:

                            grad_global_attn_output_chunk = all_to_all(dout[:, :chunk_size].contiguous(), cpg, scatter_idx, gather_idx)

                            dist.barrier()
                            dout = dout[:, chunk_size:]

                            grad_global_dout[next_q_compute_chunk_idx] = ChunkManager(grad_global_attn_output_chunk, onload=True)

                compute_stream.wait_stream(offload_stream)
                compute_stream.synchronize()

                with torch_npu.npu.stream(compute_stream):
                    dq[q_i].get_npu_chunk().add_(cur_dq)
                    dk_accum.add_(cur_dk)
                    dv_accum.add_(cur_dv)

                offload_stream.wait_stream(compute_stream)
                with torch_npu.npu.stream(offload_stream):

                    dq[q_i].overwrite_to_cpu()

                if can_offload_q:

                    attn_output[q_i].offload()
                    global_softmax_max[q_i].offload()
                    global_softmax_sum[q_i].offload()
                    grad_global_dout[q_i].offload()

            compute_stream.wait_stream(offload_stream)
            compute_stream.synchronize()

            dk_accum = dk_accum.to(dtype)
            dv_accum = dv_accum.to(dtype)

            dk_accum = all_to_all(dk_accum.contiguous(), cpg, gather_idx, scatter_idx)
            dv_accum = all_to_all(dv_accum.contiguous(), cpg, gather_idx, scatter_idx)

            dist.barrier()

            dk_final.append(dk_accum)
            dv_final.append(dv_accum)

            with torch_npu.npu.stream(compute_stream):
                dk_accum = torch.zeros(global_k[i].chunk_shape, dtype=torch.float, device=device)
                dv_accum = torch.zeros(global_v[i].chunk_shape, dtype=torch.float, device=device)

            if i != (len(global_k) - 1):

                next_kv_compute_chunk_idx = i + 1

                global_k[next_kv_compute_chunk_idx].load_to_npu()
                global_v[next_kv_compute_chunk_idx].load_to_npu()

            compute_stream.wait_stream(offload_stream)
            compute_stream.synchronize()

            global_k[i].offload()
            global_v[i].offload()

        for i in range(num_chunks):
            with torch_npu.npu.stream(offload_stream):
                dq[i].load_to_npu()
            compute_stream.wait_stream(offload_stream)
            compute_stream.synchronize()

            dq_accum = dq[i].get_npu_chunk().to(dtype)
            dq_accum = all_to_all(dq_accum.contiguous(), cpg, gather_idx, scatter_idx)

            dq[i].offload()
            dq_final.append(dq_accum)

        dq = torch.cat(dq_final, dim=1)
        dk = torch.cat(dk_final, dim=1)
        dv = torch.cat(dv_final, dim=1)

        dq = dq.reshape(dq.shape[0], dq.shape[1], -1).permute(1, 0, 2).contiguous()
        dk = dk.reshape(dk.shape[0], dk.shape[1], -1).permute(1, 0, 2).contiguous()
        dv = dv.reshape(dv.shape[0], dv.shape[1], -1).permute(1, 0, 2).contiguous()

        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def fpdt_attention(
        q, k, v,
        attention_mask,
        ulysess_context_parallel_group,
        scatter_idx,
        gather_idx,
        hidden_size,
        head_dim,
        dropout,
        FPDT_chunk_number=8,
        FPDT_with_offload=True
):
    if FPDT_with_offload:
        return _FPDTGPUAttentionOffloadImpl_.apply(
            q,
            k,
            v,
            attention_mask,
            ulysess_context_parallel_group,
            scatter_idx,
            gather_idx,
            hidden_size,
            head_dim,
            dropout,
            FPDT_chunk_number
        )
    else:
        return _FPDTGPUAttentionImpl_.apply(
            q,
            k,
            v,
            attention_mask,
            ulysess_context_parallel_group,
            scatter_idx,
            gather_idx,
            hidden_size,
            head_dim,
            dropout,
            FPDT_chunk_number
        )


class FPDTFlashAttention(FlashAttention):
    def __init__(
            self,
            ulysess_context_parallel_group,
            hidden_size,
            head_dim,
            chunk_number=None,
            with_offload=None,
    ):
        super().__init__()

        args = get_args()
        self.ulysess_context_parallel_group = ulysess_context_parallel_group
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.FPDT_chunk_number = chunk_number
        self.FPDT_with_offload = with_offload

        if not self.FPDT_chunk_number:
            raise AttributeError("when powering FPDT, chunk_number has to be determined")

    def forward(self, query, key, value, attention_mask=None, **kwargs):
        query = query.view(query.shape[0], query.shape[1], -1)
        key = key.view(key.shape[0], key.shape[1], -1)
        value = value.view(value.shape[0], value.shape[1], -1)

        return fpdt_attention(
            query, key, value,
            attention_mask=attention_mask,
            ulysess_context_parallel_group=self.ulysess_context_parallel_group,
            scatter_idx=2,
            gather_idx=1,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            dropout=self.attention_dropout,
            FPDT_chunk_number=self.FPDT_chunk_number,
            FPDT_with_offload=self.FPDT_with_offload
        )
