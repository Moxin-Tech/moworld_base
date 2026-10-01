# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import os
from functools import lru_cache

import torch
import torch_npu
import torch.distributed as dist
import numpy as np
from einops import rearrange
from scipy.sparse.linalg import eigsh
from mindspeed.ops.npu_ring_attention_update import npu_ring_attention_update

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state


@lru_cache(maxsize=8)
def get_selection_indices_for_tnd_softmax_update(t, n, sub_seq_len):
    full_indices = list(range(t * n))
    cur_seq_start_idx = 0
    indices = []
    seq_start = 0
    for seq_len in sub_seq_len:
        for i in range(n):
            start = seq_start + seq_len * 2 * i + seq_len
            end = seq_start + seq_len * 2 * (i + 1)
            indices.extend(full_indices[start:end])
        seq_start += seq_len * n * 2

    return torch.tensor(indices)


def flatten_softmax(x, sub_seq_len):
    orig_shape = x.shape
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    merged = [item.view(orig_shape[1], -1, orig_shape[-1]).transpose(0, 1) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged


def unflatten_softmax(x, sub_seq_len):
    orig_shape = x.shape
    section_len = [s * orig_shape[1] for s in sub_seq_len]
    splits = x.view(-1, orig_shape[-1]).split(section_len, dim=0)
    # Modification: 下面reshape函数替换了core原先的view，否则会出现contiguous的报错
    merged = [item.view(-1, orig_shape[1], orig_shape[-1]).transpose(0, 1) \
              .reshape(-1, orig_shape[-1]) for item in splits]
    merged = torch.cat(merged, dim=0)
    return merged.view(*orig_shape)


def forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                 cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    if layout == 'TND':
        cur_softmax_max = flatten_softmax(cur_softmax_max, actual_seq_qlen)
        cur_softmax_sum = flatten_softmax(cur_softmax_sum, actual_seq_qlen)
        prev_softmax_max = flatten_softmax(prev_softmax_max, actual_seq_qlen)
        prev_softmax_sum = flatten_softmax(prev_softmax_sum, actual_seq_qlen)
    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    if layout == 'SBH':
        n = prev_out_scale.shape[1]
        h = prev_attn_out.shape[-1]
        d = h // n
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
        cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()
    elif layout == 'TND':
        d = prev_attn_out.shape[-1]
        prev_out_scale = prev_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)
        cur_out_scale = cur_out_scale[..., 0].unsqueeze(2).repeat(1, 1, d)

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    if layout == 'TND':
        softmax_max = unflatten_softmax(softmax_max, actual_seq_qlen)
        softmax_sum = unflatten_softmax(softmax_sum, actual_seq_qlen)
    return attn_out, softmax_max, softmax_sum


class RingP2P:
    def __init__(self, ring_global_ranks, group, group_for_send_recv_overlap=None, is_backward=False) -> None:
        self.group = group
        self.group_for_send_recv_overlap = group
        if group_for_send_recv_overlap is not None:
            self.group_for_send_recv_overlap = group_for_send_recv_overlap

        global_rank = dist.get_rank()
        ring_rank = ring_global_ranks.index(global_rank)
        ring_size = len(ring_global_ranks)
        self.next = ring_global_ranks[(ring_rank + 1) % ring_size]
        self.prev = ring_global_ranks[(ring_rank + ring_size - 1) % ring_size]
        self.ring_rank = ring_rank
        if is_backward:
            self.next, self.prev = self.prev, self.next

        self.send_recv_ops = []

    def async_send_recv(self, orig_send_tensor, orig_recv_tensor, shapes=None):
        send_tensor, recv_tensor = orig_send_tensor, orig_recv_tensor

        enable_mla = isinstance(orig_send_tensor, (list, tuple))
        if enable_mla:
            if shapes is not None:
                raise ValueError("MLA context parallel does not support uneven shapes yet.")
            if len(orig_send_tensor) != 2 or len(orig_recv_tensor) != 2:
                raise ValueError(
                    f"Expected tensors of length 2 (k,v), got lengths: "
                    f"send={len(orig_send_tensor)}, recv={len(orig_recv_tensor)}"
                )
            k_send, v_send = orig_send_tensor
            k_recv, v_recv = orig_recv_tensor
            if k_send.shape != k_recv.shape or v_send.shape != v_recv.shape:
                raise ValueError(
                    "Shape mismatch in KV tensors:\n"
                    f"  k_send: {k_send.shape} vs k_recv: {k_recv.shape}\n"
                    f"  v_send: {v_send.shape} vs v_recv: {v_recv.shape}"
                )
            k_shape, v_shape = k_send.shape, v_send.shape
            k_numel = k_send.numel()
            send_tensor = torch.cat((k_send.view(-1), v_send.view(-1)), dim=-1)
            recv_tensor = torch.cat((k_recv.view(-1), v_recv.view(-1)), dim=-1)

        if self.ring_rank % 2 == 0:
            if shapes is not None:
                send_tensor_shape_list = list(send_tensor.shape)
                send_tensor_shape_list[-3] = shapes[0]
                send_tensor.resize_(send_tensor_shape_list)
            send_op = dist.isend(send_tensor, self.next, self.group)
            if shapes is not None:
                recv_tensor_shape_list = list(recv_tensor.shape)
                recv_tensor_shape_list[-3] = shapes[1]
                recv_tensor.resize_(recv_tensor_shape_list)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(send_op)
            self.send_recv_ops.append(recv_op)
        else:
            if shapes is not None:
                recv_tensor_shape_list = list(recv_tensor.shape)
                recv_tensor_shape_list[-3] = shapes[1]
                recv_tensor.resize_(recv_tensor_shape_list)
            recv_op = dist.irecv(recv_tensor, self.prev, self.group)
            if shapes is not None:
                send_tensor_shape_list = list(send_tensor.shape)
                send_tensor_shape_list[-3] = shapes[0]
                send_tensor.resize_(send_tensor_shape_list)
            send_op = dist.isend(send_tensor, self.next, self.group_for_send_recv_overlap)
            self.send_recv_ops.append(recv_op)
            self.send_recv_ops.append(send_op)

        if enable_mla:
            # Important: The original k/v tensors are views into recv_tensor's memory.
            # Must synchronize async P2P operations before using these views
            # to prevent data races or corrupted memory.
            orig_recv_tensor[0] = recv_tensor[:k_numel].view(*k_shape)  # [k_numel] -> k_shape
            orig_recv_tensor[1] = recv_tensor[k_numel:].view(*v_shape)  # [v_numel] -> v_shape

    def wait(self):
        if len(self.send_recv_ops) > 0:
            for op in self.send_recv_ops:
                op.wait()
            self.send_recv_ops = []
            return 1
        else:
            return 0


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout='SBH'):
    """
    Updates the attention output and softmax statistics for the ring attention mechanism,
    with added parameters for enhanced flexibility and extensibility.

    This function is designed to update the attention output and related softmax statistics
    for a given sequence length in a ring attention mechanism. It handles the merging of
    previous and current attention outputs and their corresponding softmax statistics.
    The introduction of `actual_seq_qlen` and `layout` parameters allows for greater flexibility
    in handling variable sequence lengths and different tensor layouts, respectively.

    Parameters:
    - prev_attn_out (Tensor): The attention output from the previous process.
    - prev_softmax_max (Tensor): The maximum value of the softmax distribution from the previous process.
    - prev_softmax_sum (Tensor): The sum of the softmax distribution from the previous process.
    - cur_attn_out (Tensor): The attention output from the current process.
    - cur_softmax_max (Tensor): The maximum value of the softmax distribution from the current process.
    - cur_softmax_sum (Tensor): The sum of the softmax distribution from the current process.
    - actual_seq_qlen (Tensor, optional): The actual sequence length for the query. This parameter
                                      is crucial for handling variable-length sequences and ensuring
                                      that the attention mechanism operates correctly under such conditions.
                                      If not provided, it defaults to the length of the current attention output.
    - layout (str, optional): The layout format of the input tensors. This parameter allows for the specification
                              of different tensor layouts, enhancing the function's versatility across various
                              model architectures. Default is 'SBH', where:
        - S: Sequence length
        - B: Batch size
        - H: Hidden size (number of attention heads)

    Returns:
    - updated_attn_out (Tensor): The updated attention output after merging previous and current process.
    - updated_softmax_max (Tensor): The updated maximum value of the softmax distribution.
    - updated_softmax_sum (Tensor): The updated sum of the softmax distribution.
    """
    # Modification: 融合算子的tnd格式需要保证D轴是64的倍数
    if layout == "TND" and cur_attn_out.shape[-1] % 64 == 0 or layout != "TND":
        def accumulate_list(input_list):
            """
            借助numpy库将列表转换为numpy数组进行元素累加，再转换回列表并在开头添加0
            """
            np_array = np.array(input_list)
            cumsum_result = np.cumsum(np_array)
            return torch.tensor([0] + list(cumsum_result), dtype=torch.int64).to(prev_attn_out.device)

        if layout == "TND":
            actual_seq_qlen = accumulate_list(actual_seq_qlen)
        return npu_ring_attention_update(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)

    return forward_update_without_fused(prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                         cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout)


def tnd_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs, q_index, softmax_indices, cur_sub_out_seq_len):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs

    layout = 'TND'

    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=cur_sub_out_seq_len, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        n = attn_out.shape[1]
        t = attn_out.shape[0]
        prev_softmax_max = softmax_max.view(-1, 8)[softmax_indices].view(-1, n, 8)
        prev_softmax_sum = softmax_sum.view(-1, 8)[softmax_indices].view(-1, n, 8)

        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            torch.index_select(attn_out, 0, q_index), prev_softmax_max, prev_softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=cur_sub_out_seq_len, layout=layout
        )
        attn_out.index_copy_(0, q_index, attn_out_updated)
        softmax_max = softmax_max.view(-1, 8).index_copy(0, softmax_indices, softmax_max_updated.view(-1, 8)).view(-1, n, 8)
        softmax_sum = softmax_sum.view(-1, 8).index_copy(0, softmax_indices, softmax_sum_updated.view(-1, 8)).view(-1, n, 8)


    return [attn_out, softmax_max, softmax_sum, rng_states]


def causal_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    if len(cur_attn_outs) > 3:
        rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    elif kv_block_id <= q_block_id:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
    else:
        # [2s, b, h] -> [2, s, b, h]
        attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
        # [b, n, 2s, 8] -> [b, n, 2, s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                        2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                        2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
            cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=None, layout=layout
        )
        attn_out[1].copy_(attn_out_updated)
        softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
        softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
        # [2, s, b, h] -> [2s, b, h]
        attn_out = attn_out.view(-1, *attn_out.shape[2:])
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                        softmax_max.shape[-1])
        softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                        softmax_sum.shape[-1])

    return [attn_out, softmax_max, softmax_sum, rng_states]


def general_out_update(q_block_id, kv_block_id, cur_attn_outs, global_attn_outs):
    cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
    attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
    layout = 'SBH'
    rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])
    if q_block_id == kv_block_id:
        attn_out = cur_attn_out
        softmax_max = cur_softmax_max
        softmax_sum = cur_softmax_sum
    else:
        attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
            attn_out, softmax_max, softmax_sum,
            cur_attn_out, cur_softmax_max, cur_softmax_sum, layout=layout
        )
        attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

    return [attn_out, softmax_max, softmax_sum, rng_states]
