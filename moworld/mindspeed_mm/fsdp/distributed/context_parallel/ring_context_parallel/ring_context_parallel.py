# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from abc import ABC, abstractmethod
from typing import Any

from functools import lru_cache
from dataclasses import dataclass
from einops import rearrange
import torch
import torch_npu

from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention, npu_fusion_attention_grad

from .context_parallel_kv_cache import ContextParallelKVCache
from .utils import RingP2P, tnd_out_update, causal_out_update, general_out_update, forward_update, get_selection_indices_for_tnd_softmax_update


def causal_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_mask=None):
    cur_attn_mask = None
    if q_block_id == kv_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_attn_mask = attn_mask
        cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
    elif kv_block_id <= q_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_q = q.view(-1, *q.shape[2:])
        # only k[0] v[0] need to be calculated
        cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # only q[1] need to be calculated
        cur_q = q[1]
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

    return cur_q, cur_k, cur_v, cur_attn_mask


def tnd_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, fetch_ptrs, attn_mask=None):
    seqlen, half_seqlen, q_index, kv_index = fetch_ptrs
    actual_seq_qlen, actual_seq_kvlen, sub_out_seq_len = seqlen
    half_actual_seq_qlen, half_actual_seq_kvlen, half_sub_out_seq_len = half_seqlen

    cur_attn_mask = None
    if q_block_id == kv_block_id:
        cur_attn_mask = attn_mask
        cur_q = q
        cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, actual_seq_kvlen
        cur_sub_out_seq_len = sub_out_seq_len
    elif kv_block_id <= q_block_id:
        cur_q = q
        cur_k, cur_v = [torch.index_select(x, 0, kv_index) for x in [cur_k, cur_v]]
        cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, half_actual_seq_kvlen
        cur_sub_out_seq_len = sub_out_seq_len
    else:
        cur_q = torch.index_select(q, 0, q_index)
        cur_seq_qlen, cur_seq_kvlen = half_actual_seq_qlen, actual_seq_kvlen
        cur_sub_out_seq_len = half_sub_out_seq_len

    return cur_q, cur_k, cur_v, cur_attn_mask, (cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len)


def tnd_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                          softmax_values, seq_lens, index_values, attn_mask=None):
    # fetch backward output
    actual_seq_qlen, actual_seq_kvlen, half_actual_seq_kvlen, half_actual_seq_qlen = seq_lens
    softmax_max, softmax_sum, half_softmax_max, half_softmax_sum = softmax_values
    q_index, kv_index = index_values
    cur_attn_mask = None
    if q_block_id >= kv_block_id:
        if q_block_id == kv_block_id:
            cur_attn_mask = attn_mask
            cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, actual_seq_kvlen
        else:
            cur_k, cur_v = [torch.index_select(x, 0, kv_index) for x in [cur_k, cur_v]]
            cur_seq_qlen, cur_seq_kvlen = actual_seq_qlen, half_actual_seq_kvlen

        cur_q, cur_attn_out, cur_dout = q, attn_out, dout
        cur_softmax_max, cur_softmax_sum = softmax_max, softmax_sum
    else:
        cur_q, cur_attn_out, cur_dout = [torch.index_select(x, 0, q_index) for x in [q, attn_out, dout]]
        cur_softmax_max, cur_softmax_sum = half_softmax_max, half_softmax_sum
        cur_seq_qlen, cur_seq_kvlen = half_actual_seq_qlen, actual_seq_kvlen

    return (cur_q, cur_k, cur_v), cur_attn_out, cur_dout, (cur_softmax_max, cur_softmax_sum), cur_attn_mask, (cur_seq_qlen, cur_seq_kvlen)


def causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                          softmax_max, softmax_sum, attn_mask=None):
    cur_attn_mask = None
    if q_block_id >= kv_block_id:
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        cur_softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                            softmax_max.shape[-1])
        cur_softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                            softmax_sum.shape[-1])
        # [2, s, b, h] -> [2s, b, h]
        cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
        if q_block_id == kv_block_id:
            cur_attn_mask = attn_mask
            # [2, s, b, h] -> [2s, b, h]
            cur_k, cur_v, = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        else:
            cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        # only q[1] attn_out[1] and dout[1] need to be calculated
        cur_q, cur_attn_out, cur_dout = [x[1] for x in [q, attn_out, dout]]
        cur_softmax_max, cur_softmax_sum = [x[:, :, 1, :, :] for x in [softmax_max, softmax_sum]]

    return cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask


def tnd_grad_update(q_block_id, kv_block_id, cur_attn_grads, global_attn_grads,
                                        q_index, kv_index):
    cur_dq, cur_dk, cur_dv = cur_attn_grads
    dq, dk, dv = global_attn_grads
    if q_block_id == kv_block_id:
        dq.add_(cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)
    elif q_block_id > kv_block_id:
        dq.add_(cur_dq)
        dk.index_add_(0, kv_index, cur_dk)
        dv.index_add_(0, kv_index, cur_dv)
    else:
        dq.index_add_(0, q_index, cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)

    return dq, dk, dv


def causal_grad_update(q_block_id, kv_block_id, cur_dq, cur_dk, cur_dv, dq, dk, dv):
    if q_block_id == kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        cur_dk = cur_dk.view(dk.shape)
        cur_dv = cur_dv.view(dv.shape)
        dq.add_(cur_dq)
        dk.add_(cur_dk)
        dv.add_(cur_dv)
    elif q_block_id > kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        dq.add_(cur_dq)
        dk[0].add_(cur_dk)
        dv[0].add_(cur_dv)
    else:
        dq[1].add_(cur_dq)
        cur_dk = cur_dk.view(dk.shape) # [2s, b, h] -> [2, s, b, h]
        cur_dv = cur_dv.view(dv.shape)
        dk.add_(cur_dk)
        dv.add_(cur_dv)

    return dq, dk, dv


def cal_row(cur_q, cur_k, cur_v, s, attn_info):
    # q: [s, b, h], kv: [2s, b, h]
    n, pse, pse_type, attn_mask, softmax_scale, keep_prob, \
    q_index_list, kv_index_list = attn_info

    # r1c0
    cur_attn_mask = None
    attn_outs_r1c0 = npu_fusion_attention(
        cur_q, cur_k[:s], cur_v[:s], n, 'SBH',
        pse=pse,
        pse_type=pse_type,
        padding_mask=None,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else kv_index_list
    )
    # r1c1
    cur_attn_mask = attn_mask
    attn_outs_r1c1 = npu_fusion_attention(
        cur_q, cur_k[s:], cur_v[s:], n, 'SBH',
        pse=pse,
        pse_type=pse_type,
        padding_mask=None,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[1] * s, ] if kv_index_list is not None else kv_index_list
    )

    # update row1
    attn_out = attn_outs_r1c0[0]
    softmax_max = attn_outs_r1c0[1]
    softmax_sum = attn_outs_r1c0[2]
    curr_attn_out = attn_outs_r1c1[0]
    curr_softmax_max = attn_outs_r1c1[1]
    curr_softmax_sum = attn_outs_r1c1[2]
    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(attn_out, softmax_max, softmax_sum,
                                                                                curr_attn_out, curr_softmax_max,
                                                                                curr_softmax_sum)
    return [attn_out_updated, softmax_max_updated, softmax_sum_updated]


def flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s):
    n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob, \
    q_index_list, kv_index_list = attn_info
    cur_q, cur_k, cur_v = cur_qkv
    if q_block_id == kv_block_id:
        attn_outs_r0c0 = npu_fusion_attention(
            cur_q[:s], cur_k[:s], cur_v[:s], n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s, ] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else None,
        )
        attn_outs_r1 = cal_row(cur_q[s:], cur_k, cur_v, s, attn_info)
        # get output
        attn_outs = []
        attn_outs.append(torch.cat([attn_outs_r0c0[0], attn_outs_r1[0]]))
        attn_outs.append(torch.cat([attn_outs_r0c0[1], attn_outs_r1[1]], dim=2))
        attn_outs.append(torch.cat([attn_outs_r0c0[2], attn_outs_r1[2]], dim=2))
    elif q_block_id > kv_block_id:
        attn_outs_r0c0 = npu_fusion_attention(
            cur_q[:s], cur_k, cur_v, n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s, ] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else None,
        )
        attn_outs_r1c0 = npu_fusion_attention(
            cur_q[s:], cur_k, cur_v, n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else None,
        )
        # get output
        attn_outs = []
        attn_outs.append(torch.cat([attn_outs_r0c0[0], attn_outs_r1c0[0]]))
        attn_outs.append(torch.cat([attn_outs_r0c0[1], attn_outs_r1c0[1]], dim=2))
        attn_outs.append(torch.cat([attn_outs_r0c0[2], attn_outs_r1c0[2]], dim=2))
    else:
        attn_outs = cal_row(cur_q, cur_k, cur_v, s, attn_info)

    return attn_outs


def cal_row_grad(cur_q, cur_k, cur_v, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_out,
                 attn_grad_info, s, kv_block_id):
    n, pse, pse_type, attn_mask, softmax_scale, keep_prob, rng_states, \
    q_index_list, kv_index_list = attn_grad_info

    cur_attn_mask = None
    attn_grad_outs_r1c0 = npu_fusion_attention_grad(
        cur_q, cur_k[:s], cur_v[:s], cur_dout, n, 'SBH',
        pse=pse,
        pse_type=pse_type,
        padding_mask=None,
        softmax_max=cur_softmax_max,
        softmax_sum=cur_softmax_sum,
        attention_in=cur_attn_out,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        seed=rng_states[kv_block_id][0],
        offset=rng_states[kv_block_id][1],
        numels=rng_states[kv_block_id][2],
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else kv_index_list
    )

    cur_attn_mask = attn_mask
    attn_grad_outs_r1c1 = npu_fusion_attention_grad(
        cur_q, cur_k[s:], cur_v[s:], cur_dout, n, 'SBH',
        pse=pse,
        pse_type=pse_type,
        padding_mask=None,
        softmax_max=cur_softmax_max,
        softmax_sum=cur_softmax_sum,
        attention_in=cur_attn_out,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        seed=rng_states[kv_block_id][0],
        offset=rng_states[kv_block_id][1],
        numels=rng_states[kv_block_id][2],
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[1] * s, ] if kv_index_list is not None else kv_index_list
    )

    return attn_grad_outs_r1c0, attn_grad_outs_r1c1


def flash_attention_with_alibi_pse_grad(q_block_id, kv_block_id, cur_qkv, cur_dout, cur_attn_out,
                                        cur_softmax_max, cur_softmax_sum, attn_grad_info, s):
    n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob, rng_states, \
    q_index_list, kv_index_list = attn_grad_info
    cur_q, cur_k, cur_v = cur_qkv

    if q_block_id == kv_block_id:
        attn_grad_outs_r0c0 = npu_fusion_attention_grad(
            cur_q[:s], cur_k[:s], cur_v[:s], cur_dout[:s], n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, :s],
            softmax_sum=cur_softmax_sum[:, :, :s],
            attention_in=cur_attn_out[:s],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s, ] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs_r1c0, attn_grad_outs_r1c1 = cal_row_grad(
            cur_q[s:], cur_k, cur_v, cur_dout[s:], cur_softmax_max[:, :, s:], cur_softmax_sum[:, :, s:],
            cur_attn_out[s:], attn_grad_info, s, kv_block_id
        )
        attn_grad_outs = []
        attn_grad_outs.append(torch.cat(
            [attn_grad_outs_r0c0[0], attn_grad_outs_r1c0[0] + attn_grad_outs_r1c1[0]]))
        attn_grad_outs.append(torch.cat(
            [attn_grad_outs_r0c0[1] + attn_grad_outs_r1c0[1], attn_grad_outs_r1c1[1]]))
        attn_grad_outs.append(torch.cat(
            [attn_grad_outs_r0c0[2] + attn_grad_outs_r1c0[2], attn_grad_outs_r1c1[2]]))

    elif q_block_id > kv_block_id:
        attn_grad_outs_r0c0 = npu_fusion_attention_grad(
            cur_q[:s], cur_k, cur_v, cur_dout[:s], n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, :s],
            softmax_sum=cur_softmax_sum[:, :, :s],
            attention_in=cur_attn_out[:s],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s, ] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs_r1c0 = npu_fusion_attention_grad(
            cur_q[s:], cur_k, cur_v, cur_dout[s:], n, 'SBH',
            pse=pse,
            pse_type=pse_type,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, s:],
            softmax_sum=cur_softmax_sum[:, :, s:],
            attention_in=cur_attn_out[s:],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[1] * s, ] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s, ] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs = []
        attn_grad_outs.append(torch.cat([attn_grad_outs_r0c0[0], attn_grad_outs_r1c0[0]]))
        attn_grad_outs.append(attn_grad_outs_r0c0[1] + attn_grad_outs_r1c0[1])
        attn_grad_outs.append(attn_grad_outs_r0c0[2] + attn_grad_outs_r1c0[2])

    else:
        attn_grad_outs_r1c0, attn_grad_outs_r1c1 = cal_row_grad(
            cur_q, cur_k, cur_v, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_out,
            attn_grad_info, s, kv_block_id
        )
        attn_grad_outs = []
        attn_grad_outs.append(attn_grad_outs_r1c0[0] + attn_grad_outs_r1c1[0])
        attn_grad_outs.append(torch.cat([attn_grad_outs_r1c0[1], attn_grad_outs_r1c1[1]]))
        attn_grad_outs.append(torch.cat([attn_grad_outs_r1c0[2], attn_grad_outs_r1c1[2]]))


    return attn_grad_outs


def get_unaligned_cp_shapes(shapes, block_id, next_block_id):
    if shapes is None:
        return None
    unaligned_cp_shapes = [shapes[block_id], shapes[next_block_id]]
    return unaligned_cp_shapes


@lru_cache(maxsize=8)
def compute_qkv_index(seq_lens):
    full_indices = list(range(seq_lens[-1]))
    prev_eod_pos = 0
    kv_indices = []
    q_indices = []
    for eod_pos in seq_lens:
        mid = (eod_pos + prev_eod_pos) // 2
        kv_indices.extend(full_indices[prev_eod_pos:mid])
        q_indices.extend(full_indices[mid:eod_pos])
        prev_eod_pos = eod_pos

    kv_index = torch.tensor(kv_indices, device=torch.npu.current_device())
    q_index = torch.tensor(q_indices, device=torch.npu.current_device())

    return q_index, kv_index


class AttentionStrategy(ABC):
    """Attention strategy base class"""

    @abstractmethod
    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        pass

    @abstractmethod
    def update_out(self, cp_config):
        pass

    @abstractmethod
    def compute_fused_attention_grad(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        pass


class CausalRegularAttentionStrategy(AttentionStrategy):
    """Causal attention strategy when using pse or layout is SBH and BNSD """

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        if cp_config.pse is None:
            # Causal attention when layout is SBH and BNSD
            return self._standard_causal_attention(attn_mask, n, q, softmax_scale, cp_config)
        else:
            # Causal attention when using pse
            return self._pse_causal_attention(attn_mask, n, q, softmax_scale, cp_config)

    def _standard_causal_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal attention when layout is SBH and BNSD
        cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(cp_config.q_block_id, cp_config.kv_block_id, q,
                                                                  cp_config.cur_k, cp_config.cur_v, attn_mask)
        layout = "SBH"
        pre_tockens_value = cur_k.shape[0]
        if cp_config.megatron_cp_in_bnsd:
            cur_q = rearrange(cur_q, 's b (h d) -> b h s d', h=n).contiguous()
            kv_n = cur_v.shape[2] // cur_q.shape[3]
            cur_k, cur_v = [rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cur_k, cur_v]]
            layout = "BNSD"
            pre_tockens_value = cur_k.shape[2]

        attn_outs = torch_npu.npu_fusion_attention(
            cur_q, cur_k, cur_v, n, layout,
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
            keep_prob=cp_config.keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0
        )
        if cp_config.megatron_cp_in_bnsd:
            attn_outs = rearrange(attn_outs[0], 'b h s d -> s b (h d)').contiguous(), attn_outs[1], attn_outs[2]
        return attn_outs, cp_config.cur_sub_out_seq_len

    def _pse_causal_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal attention when using pse
        cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(cp_config.q_block_id, cp_config.kv_block_id, q,
                                                                  cp_config.cur_k, cp_config.cur_v, attn_mask)
        q_index_list = [cp_config.q_block_id, cp_config.cp_size * 2 - 1 - cp_config.q_block_id]
        kv_index_list = [cp_config.kv_block_id, cp_config.cp_size * 2 - 1 - cp_config.kv_block_id]
        attn_info = [n, cp_config.pse, cp_config.pse_type, cur_attn_mask, softmax_scale, cp_config.keep_prob,
                     q_index_list, kv_index_list]
        s = q.shape[1]
        attn_outs = flash_attention_with_alibi_pse(
            cp_config.q_block_id, cp_config.kv_block_id,
            (cur_q, cur_k, cur_v),
            attn_info,
            s
        )
        return attn_outs, cp_config.cur_sub_out_seq_len

    def update_out(self, cp_config):
        return causal_out_update(cp_config.q_block_id, cp_config.kv_block_id, cp_config.attn_outs,
                                 cp_config.global_attn_outs)

    def compute_fused_attention_grad(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        if cp_config.pse is None:
            # Causal attention backward when layout is SBH and BNSD
            return self._standard_causal_backward(attn_mask, n, q, softmax_scale, attn_out, dout, cp_config)
        else:
            # Causal attention backward when using pse
            return self._pse_causal_backward(attn_mask, n, q, softmax_scale, attn_out, dout, cp_config)

    def _standard_causal_backward(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # Causal attention backward when layout is SBH and BNSD
        step_inputs = causal_backward_fetch(cp_config.q_block_id, cp_config.kv_block_id, q, cp_config.cur_k,
                                            cp_config.cur_v, attn_out, dout,
                                            cp_config.softmax_max, cp_config.softmax_sum, attn_mask=attn_mask)
        cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask = step_inputs
        layout = "SBH"
        pre_tockens_value = cur_k.shape[0]
        if cp_config.megatron_cp_in_bnsd:
            cur_q, cur_dout, cur_attn_out = [rearrange(x1, 's b (h d) -> b h s d', h=n).contiguous() for x1 in [cur_q, cur_dout, cur_attn_out]]
            kv_n = cur_v.shape[2] // cur_q.shape[3]
            cur_k, cur_v = [rearrange(x1, 's b (h d) -> b h s d', h=kv_n).contiguous() for x1 in [cur_k, cur_v]]
            layout = "BNSD"
            pre_tockens_value = cur_k.shape[2]
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q, cur_k, cur_v, cur_dout, n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            softmax_max=cur_softmax_max,
            softmax_sum=cur_softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        if cp_config.megatron_cp_in_bnsd:
            attn_grad_outs = [rearrange(x1, 'b h s d -> s b (h d)').contiguous() for x1 in [attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]]]
        return attn_grad_outs

    def _pse_causal_backward(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # Causal attention backward when using pse
        step_inputs = causal_backward_fetch(cp_config.q_block_id, cp_config.kv_block_id, q, cp_config.cur_k,
                                            cp_config.cur_v, attn_out, dout,
                                            cp_config.softmax_max, cp_config.softmax_sum, attn_mask=attn_mask)
        cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask = step_inputs
        q_index_list = [cp_config.q_block_id, cp_config.cp_size * 2 - 1 - cp_config.q_block_id]
        kv_index_list = [cp_config.kv_block_id, cp_config.cp_size * 2 - 1 - cp_config.kv_block_id]
        attn_grad_info = [n, cp_config.pse, cp_config.pse_type, cur_attn_mask, softmax_scale, cp_config.keep_prob,
                          cp_config.rng_states,
                          q_index_list, kv_index_list]
        s = q.shape[1]
        attn_grad_outs = flash_attention_with_alibi_pse_grad(
            cp_config.q_block_id, cp_config.kv_block_id,
            (cur_q, cur_k, cur_v), cur_dout, cur_attn_out,
            cur_softmax_max, cur_softmax_sum,
            attn_grad_info, s
        )
        return attn_grad_outs


class CausalEodAttentionStrategy(AttentionStrategy):
    """Causal eod attention strategy"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # Causal eod attention
        cur_q, cur_k, cur_v, cur_attn_mask, cur_seq_lens = tnd_forward_fetch(cp_config.q_block_id,
                                                                             cp_config.kv_block_id, q, cp_config.cur_k,
                                                                             cp_config.cur_v, cp_config.fetch_ptrs,
                                                                             attn_mask)
        cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len = cur_seq_lens
        # flash attention forward
        attn_outs = torch_npu.npu_fusion_attention(
            cur_q, cur_k, cur_v, n, "TND",
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tockens=cur_k.shape[0],
            next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
            keep_prob=cp_config.keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen
        )

        return attn_outs, cur_sub_out_seq_len

    def update_out(self, cp_config):
        return tnd_out_update(cp_config.q_block_id, cp_config.kv_block_id, cp_config.attn_outs,
                              cp_config.global_attn_outs, cp_config.q_index, cp_config.softmax_indices,
                              cp_config.cur_sub_out_seq_len)

    def compute_fused_attention_grad(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # Causal eod attention backward
        softmax_values = (
        cp_config.softmax_max, cp_config.softmax_sum, cp_config.half_softmax_max, cp_config.half_softmax_sum)
        seq_lens = (
            cp_config.actual_seq_qlen, cp_config.actual_seq_kvlen, cp_config.half_actual_seq_qlen,
            cp_config.half_actual_seq_kvlen)
        index_values = (cp_config.q_index, cp_config.kv_index)
        step_inputs = tnd_backward_fetch(cp_config.q_block_id, cp_config.kv_block_id, q, cp_config.cur_k,
                                         cp_config.cur_v, attn_out, dout,
                                         softmax_values, seq_lens, index_values, attn_mask=attn_mask)
        qkv, cur_attn_out, cur_dout, cur_softmax_values, cur_attn_mask, cur_seq_lens = step_inputs
        cur_q, cur_k, cur_v = qkv
        cur_softmax_max, cur_softmax_sum = cur_softmax_values
        cur_seq_qlen, cur_seq_kvlen = cur_seq_lens
        # flash attention backward
        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q, cur_k, cur_v, cur_dout, n,
            "TND",
            pse=None,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            softmax_max=cur_softmax_max,
            softmax_sum=cur_softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=cur_k.shape[0],
            next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            actual_seq_qlen=cur_seq_qlen,
            actual_seq_kvlen=cur_seq_kvlen,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        return attn_grad_outs


class GeneralAttentionStrategy(AttentionStrategy):
    """General attention strategy"""

    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):
        # General attention
        this_mask = AttentionWithCp.compute_mask(
            cp_config.actual_seq_qlen, cp_config.actual_seq_kvlen,
            cp_config.q_block_id, cp_config.kv_block_id,
            attn_mask
        )

        layout = "SBH"
        pre_tockens_value = cp_config.cur_k.shape[0]

        cur_q = q
        if cp_config.megatron_cp_in_bnsd:
            cur_q = rearrange(q, 's b (h d) -> b h s d', h=n).contiguous()
            kv_n = cp_config.cur_v.shape[2] // cur_q.shape[3]
            cp_config.cur_k, cp_config.cur_v = [rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cp_config.cur_k, cp_config.cur_v]]
            layout = "BNSD"
            pre_tockens_value = cp_config.cur_k.shape[2]

        attn_outs = torch_npu.npu_fusion_attention(
            cur_q, cp_config.cur_k, cp_config.cur_v, n, layout,
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            scale=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=pre_tockens_value,
            keep_prob=cp_config.keep_prob,
            sparse_mode=1
        )
        if cp_config.megatron_cp_in_bnsd:
            attn_outs = rearrange(attn_outs[0], 'b h s d -> s b (h d)').contiguous(), attn_outs[1], attn_outs[2], attn_outs[3], attn_outs[4], attn_outs[5], attn_outs[6]
        return attn_outs, cp_config.cur_sub_out_seq_len

    def update_out(self, cp_config):
        return general_out_update(cp_config.q_block_id, cp_config.kv_block_id, cp_config.attn_outs,
                                  cp_config.global_attn_outs)

    def compute_fused_attention_grad(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        # General attention backward
        this_mask = AttentionWithCp.compute_mask(
            cp_config.actual_seq_qlen, cp_config.actual_seq_kvlen,
            cp_config.q_block_id, cp_config.kv_block_id,
            attn_mask
        )

        layout = "SBH"
        pre_tockens_value = cp_config.cur_k.shape[0]
        cur_q = q
        cur_dout = dout
        cur_attn_out = attn_out
        if cp_config.megatron_cp_in_bnsd:
            cur_q, cur_dout, cur_attn_out = [rearrange(x, 's b (h d) -> b h s d', h=n).contiguous() for x in [q, dout, attn_out]]
            kv_n = cp_config.cur_v.shape[2] // cur_q.shape[3]
            cp_config.cur_k, cp_config.cur_v = [rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cp_config.cur_k, cp_config.cur_v]]
            layout = "BNSD"
            pre_tockens_value = cp_config.cur_k.shape[2]

        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q, cp_config.cur_k, cp_config.cur_v, cur_dout, n,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=this_mask,
            softmax_max=cp_config.softmax_max,
            softmax_sum=cp_config.softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            pre_tockens=pre_tockens_value,
            next_tockens=pre_tockens_value,
            sparse_mode=1,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )
        if cp_config.megatron_cp_in_bnsd:
            attn_grad_outs = [rearrange(x, 'b h s d -> s b (h d)').contiguous() for x in [attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]]]
        return attn_grad_outs


class AttentionStrategyFactory:
    """Attention strategy factory for getting strategy"""

    @staticmethod
    def get_strategy(causal, is_eod_reset):
        # Get strategy by args
        if causal:
            if is_eod_reset:
                return CausalEodAttentionStrategy()
            else:
                return CausalRegularAttentionStrategy()
        else:
            return GeneralAttentionStrategy()


class KVCacheManager:
    """KV Cache Manager"""

    def __init__(self, cache_policy):
        self.cache_policy = cache_policy
        self.k_cache_list = []
        self.v_cache_list = []

    def update_cache(self, cur_kv):
        # update the cache by kv cache policy
        if self.cache_policy == "full":
            self.k_cache_list.append(cur_kv[0].clone())
            self.v_cache_list.append(cur_kv[1].clone())
        elif self.cache_policy == "half":
            self.k_cache_list.append(cur_kv[0].clone())

    def get_cache(self, cur_kv):
        # get kv cache when config is not causal or eod
        self.k_cache_list = self.k_cache_list if self.k_cache_list else [cur_kv[0].clone()]
        self.v_cache_list = self.v_cache_list if self.v_cache_list else [cur_kv[1].clone()]
        k_stack = torch.stack(self.k_cache_list)
        v_stack = torch.stack(self.v_cache_list)
        return k_stack, v_stack

    def get_cache_causal_regular(self, cur_kv, q):
        # get kv cache when config is causal and is not eod, as causal regular condition
        self.k_cache_list = self.k_cache_list if self.k_cache_list else [cur_kv[0].clone()]
        self.v_cache_list = self.v_cache_list if self.v_cache_list else [cur_kv[1].clone()]
        q = q.view(-1, *q.shape[2:])
        self.k_cache_list = [x.view(-1, *x.shape[2:]) for x in self.k_cache_list]
        self.v_cache_list = [x.view(-1, *x.shape[2:]) for x in self.v_cache_list]
        k_stack = torch.stack(self.k_cache_list)
        v_stack = torch.stack(self.v_cache_list)
        return k_stack, v_stack, q


@dataclass
class AttentionWithCpConfig:
    """Config in CP attention"""

    # Config that init from cp_para
    causal: Any = None
    cp_group: Any = None
    cp_size: Any = None
    rank: Any = None
    cp_global_ranks: Any = None
    cp_inner_ranks: Any = None
    cp_outer_ranks: Any = None
    kv_block_id: Any = None
    keep_prob: Any = None
    cp_group_for_send_recv_overlap: Any = None
    cp_group_for_intra_window: Any = None
    cp_group_for_intra_window_send_recv_overlap: Any = None
    megatron_cp_in_bnsd: Any = None
    cache_policy: Any = None
    pse: Any = None
    pse_type: Any = None

    # Config that compute derived
    actual_seq_qlen: Any = None
    actual_seq_kvlen: Any = None
    is_eod_reset: Any = None
    inner_size: Any = None
    outer_size: Any = None
    rng_states: Any = None

    # Temporary config
    attn_outs: Any = None
    cur_k: Any = None
    cur_sub_out_seq_len: Any = None
    cur_v: Any = None
    fetch_ptrs: Any = None
    q_block_id: Any = None
    q_index: Any = None
    kv_index: Any = None
    softmax_indices: Any = None
    n: Any = None
    q: Any = None
    softmax_scale: Any = None
    half_actual_seq_qlen: Any = None
    half_actual_seq_kvlen: Any = None
    half_sub_out_seq_len: Any = None
    dout: Any = None
    softmax_max: Any = None
    softmax_sum: Any = None
    attn_out: Any = None
    half_softmax_max: Any = None
    half_softmax_sum: Any = None

    @classmethod
    def init_from_para(cls, cp_para):
        # Config init from cp_para
        return cls(
            causal=cp_para['causal'],
            cp_group=cp_para.get("cp_group"),
            cp_size=cp_para.get("cp_size"),
            rank=cp_para.get("rank"),
            cp_global_ranks=cp_para.get("cp_global_ranks"),
            cp_inner_ranks=cp_para.get("cp_inner_ranks", [torch.distributed.get_rank()]),
            cp_group_for_send_recv_overlap=cp_para.get("cp_group_for_send_recv_overlap"),
            cp_group_for_intra_window=cp_para.get('cp_group_for_intra_window'),
            cp_group_for_intra_window_send_recv_overlap=cp_para.get(
                'cp_group_for_intra_window_send_recv_overlap'),
            megatron_cp_in_bnsd=cp_para.get('megatron_cp_in_bnsd'),
            cache_policy=cp_para.get("cache_policy"),
            pse=cp_para.get("pse"),
            pse_type=cp_para.get("pse_type")
        )

    def get_backward_config(self):
        # Get the config need use in backward
        return AttentionWithCpConfig(
            # Config need use in backward
            causal=self.causal,
            cp_group=self.cp_group,
            cp_size=self.cp_size,
            rank=self.rank,
            cp_global_ranks=self.cp_global_ranks,
            cp_inner_ranks=self.cp_inner_ranks,
            cp_outer_ranks=self.cp_outer_ranks,
            kv_block_id=self.kv_block_id,
            keep_prob=self.keep_prob,
            rng_states=self.rng_states,
            pse=self.pse,
            pse_type=self.pse_type,
            cp_group_for_send_recv_overlap=self.cp_group_for_send_recv_overlap,
            cp_group_for_intra_window=self.cp_group_for_intra_window,
            cp_group_for_intra_window_send_recv_overlap=self.cp_group_for_intra_window_send_recv_overlap,
            actual_seq_qlen=self.actual_seq_qlen,
            actual_seq_kvlen=self.actual_seq_kvlen,
            is_eod_reset=self.is_eod_reset,
            megatron_cp_in_bnsd=self.megatron_cp_in_bnsd,
            cache_policy=self.cache_policy,
            # Config not need use in backward
            inner_size=None,
            outer_size=None,
            attn_outs=None,
            cur_k=None,
            cur_sub_out_seq_len=None,
            cur_v=None,
            fetch_ptrs=None,
            q_block_id=None,
            q_index=None,
            kv_index=None,
            softmax_indices=None,
            n=None,
            q=None,
            softmax_scale=None,
            half_actual_seq_qlen=None,
            half_actual_seq_kvlen=None,
            half_sub_out_seq_len=None,
            dout=None,
            softmax_max=None,
            softmax_sum=None,
            attn_out=None,
            half_softmax_max=None,
            half_softmax_sum=None
        )


class AttentionWithCp(torch.autograd.Function):
    """Attention implementation with context parallelism"""

    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                packed_seq_params=None, shapes=None):

        enable_mla = k.shape[-1] != v.shape[-1]
        ctx.enable_mla = enable_mla

        cp_config = AttentionWithCpConfig.init_from_para(cp_para)
        cp_config.cp_outer_ranks = cp_para.get("cp_outer_ranks", cp_config.cp_global_ranks)
        cp_config.inner_size = len(cp_config.cp_inner_ranks)
        cp_config.outer_size = cp_config.cp_size // cp_config.inner_size
        cp_config.keep_prob = 1. - dropout_p

        inner_ring = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                             cp_config.cp_group_for_intra_window_send_recv_overlap)
        outer_ring = RingP2P(cp_config.cp_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap)

        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q_padded is not None:
            cu_seqlens_q_padded_div_cp = packed_seq_params.cu_seqlens_q_padded // cp_config.cp_size
            q_index, kv_index = compute_qkv_index(tuple(cu_seqlens_q_padded_div_cp.tolist()))
            packed_seq_params.q_index = q_index
            packed_seq_params.kv_index = kv_index
            cp_config.actual_seq_kvlen = cu_seqlens_q_padded_div_cp.tolist() if packed_seq_params else None
            cp_config.actual_seq_qlen = cu_seqlens_q_padded_div_cp.tolist() if packed_seq_params else None
        else:
            cp_config.actual_seq_kvlen = packed_seq_params.cu_seqlens_q.tolist() if packed_seq_params else None
            cp_config.actual_seq_qlen = packed_seq_params.cu_seqlens_kv.tolist() if packed_seq_params else None
        cp_config.is_eod_reset = (cp_config.actual_seq_kvlen is not None) and (cp_config.actual_seq_qlen is not None)

        if softmax_scale is None:
            head_dim = q.shape[-1] if packed_seq_params is not None and cp_para['causal'] else q.shape[-1] // n
            softmax_scale = head_dim ** (-0.5)
        if cp_config.causal and attn_mask is None:
            attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        if cp_config.causal:
            if cp_config.is_eod_reset:

                # only first half of each sub sequence KV block need to be calculated when i <= rank
                cp_config.kv_index = packed_seq_params.kv_index
                # only last half of each sub sequence q block need to be calculated when i > rank
                cp_config.q_index = packed_seq_params.q_index

                sub_out_seq_len = (torch.tensor([0] + cp_config.actual_seq_qlen)[1:] - torch.tensor(
                    [0] + cp_config.actual_seq_qlen)[:-1]).tolist()
                seq_lens = (cp_config.actual_seq_qlen, cp_config.actual_seq_kvlen, sub_out_seq_len)
                half_seq_lens = [[x // 2 for x in lst] for lst in seq_lens]
                cp_config.fetch_ptrs = (seq_lens, half_seq_lens, cp_config.q_index, cp_config.kv_index)

                cp_config.softmax_indices = get_selection_indices_for_tnd_softmax_update(q.shape[0], q.shape[1],
                                                                                         tuple(half_seq_lens[2])).to(q.device)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        if enable_mla:
            cur_kv = [k, v]
            next_kv = [torch.empty_like(k), torch.empty_like(v)]
            next_round_kv = [torch.empty_like(k), torch.empty_like(v)]
        else:
            cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, h]
            next_kv = torch.empty_like(cur_kv)
            next_round_kv = torch.empty_like(cur_kv)

        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        cp_config.rng_states = [[0, 0, 0] for _ in range(cp_config.cp_size)]
        cp_config.global_attn_outs = [attn_out, softmax_max, softmax_sum, cp_config.rng_states]
        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = cp_config.rank, cp_config.rank, cp_config.rank

        cache_manager = KVCacheManager(cp_config.cache_policy)
        attention_strategy = AttentionStrategyFactory.get_strategy(cp_config.causal, cp_config.is_eod_reset)

        for j in range(cp_config.outer_size):
            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // cp_config.inner_size) * cp_config.inner_size
            if j < cp_config.outer_size - 1:
                next_kv_block_id_outer = (kv_block_id_outer + cp_config.cp_size - cp_config.inner_size) % cp_config.cp_size
                outer_ring.async_send_recv(cur_kv, next_round_kv, shapes=get_unaligned_cp_shapes(shapes, kv_block_id_outer, next_kv_block_id_outer))
            for i in range(cp_config.inner_size):
                # wait until KV is received from recv_src
                if i < cp_config.inner_size - 1:
                    next_kv_block_id = (cp_config.kv_block_id + cp_config.inner_size - 1) % cp_config.inner_size + kv_block_offset
                    inner_ring.async_send_recv(cur_kv, next_kv, shapes=get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id))

                cp_config.cur_k, cp_config.cur_v = cur_kv[0], cur_kv[1]  # [2, s, b, h]

                # cache kv or k
                if j * cp_config.inner_size + i + 2 != cp_config.cp_size:
                    cache_manager.update_cache(cur_kv)
                if cp_config.causal:
                    cp_config.attn_outs = None
                    cp_config.cur_sub_out_seq_len = None
                cp_config.attn_outs, cp_config.cur_sub_out_seq_len = attention_strategy.compute_fused_attention(
                    attn_mask, n, q, softmax_scale, cp_config)
                cp_config.global_attn_outs = attention_strategy.update_out(cp_config)

                if inner_ring.wait():
                    cur_kv, next_kv = next_kv, cur_kv  # double buffer
                    cp_config.kv_block_id = (cp_config.kv_block_id + cp_config.inner_size - 1) % cp_config.inner_size + kv_block_offset

            if outer_ring.wait():
                cur_kv, next_round_kv = next_round_kv, cur_kv  # double buffer
                kv_block_id_outer = (kv_block_id_outer + cp_config.cp_size - cp_config.inner_size) % cp_config.cp_size

        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]

        attn_out, softmax_max, softmax_sum, cp_config.rng_states = cp_config.global_attn_outs
        if cp_config.causal and not cp_config.is_eod_reset:
            k_stack, v_stack, q = cache_manager.get_cache_causal_regular(cur_kv, q)
        else:
            k_stack, v_stack = cache_manager.get_cache(cur_kv)

        ctx.save_for_backward(q, k_stack, v_stack, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.softmax_scale = softmax_scale
        ctx.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_config.cp_global_ranks)
        ctx.shapes = shapes
        ctx.cp_config = cp_config.get_backward_config()

        if cp_config.causal and cp_config.is_eod_reset:
            # Config need use in backward when condition is causal and eod
            ctx.cp_config.q_index = cp_config.q_index
            ctx.cp_config.kv_index = cp_config.kv_index
            ctx.cp_config.half_actual_seq_qlen = half_seq_lens[0]
            ctx.cp_config.half_actual_seq_kvlen = half_seq_lens[1]
            ctx.cp_config.half_sub_out_seq_len = half_seq_lens[2]
            ctx.cp_config.sub_out_seq_len = sub_out_seq_len
            ctx.cp_config.softmax_indices = cp_config.softmax_indices
            return attn_out

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        cp_config = ctx.cp_config
        q, k_stack, v_stack, *attn_mask, attn_out, cp_config.softmax_max, cp_config.softmax_sum = ctx.saved_tensors
        attn_mask = attn_mask[0] if len(attn_mask) == 1 else attn_mask

        n = ctx.n
        shapes = ctx.shapes
        softmax_scale = ctx.softmax_scale

        # Reversed order of forward
        inner_size = len(cp_config.cp_inner_ranks)
        outer_size = len(cp_config.cp_outer_ranks)

        intra_kv_comm = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                                cp_config.cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        intra_dkv_comm = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                                 cp_config.cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        inter_kv_comm = RingP2P(cp_config.cp_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap,
                                is_backward=True)
        inter_dkv_comm = RingP2P(ctx.cp_dkv_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap,
                                 is_backward=True)

        if cp_config.causal:
            if cp_config.is_eod_reset:
                cp_config.half_softmax_max = cp_config.softmax_max.view(-1, 8)[cp_config.softmax_indices].view(-1, n, 8)
                cp_config.half_softmax_sum = cp_config.softmax_sum.view(-1, 8)[cp_config.softmax_indices].view(-1, n, 8)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
                q, attn_out, dout = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, attn_out, dout]]
                k_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in k_stack]
                v_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in v_stack]
                # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                cp_config.softmax_max = cp_config.softmax_max.view(cp_config.softmax_max.shape[0],
                                                                   cp_config.softmax_max.shape[1],
                                                                   2, cp_config.softmax_max.shape[2] // 2,
                                                                   cp_config.softmax_max.shape[-1])
                cp_config.softmax_sum = cp_config.softmax_sum.view(cp_config.softmax_sum.shape[0],
                                                                   cp_config.softmax_sum.shape[1],
                                                                   2, cp_config.softmax_sum.shape[2] // 2,
                                                                   cp_config.softmax_sum.shape[-1])

        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = cp_config.rank, cp_config.kv_block_id, cp_config.kv_block_id
        if shapes:
            cur_shapes_list = list(k_stack[-1].shape)
            cur_shapes_list[-3] = shapes[cp_config.kv_block_id]
            cur_dkv = torch.zeros((2, *cur_shapes_list), dtype=k_stack[-1].dtype, device=k_stack[-1].device)
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        elif not ctx.enable_mla:
            cur_dkv = torch.zeros((2, *k_stack[-1].shape), dtype=k_stack[-1].dtype, device=k_stack[-1].device)
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        else:
            k_tmp, v_tmp = k_stack[-1], v_stack[-1]
            cur_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_round_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]

        outer_data = (outer_size, inter_kv_comm)
        inner_data = (inner_size, intra_kv_comm)
        cp_kv_cache = ContextParallelKVCache(cp_config.cache_policy, outer_data, inner_data, k_stack, v_stack)

        dq = torch.zeros_like(q) # [2, s, b, h]
        for j in range(outer_size):
            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // inner_size) * inner_size

            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            cp_kv_cache.communicate_outer_ring_kv(j, shapes=get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id_outer))

            for i in range(inner_size):
                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                cp_config.cur_k, cp_config.cur_v = cp_kv_cache.communicate_inner_ring_kv(i, get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id))

                dq_step, dk_step, dv_step = AttentionWithCp.backward_step_helper(attn_mask, n, q, softmax_scale,
                                                                                 attn_out, dout, cp_config)

                if i == 0 and j > 0: # receive dk dv from last window
                    inter_dkv_comm.wait()
                    cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
                elif i > 0: # receive dk dv from last step
                    intra_dkv_comm.wait()
                    cur_dkv, next_dkv = next_dkv, cur_dkv

                dk, dv = cur_dkv[0], cur_dkv[1]
                # update qkv grades
                if cp_config.is_eod_reset and cp_config.causal:
                    tnd_grad_update(cp_config.q_block_id, cp_config.kv_block_id, (dq_step, dk_step, dv_step), (dq, dk, dv),
                                    cp_config.q_index, cp_config.kv_index)
                elif cp_config.causal:
                    causal_grad_update(cp_config.q_block_id, cp_config.kv_block_id, dq_step, dk_step, dv_step, dq, dk, dv)
                else:
                    dq.add_(dq_step)
                    dk.add_(dk_step)
                    dv.add_(dv_step)

                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                if i + 1 != inner_size:
                    intra_dkv_comm.async_send_recv(cur_dkv, next_dkv, shapes=get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id))

                cp_config.kv_block_id = next_kv_block_id

            if intra_dkv_comm.wait():
                cur_dkv, next_dkv = next_dkv, cur_dkv

            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            if j + 1 != outer_size:
                inter_dkv_comm.async_send_recv(cur_dkv, next_round_dkv, shapes=get_unaligned_cp_shapes(shapes, cp_config.kv_block_id, next_kv_block_id_outer))

            kv_block_id_outer = next_kv_block_id_outer

        if inter_dkv_comm.wait():
            cur_dkv, next_round_dkv = next_round_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]

        # [2, s, b, h] -> [2s, b, h]
        if cp_config.causal and not cp_config.is_eod_reset:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None, None, None

    @classmethod
    def backward_step_helper(cls, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        attention_strategy = AttentionStrategyFactory.get_strategy(cp_config.causal, cp_config.is_eod_reset)
        attn_grad_outs = attention_strategy.compute_fused_attention_grad(attn_mask, n, q, softmax_scale, attn_out, dout,
                                                                         cp_config)
        return attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

    @classmethod
    def compute_mask(cls, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
        from bisect import bisect_right
        from mindspeed.utils import batch_index

        if actual_seq_qlen:
            seq_len = actual_seq_qlen[-1] // AttentionWithCp.batch_size
            actual_seq_qlen = batch_index(actual_seq_qlen, seq_len)
            actual_seq_kvlen = batch_index(actual_seq_kvlen, seq_len)
            block_size = cls.block_size
            actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
            sub_seq_qlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_qlen]
            sub_seq_qid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]).npu() # B S

            this_ids = sub_seq_qid[:, q_block_id * block_size:(q_block_id + 1) * block_size].npu()
            this_tile = this_ids.unsqueeze(dim=2) # B S 1

            actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
            sub_seq_kvlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_kvlen]
            sub_seq_kvid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]).npu() # B S
            other_ids = sub_seq_kvid[:, kv_block_id * block_size:(kv_block_id + 1) * block_size].npu()
            other_tile = other_ids.unsqueeze(dim=1) # B 1 S

            mask = this_tile == other_tile # B S S
            if kv_block_id > q_block_id:
                mask = torch.zeros_like(mask)
            elif kv_block_id == q_block_id:
                mask = torch.tril(mask)

            return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
        else:
            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None


def ringattn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                              packed_seq_params=None, shapes=None):
    AttentionWithCp.block_size = q.shape[0]
    AttentionWithCp.batch_size = q.shape[1]
    out = AttentionWithCp.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p,
        packed_seq_params, shapes
    )
    return out


class TNDGeneralAttentionStrategy(AttentionStrategy):
    """General attention strategy implementation using TND layout."""
    def compute_fused_attention(self, attn_mask, n, q, softmax_scale, cp_config):

        layout = "TND"

        cur_q = q
        attn_outs = torch_npu.npu_fusion_attention(
            cur_q, cp_config.cur_k, cp_config.cur_v, n, layout,
            pse=None,
            padding_mask=None,
            scale=softmax_scale,
            keep_prob=cp_config.keep_prob,
            actual_seq_qlen=cp_config.actual_seq_qlen,
            actual_seq_kvlen=cp_config.actual_seq_kvlen,
            sparse_mode=0
        )

        return attn_outs, cp_config.cur_sub_out_seq_len

    def update_out(self, cp_config):
        """General attention strategy implementation using TND layout.

        This strategy utilizes NPU fused attention operations for optimized performance
        with TND (Time, Number of heads, Dimension) tensor layout.
        """
        q_block_id, kv_block_id, cur_attn_outs, global_attn_outs, cur_sub_out_seq_len = \
            cp_config.q_block_id, cp_config.kv_block_id, cp_config.attn_outs, cp_config.global_attn_outs, cp_config.cur_sub_out_seq_len

        # Unpack current attention outputs
        cur_attn_out, cur_softmax_max, cur_softmax_sum = cur_attn_outs[0], cur_attn_outs[1], cur_attn_outs[2]
        attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
        layout = 'TND'

        # Update RNG states for dropout if present
        if len(cur_attn_outs) > 3:
            rng_states[kv_block_id] = (cur_attn_outs[4], cur_attn_outs[5], cur_attn_outs[6])

        if q_block_id == kv_block_id:
            # 对角线
            attn_out = cur_attn_out
            softmax_max = cur_softmax_max
            softmax_sum = cur_softmax_sum
        else:
            # 非对角线，由于full mask，所以不需要考虑q_block_id是否大于kv_block_id
            attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                attn_out, softmax_max, softmax_sum,
                cur_attn_out, cur_softmax_max, cur_softmax_sum, actual_seq_qlen=cur_sub_out_seq_len, layout=layout
            )
            attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

        return [attn_out, softmax_max, softmax_sum, rng_states]

    def compute_fused_attention_grad(self, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):

        layout = "TND"
        cur_q = q
        cur_dout = dout
        cur_attn_out = attn_out

        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
            cur_q, cp_config.cur_k, cp_config.cur_v, cur_dout, n,
            layout,
            pse=None,
            padding_mask=None,
            softmax_max=cp_config.softmax_max,
            softmax_sum=cp_config.softmax_sum,
            attention_in=cur_attn_out,
            scale_value=softmax_scale,
            sparse_mode=0,
            actual_seq_qlen=cp_config.actual_seq_qlen,
            actual_seq_kvlen=cp_config.actual_seq_kvlen,
            keep_prob=cp_config.keep_prob,
            seed=cp_config.rng_states[cp_config.kv_block_id][0],
            offset=cp_config.rng_states[cp_config.kv_block_id][1],
            numels=cp_config.rng_states[cp_config.kv_block_id][2],
        )

        return attn_grad_outs


class AttentionWithCpTNDGeneral(torch.autograd.Function):
    """General Attention implementation with context parallelism using TND layout."""
    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                packed_seq_params=None, split_seq_lens_per_cp_rank=None):
        '''split_seq_lens_perrank表示各个序列划分到所有rank上的长度，例如：
           cp_rank1: s1_1, s2_1, s3_1
           cp_rank2: s1_2, s2_2, s3_2
        '''

        enable_mla = k.shape[-1] != v.shape[-1]
        ctx.enable_mla = enable_mla

        # Initialize context parallelism configuration
        cp_config = AttentionWithCpConfig.init_from_para(cp_para)
        cp_config.cp_outer_ranks = cp_para.get("cp_outer_ranks", cp_config.cp_global_ranks)
        cp_config.inner_size = len(cp_config.cp_inner_ranks)
        cp_config.outer_size = cp_config.cp_size // cp_config.inner_size
        cp_config.keep_prob = 1. - dropout_p

        inner_ring = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                             cp_config.cp_group_for_intra_window_send_recv_overlap)
        outer_ring = RingP2P(cp_config.cp_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap)

        # 计算每个rank上的所有子序列的和 以及 累加序列
        total_len_per_cp_rank = None
        if split_seq_lens_per_cp_rank is not None:
            total_len_per_cp_rank = split_seq_lens_per_cp_rank.sum(1).tolist()
            cu_seq_len_per_cp_rank = split_seq_lens_per_cp_rank.cumsum(1)
            cu_seq_len_per_cp_rank = torch.nn.functional.pad(cu_seq_len_per_cp_rank, (1, 0), value=0).tolist()

        # 为当前rank计算ascual seq len
        cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[cp_config.rank] if split_seq_lens_per_cp_rank is not None else None
        cp_config.actual_seq_qlen = cu_seq_len_per_cp_rank[cp_config.rank] if split_seq_lens_per_cp_rank is not None else None
        cp_config.is_eod_reset = (cp_config.actual_seq_kvlen is not None) and (cp_config.actual_seq_qlen is not None)
        cp_config.cur_sub_out_seq_len = (torch.tensor(cp_config.actual_seq_qlen)[1:] - \
            torch.tensor(cp_config.actual_seq_qlen)[:-1]).tolist() if split_seq_lens_per_cp_rank is not None else None

        if softmax_scale is None:
            head_dim = q.shape[-1]
            softmax_scale = head_dim ** (-0.5)
        if attn_mask is None:
            # 使用压缩矩阵
            attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        # kv拼接用于通信
        cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, t, n, d]
        next_kv = torch.empty_like(cur_kv)
        next_round_kv = torch.empty_like(cur_kv)

        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        cp_config.rng_states = [[0, 0, 0] for _ in range(cp_config.cp_size)]
        cp_config.global_attn_outs = [attn_out, softmax_max, softmax_sum, cp_config.rng_states]
        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = cp_config.rank, cp_config.rank, cp_config.rank

        cache_manager = KVCacheManager(cp_config.cache_policy)
        attention_strategy = TNDGeneralAttentionStrategy()

        # Outer loop: process outer context parallelism windows
        for j in range(cp_config.outer_size):

            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // cp_config.inner_size) * cp_config.inner_size

            # Prepare next outer ring communication (if not last window)
            if j < cp_config.outer_size - 1:
                next_kv_block_id_outer = (kv_block_id_outer + cp_config.cp_size - cp_config.inner_size) % cp_config.cp_size
                outer_ring.async_send_recv(cur_kv, next_round_kv, shapes=get_unaligned_cp_shapes(total_len_per_cp_rank, kv_block_id_outer, next_kv_block_id_outer))# shapes是各个rank上的序列长度

            # Inner loop: process inner context parallelism windows
            for i in range(cp_config.inner_size):
                # wait until KV is received from recv_src
                if i < cp_config.inner_size - 1:
                    next_kv_block_id = (cp_config.kv_block_id + cp_config.inner_size - 1) % cp_config.inner_size + kv_block_offset
                    inner_ring.async_send_recv(cur_kv, next_kv, shapes=get_unaligned_cp_shapes(total_len_per_cp_rank, cp_config.kv_block_id, next_kv_block_id))

                cp_config.cur_k, cp_config.cur_v = cur_kv[0], cur_kv[1]  # [t, n, d]

                # cache kv or k
                if j * cp_config.inner_size + i + 2 != cp_config.cp_size:
                    cache_manager.update_cache(cur_kv)
                if cp_config.causal:
                    cp_config.attn_outs = None
                    cp_config.cur_sub_out_seq_len = None
                cp_config.attn_outs, cp_config.cur_sub_out_seq_len = attention_strategy.compute_fused_attention(
                    attn_mask, n, q, softmax_scale, cp_config)
                cp_config.global_attn_outs = attention_strategy.update_out(cp_config)

                if inner_ring.wait():
                    cur_kv, next_kv = next_kv, cur_kv  # double buffer
                    cp_config.kv_block_id = (cp_config.kv_block_id + cp_config.inner_size - 1) % cp_config.inner_size + kv_block_offset
                    # 更新cu_seq_kvlen
                    cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[cp_config.kv_block_id] if split_seq_lens_per_cp_rank is not None else cp_config.actual_seq_kvlen

            if outer_ring.wait():
                cur_kv, next_round_kv = next_round_kv, cur_kv  # double buffer
                kv_block_id_outer = (kv_block_id_outer + cp_config.cp_size - cp_config.inner_size) % cp_config.cp_size
                # 更新cu_seq_kvlen
                cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[kv_block_id_outer] if split_seq_lens_per_cp_rank is not None else cp_config.actual_seq_kvlen

        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]
        # Extract final attention outputs
        attn_out, softmax_max, softmax_sum, cp_config.rng_states = cp_config.global_attn_outs

        k_stack, v_stack = cache_manager.get_cache(cur_kv)

        # Save tensors for backward pass
        ctx.save_for_backward(q, k_stack, v_stack, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.softmax_scale = softmax_scale
        ctx.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_config.cp_global_ranks)
        ctx.split_seq_lens_per_cp_rank = split_seq_lens_per_cp_rank
        ctx.cp_config = cp_config.get_backward_config()

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        cp_config = ctx.cp_config
        # Restore saved tensors from forward pass
        q, k_stack, v_stack, *attn_mask, attn_out, cp_config.softmax_max, cp_config.softmax_sum = ctx.saved_tensors
        attn_mask = attn_mask[0] if len(attn_mask) == 1 else attn_mask

        n = ctx.n
        split_seq_lens_per_cp_rank = ctx.split_seq_lens_per_cp_rank
        softmax_scale = ctx.softmax_scale

        # Reversed order of forward
        inner_size = len(cp_config.cp_inner_ranks)
        outer_size = len(cp_config.cp_outer_ranks)

        # Communication rings for KV and gradient exchange
        intra_kv_comm = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                                cp_config.cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        intra_dkv_comm = RingP2P(cp_config.cp_inner_ranks, cp_config.cp_group_for_intra_window,
                                 cp_config.cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        inter_kv_comm = RingP2P(cp_config.cp_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap,
                                is_backward=True)
        inter_dkv_comm = RingP2P(ctx.cp_dkv_outer_ranks, cp_config.cp_group, cp_config.cp_group_for_send_recv_overlap,
                                 is_backward=True)

        # 计算每个rank上的所有子序列的和 以及 累加序列
        total_len_per_cp_rank = None
        if split_seq_lens_per_cp_rank is not None:
            total_len_per_cp_rank = split_seq_lens_per_cp_rank.sum(1).tolist()
            cu_seq_len_per_cp_rank = split_seq_lens_per_cp_rank.cumsum(1)
            cu_seq_len_per_cp_rank = torch.nn.functional.pad(cu_seq_len_per_cp_rank, (1, 0), value=0).tolist()

        # Initialize block IDs and sequence lengths
        cp_config.q_block_id, cp_config.kv_block_id, kv_block_id_outer = cp_config.rank, cp_config.kv_block_id, cp_config.kv_block_id
        cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[cp_config.kv_block_id] if split_seq_lens_per_cp_rank is not None else None
        cp_config.actual_seq_qlen = cu_seq_len_per_cp_rank[cp_config.q_block_id] if split_seq_lens_per_cp_rank is not None else None

        if total_len_per_cp_rank:
            cur_shapes_list = list(k_stack[-1].shape)
            cur_shapes_list[-3] = total_len_per_cp_rank[cp_config.kv_block_id]
            cur_dkv = torch.zeros((2, *cur_shapes_list), dtype=k_stack[-1].dtype, device=k_stack[-1].device)
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        elif not ctx.enable_mla:
            cur_dkv = torch.zeros((2, *k_stack[-1].shape), dtype=k_stack[-1].dtype, device=k_stack[-1].device)
            next_dkv = cur_dkv.clone()
            next_round_dkv = cur_dkv.clone()
        else:
            k_tmp, v_tmp = k_stack[-1], v_stack[-1]
            cur_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]
            next_round_dkv = [torch.zeros_like(k_tmp), torch.zeros_like(v_tmp)]

        outer_data = (outer_size, inter_kv_comm)
        inner_data = (inner_size, intra_kv_comm)
        cp_kv_cache = ContextParallelKVCache(cp_config.cache_policy, outer_data, inner_data, k_stack, v_stack)

        dq = torch.zeros_like(q) # [2, t, n, d]
        # Outer loop: process outer windows in reverse order
        for j in range(outer_size):
            cp_config.kv_block_id = kv_block_id_outer
            kv_block_offset = (cp_config.kv_block_id // inner_size) * inner_size

            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            cp_kv_cache.communicate_outer_ring_kv(j, shapes=get_unaligned_cp_shapes(total_len_per_cp_rank, cp_config.kv_block_id, next_kv_block_id_outer))

            # Inner loop: process inner windows in reverse order
            for i in range(inner_size):
                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                cp_config.cur_k, cp_config.cur_v = cp_kv_cache.communicate_inner_ring_kv(i, get_unaligned_cp_shapes(total_len_per_cp_rank, cp_config.kv_block_id, next_kv_block_id))

                # Compute gradients for current window
                dq_step, dk_step, dv_step = AttentionWithCpTNDGeneral.backward_step_helper(attn_mask, n, q, softmax_scale,
                                                                                 attn_out, dout, cp_config)

                if i == 0 and j > 0: # receive dk dv from last window
                    inter_dkv_comm.wait()
                    cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
                elif i > 0: # receive dk dv from last step
                    intra_dkv_comm.wait()
                    cur_dkv, next_dkv = next_dkv, cur_dkv

                dk, dv = cur_dkv[0], cur_dkv[1]

                # update qkv grades
                dq.add_(dq_step)
                dk.add_(dk_step)
                dv.add_(dv_step)

                next_kv_block_id = (cp_config.kv_block_id + 1) % inner_size + kv_block_offset
                if i + 1 != inner_size:
                    intra_dkv_comm.async_send_recv(cur_dkv, next_dkv, shapes=get_unaligned_cp_shapes(total_len_per_cp_rank, cp_config.kv_block_id, next_kv_block_id))

                cp_config.kv_block_id = next_kv_block_id
                # 更新cu_seq_kvlen
                cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[cp_config.kv_block_id] if split_seq_lens_per_cp_rank is not None else cp_config.actual_seq_kvlen

            if intra_dkv_comm.wait():
                cur_dkv, next_dkv = next_dkv, cur_dkv


            next_kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_config.cp_size
            if j + 1 != outer_size:
                inter_dkv_comm.async_send_recv(cur_dkv, next_round_dkv, shapes=get_unaligned_cp_shapes(total_len_per_cp_rank, cp_config.kv_block_id, next_kv_block_id_outer))

            kv_block_id_outer = next_kv_block_id_outer
            # 更新cu_seq_kvlen
            cp_config.actual_seq_kvlen = cu_seq_len_per_cp_rank[kv_block_id_outer] if split_seq_lens_per_cp_rank is not None else cp_config.actual_seq_kvlen

        if inter_dkv_comm.wait():
            cur_dkv, next_round_dkv = next_round_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]

        # [2, s, b, h] -> [2s, b, h]
        if cp_config.causal and not cp_config.is_eod_reset:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None, None, None

    @classmethod
    def backward_step_helper(cls, attn_mask, n, q, softmax_scale, attn_out, dout, cp_config):
        attention_strategy = TNDGeneralAttentionStrategy()
        attn_grad_outs = attention_strategy.compute_fused_attention_grad(attn_mask, n, q, softmax_scale, attn_out, dout,
                                                                         cp_config)
        return attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]


def ringattn_context_parallel_tnd_general(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                              packed_seq_params=None, shapes=None):
    out = AttentionWithCpTNDGeneral.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p,
        packed_seq_params, shapes
    )
    return out
