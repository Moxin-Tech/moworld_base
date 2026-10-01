import math
import torch
import torch_npu
from mindspeed.core.context_parallel import AttnMaskType
from mindspeed.core.context_parallel import mpu as parallel_state
from mindspeed.core.context_parallel.adaptive_context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import AttentionWithCp, ringattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import cal_split_sizes, split_forward_gather_backward, gather_forward_split_backward
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.model.transformer import get_attention_mask
from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.utils.utils import ensure_valid


try:
    from einops import rearrange
except ImportError:
    rearrange = None


@classmethod
def vlm_cp_compute_mask(cls, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
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

        return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
    else:
        return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None


def vlm_cp_dot_product_attention_forward(
    self,
    query,
    key,
    value,
    attention_mask,
    attn_mask_type=None,
    attention_bias=None,
    packed_seq_params=None,
):
    if attention_mask is None and self.attn_mask_type == AttnMaskType.causal:
        if not getattr(self.config, 'is_llava', False):
            attention_mask = get_attention_mask()
            if self.config.attention_mask_type == 'causal':
                self.config.sparse_mode = 2
            if getattr(self.config, 'reset_attention_mask', False):
                if self.config.attention_mask_type == 'general':
                    self.config.sparse_mode = 2
                    if not (self.config.context_parallel_size == 1 or self.config.context_parallel_algo == 'ulysses_cp_algo'):
                        self.config.sparse_mode = 1

    sparse_mode = self.config.sparse_mode
    is_ulysses_algo = getattr(self.config, 'context_parallel_algo', None) in ['ulysses_cp_algo', 'hybrid_cp_algo']
    if packed_seq_params is not None and self.config.attention_mask_type == 'causal':
        attention_mask = torch.triu(
                        torch.ones((2048, 2048),
                        device='npu', dtype=torch.bool), diagonal=1)
        sparse_mode = 2
    ensure_valid(attention_bias is None, 'Attention bias is not supported for DotProductAttention.')

    if packed_seq_params is not None and not is_ulysses_algo:
        #TND
        T, n_head, D = query.shape[0], query.shape[1], query.shape[2]
    else:
        seq_length, bsz, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]

    if packed_seq_params is not None and not is_ulysses_algo:
        # TND
        cp_size = parallel_state.get_context_parallel_world_size()
        actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
        actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
        shape_order = 'TND'
    else:
        # SBH
        actual_seq_qlen = None if packed_seq_params is None else packed_seq_params.cu_seqlens_q.tolist()
        actual_seq_kvlen = None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv.tolist()
        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        shape_order = 'SBH'

    if attn_mask_type == AttnMaskType.no_mask:
        sparse_mode = 0  # default mask

    scale = 1.0 / math.sqrt(
        self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

    cp_expanded_by_2d_tp = getattr(self.config, 'tp_2d', False) and getattr(self.config, 'tp_y', 1) > 1
    if cp_expanded_by_2d_tp:
        tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
    else:
        tp_y_cp_sz = self.config.context_parallel_size

    if (self.config.context_parallel_size > 1 and self.config.context_parallel_algo == 'ulysses_cp_algo'
            and self.config.context_parallel_kv_cache_policy):
        self.ulysses_comm_para['cache_policy'] = get_cache_policy(
            self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
        )
        self.ulysses_comm_para['use_ulysses_allgather_kv'] = self.config.use_ulysses_allgather_kv

        attn_para = dict()
        attn_para['packed_seq_params'] = packed_seq_params
        attn_para['attention_mask'] = attention_mask
        attn_para['scale'] = scale
        attn_para['pre_tokens'] = self.config.pre_tockens
        attn_para['next_tokens'] = self.config.next_tockens
        attn_para['keep_prob'] = 1 - self.attention_dropout.p
        attn_para['sparse_mode'] = sparse_mode
        attn_para['n_head'] = n_head
        output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)

        return output
    if tp_y_cp_sz > 1 and self.config.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                            'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
        in_hybrid_mode = False
        if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
            in_hybrid_mode = True

        if not in_hybrid_mode:
            if cp_expanded_by_2d_tp:
                tp_y_cp = TensorParallelYUnionCP()
                cp_group = tp_y_cp.group
                cp_size = tp_y_cp.get_parallel_group_world_size()
                rank = tp_y_cp.get_parallel_rank()
                cp_global_ranks = tp_y_cp.global_ranks
            else:
                cp_group = parallel_state.get_context_parallel_group()
                cp_size = parallel_state.get_context_parallel_world_size()
                rank = parallel_state.get_context_parallel_rank()
                cp_global_ranks = parallel_state.get_context_parallel_global_ranks()
        else:
            cp_group = get_context_parallel_group_for_hybrid_ring()
            cp_size = get_context_parallel_for_hybrid_ring_world_size()
            rank = get_context_parallel_for_hybrid_ring_rank()
            cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

        cp_para = dict()
        cp_para['megatron_cp_in_bnsd'] = self.config.megatron_cp_in_bnsd
        cp_para['causal'] = self.config.attention_mask_type == 'causal'
        cp_para['cp_group'] = cp_group
        cp_para['cp_size'] = cp_size
        cp_para['rank'] = rank

        if cp_para['causal']:
            attention_mask = torch.triu(torch.ones([2048, 2048], dtype=torch.bool, device=query.device), diagonal=1)

        if self.config.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            is_general_eod = ((getattr(self.config, 'attention_mask_type', None) == 'general') and (packed_seq_params is not None))
            if is_general_eod and not is_ulysses_algo:
                query, key, value = [rearrange(x, '(b s) n d -> s b (n d)', b=self.config.micro_batch_size) for x in [query, key, value]]
            cp_para['cp_global_ranks'] = cp_global_ranks
            if self.config.use_cp_send_recv_overlap:
                if cp_expanded_by_2d_tp:
                    cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
                else:
                    cp_para[
                        'cp_group_for_send_recv_overlap'] = parallel_state.get_context_parallel_group_for_send_recv_overlap()
            else:
                cp_para['cp_group_for_send_recv_overlap'] = None
            cp_para['pse'] = self.pse
            cp_para['pse_type'] = self.pse_type

            if self.config.context_parallel_size > 1 and not getattr(self.config, 'tp_2d', False):
                cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
                cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
                cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
                cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
                cp_para[
                    'cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
                cp_para['cache_policy'] = get_cache_policy(
                    self.layer_number, self.config.context_parallel_kv_cache_policy, self.config.context_parallel_cache_interval
                )
            output = ringattn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask,
                                                self.attention_dropout.p,
                                                packed_seq_params)
            if is_general_eod:
                output = rearrange(output, 's b (n d) -> (b s) n d', n=n_head)
        else:
            cp_para['scheduling_info'] = get_scheduling_info()
            output = adaptive_attn_context_parallel(query, key, value, n_head, cp_para, scale, attention_mask,
                                                    self.attention_dropout.p)

    else:
        # For EoD ulysses
        if packed_seq_params is not None:
            query, key, value = [rearrange(x, 's b (h d) -> (b s) h d', d=head_dim) for x in [query, key, value]]
            shape_order = 'TND'

        if self.config.use_fusion_attn_v2:
            output = npu_fusion_attention(
                query, key, value, n_head, shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pse_type=self.pse_type,
                pre_tokens=self.config.pre_tockens,
                next_tokens=self.config.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=sparse_mode,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
            )[0]
        else:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, shape_order,
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=self.config.pre_tockens,
                next_tockens=self.config.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=sparse_mode,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen
            )[0]
        if packed_seq_params is not None:
            output = rearrange(output, '(b s) h d -> s b (h d)', b=bsz)
            shape_order = 'TND'
    return output


mindspeed_args = get_mindspeed_args()
if getattr(mindspeed_args, 'context_parallel_algo') in ['megatron_cp_algo', 'hybrid_cp_algo'] and \
    int(getattr(mindspeed_args, 'context_parallel_size', 1)) > 1:

    pm.register_patch('mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel.AttentionWithCp.compute_mask',
                    vlm_cp_compute_mask, force_patch=True)
    pm.register_patch('mindspeed.core.context_parallel.dot_product_attention.CPDotProductAttentionImpl.forward',
                        vlm_cp_dot_product_attention_forward, force_patch=True)
    pm.apply_patches()
