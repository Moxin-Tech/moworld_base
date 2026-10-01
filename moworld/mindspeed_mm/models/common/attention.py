from typing import Tuple, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_npu
from einops import rearrange, repeat
from megatron import core
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from megatron.legacy.model.rms_norm import RMSNorm
from megatron.legacy.model.enums import AttnType
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ring_rank,
    get_context_parallel_for_hybrid_ring_global_ranks,
    get_ring_ranks_for_intra_window,
    get_ring_ranks_for_inter_window_kv,
    get_ring_ranks_for_inter_window_dkv,
    get_ring_group_for_intra_window,
    get_ring_group_for_intra_window_send_recv_overlap
)
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp import mapping

from mindspeed_mm.utils.utils import video_to_image, change_tensor_layout
from mindspeed_mm.models.common.normalize import normalize, FP32LayerNorm
from mindspeed_mm.models.common.embeddings.rope import RoPE3D, PositionGetter3D
from mindspeed_mm.models.common.conv import CausalConv3d, WfCausalConv3d
from mindspeed_mm.models.common.linear import MatmulAddLinear
from mindspeed_mm.utils.async_offload import async_save_on_cpu

# Replace with Megatron communication interface
from .communications import (
    all_to_all,
    all_to_all_SBH,
    split_forward_gather_backward,
)


def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None):
    args = get_args()
    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
        if args.use_cp_send_recv_overlap else None
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type

    cp_para['megatron_cp_in_bnsd'] = args.megatron_cp_in_bnsd

    output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p)

    return output


def do_npu_fusion_attention(
        q, k, v,
        head_num,
        softmax_scale,
        layout="BNSD",
        attn_mask=None,
        actual_seq_qlen=None,
        actual_seq_kvlen=None,
        dropout_p=0.,
        pse=None,
        sparse_mode=0,
        async_offload=False,
        block_idx=0,
        depth=0,
        h2d_stream=None,
        d2h_stream=None
    ):
    if async_offload:
        with async_save_on_cpu(
            h2d_stream=h2d_stream,
            d2h_stream=d2h_stream,
            block_idx=block_idx,
            depth=depth,
            custom_check_fn=lambda x: x.storage().size() >= q.storage().size()
        ):
            output = torch_npu.npu_fusion_attention(
                q, k, v, head_num, layout,
                pse=pse,
                padding_mask=None,
                atten_mask=attn_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                scale=softmax_scale,
                keep_prob=1 - dropout_p,
                inner_precise=0,
                sparse_mode=sparse_mode
            )[0]
    else:
        output = torch_npu.npu_fusion_attention(
            q, k, v, head_num, layout,
            pse=pse,
            padding_mask=None,
            atten_mask=attn_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            scale=softmax_scale,
            keep_prob=1 - dropout_p,
            inner_precise=0,
            sparse_mode=sparse_mode
        )[0]
    return output


class FlashAttention(nn.Module):
    """Implement the multihead softmax attention.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
        fa_layout: The input layout in Flash attention.
    """
    def __init__(
        self,
        softmax_scale=None,
        attention_dropout=0.0,
        fa_layout="sbh"
    ):
        super().__init__()
        args = get_args()
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout
        self.fa_layout = fa_layout
        self.pse = None
        self.pse_type = 1
        self.sparse_mode = args.sparse_mode
        self.context_parallel_algo = args.context_parallel_algo
        self.context_parallel_size = args.context_parallel_size

    def forward(self, query, key, value, attention_mask=None, **kwargs):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (S, B, N, D)
        """
        seq_length, batch_size, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]

        if attention_mask is not None and self.context_parallel_algo not in ["megatron_cp_algo", "hybrid_cp_algo"]:
            attention_mask = attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])

        if self.context_parallel_size > 1 and self.context_parallel_algo in ["megatron_cp_algo", "hybrid_cp_algo"]:
            if self.fa_layout.lower() != "sbh":
                raise ValueError(f"Flash attention layout mulst be `sbh` when using Ring Attention, but got {self.fa_layout}!")
            query, key, value = [rearrange(x, "s b n d -> s b (n d)") for x in [query, key, value]]
            output = do_ring_context_parallel(
                query,
                key,
                value,
                head_num=n_head,
                softmax_scale=self.softmax_scale,
                attn_mask=attention_mask,
                pse=self.pse,
                pse_type=self.pse_type
            )
        else:
            query, key, value = [change_tensor_layout(x, "sbnd", self.fa_layout, batch_size=batch_size) for x in [query, key, value]]

            actual_seq_qlen = []
            actual_seq_kvlen = []
            if self.fa_layout == "tnd":
                if attention_mask is not None:
                    ans = 0
                    for _ in range(batch_size):
                        ans += seq_length
                        actual_seq_qlen.append(ans)
                    ans = 0
                    for m in attention_mask:
                        ans += m
                        actual_seq_kvlen.append(ans)
            else:
                if attention_mask is not None:
                    attention_mask.view(batch_size, 1, -1, attention_mask.shape[-1])

            output = do_npu_fusion_attention(
                query, key, value,
                head_num=n_head,
                softmax_scale=self.softmax_scale,
                layout=self.fa_layout.upper(),
                attn_mask=attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                dropout_p=self.attention_dropout,
                sparse_mode=self.sparse_mode,
                **kwargs,
            )

        output = change_tensor_layout(output, self.fa_layout, "sbh", batch_size=batch_size)
        return output


class ParallelAttention(nn.Module):
    """
    A multi-head attention layer for both self-atten and cross-atten".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_attention_heads: The number of heads to use for multi-head attention.
        hidden_size: The hidden layer size.
        proj_q_bias: Whether to use bias in query projection.
        proj_k_bias: Whether to use bias in key projection.
        proj_v_bias: Whether to use bias in value projection.
        proj_out_bias: Whether to use bias in out projection.
        dropout: The dropout probability to use.
        use_qk_norm: Whether to use normalization for q and k.
        norm_type: The type of normalization layer.
        norm_elementwise_affine: A boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        norm_eps: A value added to the denominator for numerical stability. Default: 1e-5
        is_qkv_concat: Whether to concatenate qkv in projection.
        attention_type: Self attention or cross attention.
        is_kv_concat: Whether to concatenate kv in projection.
        fa_layout: The input layout of Flash Attention.
        rope: The Rotary Position Embedding object, default to `None`.
        split_kv_in_forward: Whether the input kv in the forward function is split or not.
            This argument is valid only in ["ulysses_cp_algo", "hybrid_cp_algo"].
    """
    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int],
        num_attention_heads: int,
        hidden_size: int,
        proj_q_bias: bool = False,
        proj_k_bias: bool = False,
        proj_v_bias: bool = False,
        proj_out_bias: bool = False,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        norm_type: str = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        is_qkv_concat: bool = False,
        attention_type: int = AttnType.self_attn,
        is_kv_concat: bool = False,
        fa_layout: str = "sbh",
        rope=None,
        split_kv_in_forward: bool = True,
        **kwargs
    ):
        super().__init__()

        args = get_args()
        config = core_transformer_config_from_args(args)
        key_dim = key_dim if key_dim is not None else query_dim
        self.sequence_parallel = args.sequence_parallel
        self.is_qkv_concat = is_qkv_concat
        self.is_kv_concat = is_kv_concat
        self.use_qk_norm = use_qk_norm
        self.attention_type = attention_type
        self.fa_layout = fa_layout
        self.rope = rope

        # Per attention head and per partition values.
        self.cp_size = mpu.get_context_parallel_world_size()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.head_dim = core.utils.divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(num_attention_heads, self.tp_size)
        self.attention_mask_type = args.attention_mask_type

        # Strided linear layer.
        if self.attention_type == AttnType.self_attn and self.is_qkv_concat:
            self.proj_qkv = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size * 3,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False)
        elif self.attention_type == AttnType.cross_attn and self.is_kv_concat:
            self.proj_q = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False)

            self.proj_kv = tensor_parallel.ColumnParallelLinear(
                key_dim,
                hidden_size * 2,
                config=config,
                init_method=config.init_method,
                bias=proj_k_bias,
                gather_output=False)
        else:
            self.proj_q = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False
            )
            self.proj_k = tensor_parallel.ColumnParallelLinear(
                key_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_k_bias,
                gather_output=False
            )
            self.proj_v = tensor_parallel.ColumnParallelLinear(
                key_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_v_bias,
                gather_output=False
            )

        # Normalize
        if self.use_qk_norm:
            self.q_norm = normalize(
                norm_type=norm_type,
                in_channels=self.head_dim,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs
            )
            self.k_norm = normalize(
                norm_type=norm_type,
                in_channels=self.head_dim,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs
            )
            if isinstance(self.q_norm, nn.LayerNorm):
                for param in self.q_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)
            if isinstance(self.k_norm, nn.LayerNorm):
                for param in self.k_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)

        # Flash Attention
        self.core_attention_flash = FlashAttention(
            attention_dropout=dropout,
            fa_layout=self.fa_layout,
            softmax_scale=1 / math.sqrt(self.head_dim)
        )
        if mpu.get_context_parallel_world_size() > 1 and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]:
            ulysses_group = mpu.get_context_parallel_group()
            if args.context_parallel_algo == "hybrid_cp_algo":
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            if split_kv_in_forward:
                self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, ulysses_group)

        # Output
        self.proj_out = tensor_parallel.RowParallelLinear(
            hidden_size,
            query_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_out_bias,
            input_is_parallel=True,
            skip_bias_add=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensor
        from `hidden_states` or `key_value_states`.
        """
        if self.attention_type == AttnType.self_attn and self.is_qkv_concat:
            # Attention heads [s, b, h] --> [s, b, 3*h]
            mixed_qkv = self.proj_qkv(hidden_states)[0]
            # [s, b, 3*h] --> [s, b, h], [s, b, h], [s, b, h]
            (query, key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_qkv, 3)
        elif self.attention_type == AttnType.cross_attn and self.is_kv_concat:
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            # Attention heads [s, b, h] --> [s, b, 2*h]
            mixed_kv = self.proj_kv(key_value_states)[0]
            # [s, b, 2*h] --> [s, b, h], [s, b, h]
            (key, value) = tensor_parallel.split_tensor_along_last_dim(mixed_kv, 2)
        else:
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            key = self.proj_k(key_value_states)[0]
            value = self.proj_v(key_value_states)[0]

        # [s, b, h] --> [s, b, n, d]
        batch_size = query.shape[1]
        query = query.view(-1, batch_size, self.num_attention_heads_per_partition, self.head_dim)
        key = key.view(-1, batch_size, self.num_attention_heads_per_partition, self.head_dim)
        value = value.view(-1, batch_size, self.num_attention_heads_per_partition, self.head_dim)

        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)

        return query, key, value

    def function_before_core_attention(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None
    ):
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # Reshape inputs into `sbh`
        query = change_tensor_layout(query, input_layout, "sbh")
        key = query if key is None else change_tensor_layout(key, input_layout, "sbh")

        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(query, key)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if self.rope is not None and rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query = self.rope.apply_rotary_pos_emb(query, q_pos_emb)
            key = self.rope.apply_rotary_pos_emb(key, k_pos_emb)

        return query, key, value

    def function_after_core_attention(
        self,
        core_attn_out,
        output_layout: str = "sbh"
    ):
        output, bias = self.proj_out(core_attn_out)
        # reshape
        output = change_tensor_layout(output, "sbh", output_layout)

        output = self.dropout(output)

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None
    ):
        # ==================================
        # Query, Key, and Value
        # ==================================
        query, key, value = self.function_before_core_attention(query, key, input_layout, rotary_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        core_attn_out = self.core_attention_flash(query, key, value, mask)

        # ==================================
        # output
        # ==================================
        out = self.function_after_core_attention(core_attn_out, input_layout)

        return out


class MultiHeadSparseAttentionSBH(ParallelAttention):
    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int],
        num_attention_heads: int,
        hidden_size: int,
        proj_q_bias: bool = False,
        proj_k_bias: bool = False,
        proj_v_bias: bool = False,
        proj_out_bias: bool = True,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        norm_type: str = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        is_qkv_concat: bool = False,
        attention_type: int = AttnType.self_attn,
        is_kv_concat: bool = False,
        fa_layout: str = "sbh",
        rope=None,
        sparse1d=False,
        sparse_n=None,
        sparse_group=None,
        **kwargs
    ):
        super().__init__(
            query_dim,
            key_dim,
            num_attention_heads,
            hidden_size,
            proj_q_bias,
            proj_k_bias,
            proj_v_bias,
            proj_out_bias,
            dropout,
            use_qk_norm,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            is_qkv_concat,
            attention_type,
            is_kv_concat,
            fa_layout,
            rope,
            **kwargs
        )
        args = get_args()
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group

        if args.context_parallel_algo == 'ulysses_cp_algo':
            self.num_attention_heads_per_partition_per_cp = core.utils.divide(self.num_attention_heads_per_partition, mpu.get_context_parallel_world_size())
        elif args.context_parallel_algo == 'hybrid_cp_algo':
            self.num_attention_heads_per_partition_per_cp = core.utils.divide(self.num_attention_heads_per_partition, args.ulysses_degree_in_cp)
        else:
            self.num_attention_heads_per_partition_per_cp = self.num_attention_heads_per_partition

        if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            in_hybrid_mode = False
            if get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None:
                in_hybrid_mode = True

            if not in_hybrid_mode:
                cp_group = mpu.get_context_parallel_group()
                cp_size = mpu.get_context_parallel_world_size()
                rank = mpu.get_context_parallel_rank()
                cp_global_ranks = mpu.get_context_parallel_global_ranks()
            else:
                cp_group = get_context_parallel_group_for_hybrid_ring()
                cp_size = get_context_parallel_for_hybrid_ring_world_size()
                rank = get_context_parallel_for_hybrid_ring_rank()
                cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()

            self.pse = None
            self.pse_type = 1

            self.cp_para = dict()
            self.cp_para['causal'] = args.attention_mask_type == 'causal'
            self.cp_para['cp_group'] = cp_group
            self.cp_para['cp_size'] = cp_size
            self.cp_para['rank'] = rank
            self.cp_para['cp_global_ranks'] = cp_global_ranks
            self.cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
                    if args.use_cp_send_recv_overlap else None
            self.cp_para['pse'] = self.pse
            self.cp_para['pse_type'] = self.pse_type
            self.cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
            self.cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
            self.cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
            self.cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
            self.cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        _len = x.shape[0]
        if _len != frame * height * width:
            raise ValueError("shape mismatched.")
        pad_len = 0
        if _len % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - _len % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        if not self.sparse_group:
            x = rearrange(x, '(g k) b d -> g (k b) d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n m k) b d -> (n k) (m b) d', m=self.sparse_n, k=self.sparse_n)
        return x, pad_len

    def _reverse_sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        if not self.sparse_group:
            x = rearrange(x, 'g (k b) d -> (g k) b d', k=self.sparse_n)
        else:
            x = rearrange(x, '(n k) (m b) d -> (n m k) b d', m=self.sparse_n, k=self.sparse_n)
        x = x[:frame * height * width, :, :]
        return x

    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.sparse_n)
        return x

    def function_before_core_attention(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        frames: int = 8,
        height: int = 16,
        width: int = 16,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):
        args = get_args()
        q, k, v = super().function_before_core_attention(query, key, input_layout, rotary_pos_emb=rotary_pos_emb)

        total_frames = frames
        if mpu.get_context_parallel_world_size() > 1:

            if args.context_parallel_algo == 'ulysses_cp_algo':
                cp_group = mpu.get_context_parallel_group()
                total_frames = frames * mpu.get_context_parallel_world_size()
                # apply all_to_all to gather sequence and split attention heads [s // sp, b, h, d] -> [s, b, h // sp, d]
                q = mapping.all_to_all(q, cp_group, scatter_dim=2, gather_dim=0)
                k = mapping.all_to_all(k, cp_group, scatter_dim=2, gather_dim=0)
                v = mapping.all_to_all(v, cp_group, scatter_dim=2, gather_dim=0)
            if args.context_parallel_algo == 'hybrid_cp_algo':
                cp_group = get_context_parallel_group_for_hybrid_ulysses()
                total_frames = frames * args.ulysses_degree_in_cp
                # apply all_to_all to gather sequence and split attention heads [s // sp, b, h, d] -> [s, b, h // sp, d]
                q = mapping.all_to_all(q, cp_group, scatter_dim=2, gather_dim=0)
                k = mapping.all_to_all(k, cp_group, scatter_dim=2, gather_dim=0)
                v = mapping.all_to_all(v, cp_group, scatter_dim=2, gather_dim=0)

        batch_size = q.shape[1]
        self.num_attention_heads_per_partition_per_cp = core.utils.divide(self.num_attention_heads_per_partition, mpu.get_context_parallel_world_size())
        q = q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        k = k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        v = v.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        pad_len = None
        if self.sparse1d:
            q, pad_len = self._sparse_1d(q, total_frames, height, width)
            if self.attention_type == AttnType.cross_attn:
                k = self._sparse_1d_kv(k)
                v = self._sparse_1d_kv(v)
            else:
                try:
                    k, pad_len = self._sparse_1d(k, total_frames, height, width)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    k, pad_len = None, None
                try:
                    v, pad_len = self._sparse_1d(v, total_frames, height, width)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    v, pad_len = None, None
        return q, k, v

    def function_after_core_attention(
        self,
        out,
        output_layout="sbh",
        frames=None,
        height=None,
        width=None,
        dtype=torch.bfloat16
    ):
        args = get_args()
        total_frames = frames
        if mpu.get_context_parallel_world_size() > 1:

            if args.context_parallel_algo == 'ulysses_cp_algo':
                cp_group = mpu.get_context_parallel_group()
                total_frames = frames * mpu.get_context_parallel_world_size()

            if args.context_parallel_algo == 'hybrid_cp_algo':
                cp_group = get_context_parallel_group_for_hybrid_ulysses()
                total_frames = frames * args.ulysses_degree_in_cp

        if self.sparse1d:
            out = self._reverse_sparse_1d(out, total_frames, height, width)

        if mpu.get_context_parallel_world_size() > 1:
            if args.context_parallel_algo == 'ulysses_cp_algo':
                cp_group = mpu.get_context_parallel_group()
                out = mapping.all_to_all(out, cp_group, scatter_dim=0, gather_dim=2)
            elif args.context_parallel_algo == 'hybrid_cp_algo':
                cp_group = get_context_parallel_group_for_hybrid_ulysses()
                out = mapping.all_to_all(out, cp_group, scatter_dim=0, gather_dim=2)
        out = out.to(dtype)
        out = super().function_after_core_attention(out, output_layout)

        return out

    def function_core_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        args = get_args()

        batch_size = query.shape[1]
        if mask is not None and args.context_parallel_algo not in ['megatron_cp_algo', 'hybrid_cp_algo']:
            mask = mask.view(batch_size, 1, -1, mask.shape[-1])

        if mpu.get_context_parallel_world_size() > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            scale = 1.0 / math.sqrt(self.head_dim)
            head_num = self.num_attention_heads_per_partition_per_cp
            cp_para = self.cp_para
            out = ringattn_context_parallel(query, key, value, head_num, cp_para, scale, mask)
        else:
            out = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                head_num=self.num_attention_heads_per_partition_per_cp,
                atten_mask=mask,
                input_layout="SBH",
                scale=1 / math.sqrt(self.head_dim)
            )[0]

        return out

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        frames: int = 8,
        height: int = 16,
        width: int = 16,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):

        # =====================
        # Query, Key, and Value
        # =====================
        q, k, v = self.function_before_core_attention(query, key, frames, height, width, input_layout, rotary_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        out = self.function_core_attention(q, k, v, mask)

        # =================
        # Output
        # =================
        out = self.function_after_core_attention(out, frames=frames, height=height, width=width, dtype=query.dtype)

        return out


class MultiHeadSparseMMAttentionSBH(MultiHeadSparseAttentionSBH):
    """
    A multi-head attention layer for both self-atten and cross-atten, layout "SBH", MMdit.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        added_kv_proj_dim: int = None,
        dropout: float = 0.0,
        proj_qkv_bias: bool = False,
        proj_out_bias: bool = True,
        sparse1d: bool = False,
        sparse_n: int = None,
        sparse_group: bool = None,
        is_cross_attn: bool = False,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        proj_q_bias = proj_k_bias = proj_v_bias = proj_qkv_bias
        super().__init__(
            query_dim=query_dim,
            key_dim=key_dim,
            num_attention_heads=num_heads,
            hidden_size=num_heads * head_dim,
            proj_q_bias=proj_q_bias,
            proj_k_bias=proj_k_bias,
            proj_v_bias=proj_v_bias,
            proj_out_bias=proj_out_bias,
            dropout=dropout,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            elementwise_affine=elementwise_affine,
        )

        self.head_dim = head_dim
        self.inner_dim = num_heads * self.head_dim

        if qk_norm is None:
            self.norm_proj_q = None
            self.norm_proj_k = None
        elif qk_norm == "fp32_layer_norm":
            self.norm_proj_q = FP32LayerNorm(head_dim, eps=eps, sequence_parallel=True)
            self.norm_proj_k = FP32LayerNorm(head_dim, eps=eps, sequence_parallel=True)
        elif qk_norm == "rms_norm":
            self.norm_proj_q = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
            self.norm_proj_k = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm}")

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_proj_q = FP32LayerNorm(head_dim, eps=eps, sequence_parallel=True)
                self.norm_added_proj_k = FP32LayerNorm(head_dim, eps=eps, sequence_parallel=True)
            elif qk_norm == "rms_norm":
                self.norm_added_proj_q = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
                self.norm_added_proj_k = RMSNorm(head_dim, eps=eps, sequence_parallel=True)
            else:
                raise ValueError(f"Unsupported qk_norm: {qk_norm}")

        args = get_args()
        config = core_transformer_config_from_args(args)

        self.context_pre_only = context_pre_only
        self.added_kv_proj_dim = added_kv_proj_dim

        if self.added_kv_proj_dim is not None:
            self.added_proj_k = tensor_parallel.ColumnParallelLinear(
                self.inner_dim,
                self.added_kv_proj_dim,
                config=config,
                init_method=config.init_method,
                bias=proj_k_bias,
                gather_output=False
            )
            self.added_proj_v = tensor_parallel.ColumnParallelLinear(
                self.inner_dim,
                self.added_kv_proj_dim,
                config=config,
                init_method=config.init_method,
                bias=proj_v_bias,
                gather_output=False
            )
            if self.context_pre_only is not None:
                self.added_proj_q = tensor_parallel.ColumnParallelLinear(
                    self.inner_dim,
                    self.added_kv_proj_dim,
                    config=config,
                    init_method=config.init_method,
                    bias=proj_q_bias,
                    gather_output=False
                )

        if not self.context_pre_only:
            self.added_proj_out = tensor_parallel.RowParallelLinear(
                added_kv_proj_dim,
                self.inner_dim,
                config=config,
                init_method=config.init_method,
                bias=proj_out_bias,
                input_is_parallel=True,
                skip_bias_add=False
            )

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, cos, sin):
        """
            * tokens: ntokens x batch_size x nheads x dim
        """
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def apply_rotary_emb(self, tokens, video_rotary_emb):
        cos_t, sin_t, cos_y, sin_y, cos_x, sin_x = video_rotary_emb
        # split features into three along the feature dimension, and apply rope1d on each half
        dim = tokens.shape[-1]
        D_t = dim // 16 * 4
        D = dim // 16 * 6
        origin_dtype = tokens.dtype
        t, y, x = torch.split(tokens, [D_t, D, D], dim=-1)
        t = self.apply_rope1d(t, cos_t, sin_t)
        y = self.apply_rope1d(y, cos_y, sin_y)
        x = self.apply_rope1d(x, cos_x, sin_x)
        tokens = torch.cat((t, y, x), dim=-1).to(origin_dtype)
        return tokens

    def _reverse_sparse_1d_enc(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = rearrange(x, 's (k b) d -> s k b d', k=self.sparse_n).mean(1)
        return x

    def _sparse_1d_enc(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        x = repeat(x, 's b d -> s (k b) d', k=self.sparse_n)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        attention_mask: Optional[torch.Tensor] = None,
        video_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: The hidden states of the visual stream.
            encoder_hidden_states: The hidden states of the textual stream.
            frames: The frame number of video
            height: The height of the video
            width: The width of the video
            attention_mask: The attention mask to use.
            video_rotary_emb: The rotary embeddings for the video
        """

        # Step 1: Project the hidden states and encoder hidden states
        q, _ = self.proj_q(hidden_states)
        k, _ = self.proj_k(hidden_states)
        v, _ = self.proj_v(hidden_states)
        added_q, _ = self.added_proj_q(encoder_hidden_states)
        added_k, _ = self.added_proj_k(encoder_hidden_states)
        added_v, _ = self.added_proj_v(encoder_hidden_states)

        batch_size = q.shape[1]
        batch_size = added_q.shape[1]

        total_frames = frames


        # Step 2: QK Norm
        q = q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        k = k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        q = self.norm_proj_q(q)
        k = self.norm_proj_k(k)

        added_q = added_q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        added_k = added_k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        added_q = self.norm_added_proj_q(added_q)
        added_k = self.norm_added_proj_k(added_k)

        # Step 3: Apply rope
        q = self.apply_rotary_emb(q, video_rotary_emb)
        k = self.apply_rotary_emb(k, video_rotary_emb)

        q = q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        k = k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        v = v.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)

        added_q = added_q.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        added_k = added_k.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        added_v = added_v.view(-1, batch_size, self.num_attention_heads_per_partition_per_cp * self.head_dim)

        # Step 4: Sparse 1D
        if self.sparse1d:
            q, _ = self._sparse_1d(q, total_frames, height, width)
            k, _ = self._sparse_1d(k, total_frames, height, width)
            v, _ = self._sparse_1d(v, total_frames, height, width)
            added_q = self._sparse_1d_enc(added_q)
            added_k = self._sparse_1d_enc(added_k)
            added_v = self._sparse_1d_enc(added_v)

        # Step 5: Concat hidden_states and encoder_hidden_states to do mm attention
        fa_visual_sequence_length = q.shape[0]
        fa_text_sequence_length_length = added_q.shape[0]
        q = torch.cat([q, added_q], dim=0)
        k = torch.cat([k, added_k], dim=0)
        v = torch.cat([v, added_v], dim=0)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_attention_heads_per_partition_per_cp,
            atten_mask=attention_mask,
            input_layout="SBH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        hidden_states, encoder_hidden_states = out.split([fa_visual_sequence_length, fa_text_sequence_length_length], dim=0)

        # Step 6: Reverse sparse 1D
        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frames, height, width)
            encoder_hidden_states = self._reverse_sparse_1d_enc(encoder_hidden_states)


        # Step 7: Project out
        hidden_states, _ = self.proj_out(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if not self.context_pre_only:
            encoder_hidden_states, _ = self.added_proj_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class ParallelMultiHeadAttentionSBH(nn.Module):
    """
    A multi-head context parallel attention layer for both self-attention and cross-attention, layout "SBH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        use_rope: Whether to use rope
        interpolation_scale: The scale of interpolation.

    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        proj_qkv_bias: bool = False,
        proj_out_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[int] = (1, 1, 1)
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE3D(interpolation_scale=interpolation_scale)
            self.position_getter = PositionGetter3D(atten_layout="SBH")

        args = get_args()
        config = core_transformer_config_from_args(args)
        self.sp_size = mpu.get_context_parallel_world_size()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()

        self.num_attention_heads_per_partition = core.utils.divide(num_heads, self.tp_size)
        self.num_attention_heads_per_partition_per_cp = core.utils.divide(self.num_attention_heads_per_partition,
                                                                          self.sp_size)

        key_dim = key_dim if key_dim is not None else query_dim

        self.proj_q = tensor_parallel.ColumnParallelLinear(
            query_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )
        self.proj_k = tensor_parallel.ColumnParallelLinear(
            key_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )
        self.proj_v = tensor_parallel.ColumnParallelLinear(
            key_dim,
            self.inner_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_qkv_bias,
            gather_output=False
        )

        self.proj_out = tensor_parallel.RowParallelLinear(
            self.inner_dim,
            query_dim,
            config=config,
            init_method=config.init_method,
            bias=proj_out_bias,
            input_is_parallel=True,
            skip_bias_add=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            query: The hidden states of the query.
            key: The hidden states of the key.
            mask: The attention mask to use.
            frames: The frame number of latents
            height: The height of the frame
            width: The width of the frame
            **kwargs: Additional keyword arguments to pass along
        """
        if len(query.shape) != 3:
            raise AssertionError("Parallel attention only support SBH.")

        is_cross_attention = key is not None

        key = query if key is None else key
        s, b, _ = query.shape

        if mask is not None:
            mask = mask.view(b, 1, -1, mask.shape[-1])

        q, _ = self.proj_q(query)
        k, _ = self.proj_k(key)
        v, _ = self.proj_v(key)

        q = q.view(-1, self.num_attention_heads_per_partition, self.head_dim)
        k = k.view(-1, self.num_attention_heads_per_partition, self.head_dim)
        v = v.view(-1, self.num_attention_heads_per_partition, self.head_dim)
        sp_group = mpu.get_context_parallel_group()

        q = all_to_all_SBH(q, sp_group, scatter_dim=1, gather_dim=0)
        if not is_cross_attention:
            k = all_to_all_SBH(k, sp_group, scatter_dim=1, gather_dim=0)
            v = all_to_all_SBH(v, sp_group, scatter_dim=1, gather_dim=0)
        else:
            k = split_forward_gather_backward(k, sp_group, dim=1, grad_scale="down")
            v = split_forward_gather_backward(v, sp_group, dim=1, grad_scale="down")

        if self.use_rope:
            #  Official repo bug: incorrect use of view, cross-axis view is invalid
            q = q.view(-1, b, self.num_attention_heads_per_partition_per_cp, self.head_dim)
            k = k.view(-1, b, self.num_attention_heads_per_partition_per_cp, self.head_dim)

            if (frames is None) or (height is None) or (width is None):
                raise ValueError("frames, height and width can not be none when use_rope")
            pos_thw = self.position_getter(b, t=frames, h=height, w=width, device=query.device)
            q = self.rope(q, pos_thw)
            k = self.rope(k, pos_thw)

        q = q.view(-1, b, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        k = k.view(-1, b, self.num_attention_heads_per_partition_per_cp * self.head_dim)
        v = v.view(-1, b, self.num_attention_heads_per_partition_per_cp * self.head_dim)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_attention_heads_per_partition_per_cp,
            atten_mask=mask,
            input_layout="SBH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        out = out.view(-1, self.num_attention_heads_per_partition_per_cp, self.head_dim)
        out = all_to_all_SBH(out, sp_group, scatter_dim=0, gather_dim=1)
        out = out.view(-1, b, self.num_attention_heads_per_partition * self.head_dim)

        out, _ = self.proj_out(out)
        out = self.dropout(out)

        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise AssertionError(
                "dim (%d) must be divisible by num_heads (%d)" % (dim, num_heads)
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def npu_spatial_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _ = qkv.shape
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)

        x = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, input_layout="BSND",
            pse=None,
            scale=self.scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1. - self.attn_drop.p if self.training else 1.,
            sync=False,
            inner_precise=0
        )[0]

        x = x.view(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def npu_temporal_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _ = qkv.shape
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm_legacy:
            q, k = self.rotary_emb(q), self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            q, k = self.rotary_emb(q), self.rotary_emb(k)

        x = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, input_layout="BNSD",
            pse=None,
            scale=self.scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1. - self.attn_drop.p if self.training else 1.,
            sync=False,
            inner_precise=0,
        )[0]

        x = x.transpose(1, 2)
        x = x.reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flashattn = self.enable_flashattn
        qkv = self.qkv(x)

        if enable_flashattn:
            if qkv.dtype in [torch.float16, torch.bfloat16]:
                if self.rope:
                    return self.npu_temporal_attention(qkv)
                else:
                    return self.npu_spatial_attention(qkv)
            else:
                raise ValueError("The dtype of x must be torch.float16 or torch.bfloat16, got torch.float32 instead.")

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        attn = self.attn_drop(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flashattn=enable_flashattn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = (
            x.shape
        )  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape)


        sp_group = mpu.get_context_parallel_group()

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flashattn:
            # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.permute(qkv_permute_shape)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn and q.dtype in [torch.float16, torch.bfloat16]:
            x = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                q.shape[-2],
                input_layout="BSND",
                pse=None,
                scale=self.scale,
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0 - self.attn_drop.p if self.training else 1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flashattn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise AssertionError(
                "d_model (%d) must be divisible by num_heads (%d)"
                % (d_model, num_heads)
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = self.q_linear(x).view(-1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            self.num_heads,
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = mpu.get_context_parallel_group()
        sp_size = mpu.get_context_parallel_world_size()
        B, SUB_N, C = x.shape
        N = SUB_N * sp_size

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)

        k = split_forward_gather_backward(
            k, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )
        v = split_forward_gather_backward(
            v, mpu.get_context_parallel_group(), dim=2, grad_scale="down"
        )

        if x.dtype not in [torch.float16, torch.bfloat16]:
            raise AssertionError("QKV's dtype must be in bf16 or fp16")
        q = q.view(-1, self.num_heads // sp_size, self.head_dim)
        k = k.view(-1, self.num_heads // sp_size, self.head_dim)
        v = v.view(-1, self.num_heads // sp_size, self.head_dim)

        actual_seq_qlen = []
        actual_seq_kvlen = []
        if mask is not None:
            ans = 0
            for _ in range(B):
                ans += N
                actual_seq_qlen.append(ans)
            ans = 0
            for m in mask:
                ans += m
                actual_seq_kvlen.append(ans)
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            q.shape[-2],
            input_layout="TND",
            pse=None,
            scale=1.0 / math.sqrt(self.head_dim),
            pre_tockens=65536,
            next_tockens=65536,
            actual_seq_qlen=tuple(actual_seq_qlen),
            actual_seq_kvlen=tuple(actual_seq_kvlen),
            keep_prob=1.0 - self.attn_drop.p,
            sparse_mode=0,
        )[0]

        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Conv2dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_groups=32,
        eps=1e-6,
        kernel_size=1,
        stride=1,
        padding=0,
        affine=True,
        use_sdp_attention=False,
    ):
        super().__init__()
        self.use_sdp_attention = use_sdp_attention
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
        )
        self.q = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.k = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.v = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def attention_sdp(self, h_):
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def attention_manual(self, x):
        y = x
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # [b, hw, c]
        k = k.reshape(b, c, h * w)  # [b, c, hw]
        z = torch.bmm(q, k)  # [b, hw, hw]
        z = z * (int(c) ** (-0.5))
        z = torch.nn.functional.softmax(z, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        z = z.permute(0, 2, 1)  # [b, hw, hw] (first hw of k, second of q)
        y = torch.bmm(v, z)  # [b, c, hw] (hw of q)
        y = y.reshape(b, c, h, w)

        return y

    @video_to_image
    def forward(self, x):
        if self.use_sdp_attention:
            attention_output = self.attention_sdp(x)
        else:
            attention_output = self.attention_manual(x)

        return x + self.proj_out(attention_output)


class CausalConv3dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine
        )
        self.q = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.k = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.v = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.proj_out = CausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        y = x
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        # compute attention
        # q: (b c t h w) -> (b t c h w) -> (b*t c h*w) -> (b*t h*w c)
        b, c, t, h, w = q.shape
        q = torch_npu.npu_confusion_transpose(
            q, (0, 2, 1, 3, 4), (b * t, c, h * w), True
        )
        q = q.permute(0, 2, 1)

        # k: (b c t h w) -> (b t c h w) -> (b*t c h*w)
        k = torch_npu.npu_confusion_transpose(
            k, (0, 2, 1, 3, 4), (b * t, c, h * w), True
        )

        # w: (b*t hw hw)
        z = torch.bmm(q, k)
        z = z * (int(c) ** (-0.5))
        z = torch.nn.functional.softmax(z, dim=2)

        # attend to values
        # v: (b c t h w) -> (b t c h w) -> (bt c hw)
        # z: (bt hw hw) -> (bt hw hw)
        v = torch_npu.npu_confusion_transpose(v, (0, 2, 1, 3, 4), (b * t, c, h * w), True)
        z = z.permute(0, 2, 1)  # [b, hw, hw] (first hw of k, second of q)
        y = torch.bmm(v, z)  # [b, c, hw] (hw of q)

        # y: (b*t c hw) -> (b t c h w) -> (b c t h w)
        y = torch_npu.npu_confusion_transpose(y, (0, 2, 1, 3, 4), (b, t, c, h, w), False)

        y = self.proj_out(y)

        return x + y


@dataclass
class AttentionParams:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    input_layout: str
    head_num: int = None
    atten_mask: torch.Tensor = None


class WfCausalConv3dAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.norm = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.q = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.k = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.v = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.proj_out = WfCausalConv3d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, t, h, w = q.shape
        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        k = k.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
        v = v.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()

        params = AttentionParams(
            query=q,
            key=k,
            value=v,
            input_layout="BSH",
            head_num=1,
        )

        attn_output = self.run_attention(params, head_dim=c, enable_FA=c <= 512)

        attn_output = attn_output.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        h_ = self.proj_out(attn_output)

        return x + h_

    def run_attention(self, params: AttentionParams, head_dim, enable_FA=True):
        if enable_FA:
            hidden_states = torch_npu.npu_fusion_attention(params.query, params.key, params.value,
                                                           head_num=params.head_num,
                                                           atten_mask=None,
                                                           input_layout=params.input_layout,
                                                           scale=1 / math.sqrt(head_dim))[0]
        else:
            hidden_states = self.scaled_dot_product_attention(params,
                                                              scale=1 / math.sqrt(head_dim))
        return hidden_states

    def scaled_dot_product_attention(self, params: AttentionParams, scale=None, dropout_p=0.0, is_causal=False) -> torch.Tensor:
        def trans_tensor_shape(x, layout, head_num):
            if layout == "BSH":
                batch = x.shape[0]
                x = x.view(batch, -1, head_num, x.shape[-1] // head_num).transpose(1, 2).contiguous()
            elif layout == "SBH":
                batch = x.shape[1]
                x = x.view(-1, batch * head_num, x.shape[-1] // head_num).transpose(0, 1).contiguous()
                x = x.view(batch, head_num, -1, x.shape[-1])
            return x

        query = trans_tensor_shape(params.query, params.input_layout, params.head_num)
        key = trans_tensor_shape(params.key, params.input_layout, params.head_num)
        value = trans_tensor_shape(params.value, params. input_layout, params.head_num)

        attn_weight = query @ key.transpose(-2, -1) * scale
        attn_bias = torch.zeros_like(attn_weight, dtype=query.dtype, device=query.device)
        if is_causal:
            if params.atten_mask is not None:
                raise ValueError("atten_mask should be None when is_causal is True")
            temp_mask = torch.zeros_like(attn_weight, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), -10000.0)
            attn_bias.to(query.dtype)

        if params.atten_mask is not None and self.enable_FA and params.atten_mask.dtype == torch.bool:
            raise ValueError("attention_mask must not be bool type when use this function")

        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value
        if params.input_layout == "BSH":
            output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, params.head_num * output.shape[-1])
        else:
            output = output.view(output.shape[0] * params.head_num, -1, output.shape[-1]).transpose(0, 1).contiguous()
            output = output.view(output.shape[0], -1, params.head_num * output.shape[-1])
        return output


class WhisperAttention(nn.Module):
    """
    A multi-head attention layer for both self-atten and cross-atten, layout "BSH".

    Args:
        query_dim: The number of channels in the query.
        key_dim: The number of channels in the key, defaults to `query_dim`.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        dropout: The dropout probability to use.
        proj_qkv_bias: Whether to use bias in qkv projection.
        proj_out_bias: Whether to use bias in out projection.
        use_rope: Whether to use rope
        interpolation_scale: The scale of interpolation.

    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        proj_qv_bias: bool = False,
        proj_out_bias: bool = True,
        interpolation_scale: Tuple[int] = (1, 1, 1),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = self.num_heads * self.head_dim

        key_dim = key_dim if key_dim is not None else query_dim

        self.proj_q = nn.Linear(query_dim, self.inner_dim, bias=proj_qv_bias)
        self.proj_k = nn.Linear(key_dim, self.inner_dim, bias=False)
        self.proj_v = nn.Linear(key_dim, self.inner_dim, bias=proj_qv_bias)

        self.proj_out = nn.Linear(self.inner_dim, query_dim, bias=proj_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: The hidden states of the query.
            key: The hidden states of the key.
            mask: The attention mask to use.
            **kwargs: Additional keyword arguments to pass along
        """
        input_ndim = query.ndim
        if input_ndim == 4:
            b, c, h, w = query.shape
            query = query.view(b, c, h * w).transpose(1, 2)

        key = query if key is None else key
        b, _, _ = query.shape

        if mask is not None:
            mask = mask.view(b, 1, -1, mask.shape[-1])

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(key)

        q = q.view(b, -1, self.inner_dim)
        k = k.view(b, -1, self.inner_dim)

        out = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=self.num_heads,
            atten_mask=mask,
            input_layout="BSH",
            scale=1 / math.sqrt(self.head_dim)
        )[0]

        out = self.proj_out(out)
        out = self.dropout(out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        return out
