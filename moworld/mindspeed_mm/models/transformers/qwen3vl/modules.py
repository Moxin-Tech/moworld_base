# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

from collections.abc import Callable
from typing import Optional

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from einops import rearrange

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from megatron.core import mpu

from megatron.training import get_args
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ulysses_world_size,
    get_context_parallel_for_hybrid_ring_global_ranks,
    get_context_parallel_for_hybrid_ring_rank
)
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel_tnd_general, ringattn_context_parallel
from mindspeed.utils import get_actual_seq_len

from mindspeed_mm.models.common.communications import cal_split_sizes, cal_split_sizes_multi, split_forward_gather_backward
from mindspeed_mm.utils.utils import get_packed_seq_params, get_packed_seq_len
from ..cp_utils import get_seq_len, gather_seq_scatter_heads_qkv, gather_heads_scatter_seq, gather_visual_seqs_with_cp
from ..attention_utils import ALL_ATTENTION_FUNCTIONS, pad_out


class Qwen3VLEmptyModule(nn.Module):
    """
    This class does not implement any functionality. It serves solely as a placeholder
    to provide a registration point for attaching FSDP2 hooks to all normalization (e.g., LayerNorm, RMSNorm)
    and gate-related parameters when the `align_fsdp_param_groups` feature is enabled.

    Its purpose is structural: to ensure these specific parameters are correctly identified
    and included in FSDP2's parameter grouping and communication logic, without participating
    in forward/backward computation or maintaining any internal state.
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.spatial_merge_size = config.spatial_merge_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if mpu.get_context_parallel_world_size() > 1:
            if self.use_postshuffle_norm:
                x = gather_visual_seqs_with_cp(x, dim=0)
                x = x.view(-1, self.hidden_size)
                # after down_sample hidden_state, should split it again
                split_sizes = cal_split_sizes(x.shape[0], mpu.get_context_parallel_world_size())
                # Split the merged tensor back for distributed processing
                # Since no attention computation follows, we can use simple splitting
                x = split_forward_gather_backward(x, mpu.get_context_parallel_group(), dim=0, grad_scale="down", split_sizes=split_sizes)
                x = self.norm(x)
            else:
                x = self.norm(x)
                x = gather_visual_seqs_with_cp(x, dim=0)
                x = x.view(-1, self.hidden_size)
                # after down_sample hidden_state, should split it again
                split_sizes = cal_split_sizes(x.shape[0], mpu.get_context_parallel_world_size())
                # Split the merged tensor back for distributed processing
                # Since no attention computation follows, we can use simple splitting
                x = split_forward_gather_backward(x, mpu.get_context_parallel_group(), dim=0, grad_scale="down", split_sizes=split_sizes)
            x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        else:
            x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
            x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed = q_embed.squeeze(0)
    k_embed = k_embed.squeeze(0)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int, layout: str) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). Adapt to different attention layouts:
    insert expansion dim after num_key_value_heads, merge to num_attention_heads, keep other dims unchanged.
    """
    if n_rep == 1:
        return hidden_states
    if layout == "BNSD":
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    elif layout == "BSND":
        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)
    elif layout == "TND":
        token, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :].expand(token, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(token, num_key_value_heads * n_rep, head_dim)
    else:
        raise NotImplementedError(
            f"Unsupported Attention layout: {layout}, "
            "Qwen3OmniMoeThinkerTextAttention only support ['BNSD', 'BSND', 'TND'] now.")


def do_vit_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask=None, dropout_p=0., pse=None, pse_type=None, shapes=None):
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

    cp_para['causal'] = False
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
        if args.use_cp_send_recv_overlap else None
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type

    output = ringattn_context_parallel_tnd_general(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=shapes)

    return output


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        total_visual_seqlen = int(cu_seqlens[-1])
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)  # TND

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        layout = self.config.attn_layout.upper()
        seq_dim, head_dim = None, None
        attention_kwargs = {"scale": self.scaling, "dropout": self.attention_dropout, "is_causal": self.is_causal, "attention_mask": None}

        if self.config._attn_implementation == "flash_attention_2" and layout == "TND":
            seq_dim, head_dim = 0, 1
            attention_kwargs["actual_seq_qlen"] = cu_seqlens
            attention_kwargs["actual_seq_kvlen"] = cu_seqlens
            attention_kwargs["layout"] = "TND"

        elif self.config._attn_implementation in ["eager", "sdpa", "flash_attention_2"] and layout == "BNSD":
            # layout, TND --> BNSD
            query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1, N, T(B*S), D]
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            seq_dim, head_dim = 2, 1
            attention_kwargs["layout"] = "BNSD"
        else:
            raise NotImplementedError(
                f"Unsupported Attention: {self.config._attn_implementation}, or layout: {layout}"
                "Qwen3VLTextAttention only support ['eager', 'sdpa', 'flash_attention_2'], layout TND and BNSD")

        if mpu.get_context_parallel_world_size() > 1:
            megatron_args = get_args()
            if megatron_args.context_parallel_algo == "ulysses_cp_algo":
                query_states, key_states, value_states = gather_seq_scatter_heads_qkv(
                    query_states,
                    key_states,
                    value_states,
                    seq_dim=seq_dim,
                    head_dim=head_dim,
                    gather_size=total_visual_seqlen
                )
            elif megatron_args.context_parallel_algo == "megatron_cp_algo":
                if layout != "TND":
                    raise ValueError(f"Vision Attention only support layout `TND` when using Ring Attention.")
                all_split_sizes_tensor = cal_split_sizes_multi(get_seq_len("per_visual"), mpu.get_context_parallel_world_size())
                attn_output = do_vit_ring_context_parallel(
                    query_states,
                    key_states,
                    value_states,
                    self.num_heads,
                    self.scaling,
                    attn_mask=None,
                    dropout_p=0.,
                    pse=None,
                    pse_type=None,
                    shapes=all_split_sizes_tensor
                )
                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                attn_output = self.proj(attn_output)
                return attn_output
            elif megatron_args.context_parallel_algo == "hybrid_cp_algo":
                if layout != "TND":
                    raise ValueError(f"Vision Attention only support layout `TND` when using Hybrid Attention.")
                # ulysses a2a
                ulysses_process_group = get_context_parallel_group_for_hybrid_ulysses()
                query_states, key_states, value_states = gather_seq_scatter_heads_qkv(query_states, key_states, value_states, seq_dim=0, head_dim=1, gather_size=total_visual_seqlen, group=ulysses_process_group)
                # ring attention
                all_split_sizes_tensor = cal_split_sizes_multi(get_seq_len("per_visual"), get_context_parallel_for_hybrid_ring_world_size())
                attn_output = do_vit_ring_context_parallel(
                    query_states,
                    key_states,
                    value_states,
                    self.num_heads // get_context_parallel_for_hybrid_ulysses_world_size(),  # Num of heads per ring rank
                    self.scaling,
                    attn_mask=None,
                    dropout_p=0.,
                    pse=None,
                    pse_type=None,
                    shapes=all_split_sizes_tensor
                )
                attn_output = gather_heads_scatter_seq(attn_output, seq_dim=0, head_dim=1, gather_size=self.num_heads, group=get_context_parallel_group_for_hybrid_ulysses())
                attn_output = attn_output.reshape(seq_length, -1).contiguous()
                attn_output = self.proj(attn_output)
                return attn_output
            else:
                raise NotImplementedError(f"Only support `ulysses_cp_algo`,`megatron_cp_algo`,`hybrid_cp_algo`, but got {megatron_args.context_parallel_algo}")


        if layout == "TND":

            attn_output = attention_interface(
                query_states,
                key_states,
                value_states,
                **attention_kwargs
            )
        else:
            # Other FA implementations: Process each chunk separately
            lengths = [cu_seqlens[0]] + [post_len - seqlen for seqlen, post_len in zip(cu_seqlens, cu_seqlens[1:])]
            splits = [
                torch.split(tensor, lengths, dim=seq_dim)
                for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    q,
                    k,
                    v,
                    **attention_kwargs,
                )
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=seq_dim)

        if mpu.get_context_parallel_world_size() > 1:
            attn_output = gather_heads_scatter_seq(
                attn_output,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=self.num_heads
            )

        if layout == "BNSD":
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)

        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.config.synchronize_per_layer:
            torch.npu.synchronize()

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if hasattr(self.config, "rope_parameters"):
            self.rope_type = self.config.rope_parameters["rope_type"]
        elif hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
            self.rope_type = self.config.rope_scaling["rope_type"]
        else:
            self.rope_type = "default"
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Qwen3VLTextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        if hasattr(config, "rope_parameters"):
            base = config.rope_parameters["rope_theta"]
        else:
            base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def do_llm_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask=None, dropout_p=0., pse=None, pse_type=None, shapes=None, packed_seq_params=None, layout="SBH"):
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

    cp_para['causal'] = True
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
        if args.use_cp_send_recv_overlap else None
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type

    cp_para['megatron_cp_in_bnsd'] = args.megatron_cp_in_bnsd

    if layout == "TND":
        actual_seq_len = get_actual_seq_len()
        packed_seq_params, shapes = get_packed_seq_params(actual_seq_len, cp_size=cp_size)
        attn_output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, packed_seq_params=packed_seq_params, shapes=shapes)

        return attn_output
    elif layout == "BNSD":
        # mindspeed core only support SBH as input layout
        D = q.shape[-1]
        q = rearrange(q, "b s n d -> s b (n d)").contiguous()
        k = rearrange(k, "b s n d -> s b (n d)").contiguous()
        v = rearrange(v, "b s n d -> s b (n d)").contiguous()
        attn_output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, packed_seq_params=None, shapes=shapes)
        # Convert back from SBH to BNSD format
        attn_output = rearrange(attn_output, "s b (n d) -> b s n d", d=D).contiguous()
        return attn_output
    else:
        raise NotImplementedError("LLM Ring CP only support TND and BNSD now")


class Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    def forward(
        self,
        hidden_states: torch.Tensor, # [B, S, H]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, seqlen = hidden_states.shape[:-1]
        hidden_shape = (batch_size, seqlen, -1, self.head_dim)  # BSND

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))  # BSND
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings # b s d
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        layout = self.config.attn_layout.upper()
        dropout = 0.0 if not self.training else self.attention_dropout
        attention_kwargs = {
            "scale": self.scaling,
            "dropout": dropout,
            "is_causal": self.is_causal,
            "layout": layout,
            "enable_gqa": True
        }

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        total_seq_len = get_seq_len("total")
        if mpu.get_context_parallel_world_size() > 1:
            megatron_args = get_args()
            seq_dim, head_dim = 1, 2

            if megatron_args.context_parallel_algo == "ulysses_cp_algo":
                if mpu.get_context_parallel_world_size() > self.config.num_key_value_heads:
                    key_states = repeat_kv(key_states, self.num_key_value_groups, "BSND")
                    value_states = repeat_kv(value_states, self.num_key_value_groups, "BSND")
                    attention_kwargs["enable_gqa"] = False
                query_states, key_states, value_states = gather_seq_scatter_heads_qkv(
                    query_states,
                    key_states,
                    value_states,
                    seq_dim=seq_dim,
                    head_dim=head_dim,
                    gather_size=total_seq_len
                )
            elif megatron_args.context_parallel_algo == "megatron_cp_algo":
                if layout not in ["BNSD", "TND"]:
                    raise ValueError(f"TextAttention only support layout `BNSD` and `TND` when using Ring Attention.")

                if layout == "TND":
                    query_states = query_states.view(-1, *query_states.shape[2:])
                    key_states = key_states.view(-1, *key_states.shape[2:])
                    value_states = value_states.view(-1, *value_states.shape[2:])

                attn_output = do_llm_ring_context_parallel(
                    query_states,
                    key_states,
                    value_states,
                    self.config.num_attention_heads,
                    softmax_scale=self.scaling,
                    attn_mask=None,
                    dropout_p=0.,
                    pse=None,
                    pse_type=None,
                    shapes=None,  # LLM inputs are padded to be divisible by 2*cp_size
                    layout=layout
                )
                attn_output = attn_output.reshape(batch_size, seqlen, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output
            elif megatron_args.context_parallel_algo == "hybrid_cp_algo":
                if layout not in ["BNSD", "TND"]:
                    raise ValueError(f"TextAttention only support layout `BNSD` and `TND` when using Ring Attention.")

                if layout == "TND" or get_context_parallel_for_hybrid_ulysses_world_size() > self.config.num_key_value_heads:
                    key_states = repeat_kv(key_states, self.num_key_value_groups, layout="BSND")
                    value_states = repeat_kv(value_states, self.num_key_value_groups, layout="BSND")

                # Calculate sequence length per ring group, Division is exact due to padding in ring groups
                actual_seq_len = get_actual_seq_len()
                if actual_seq_len is not None:
                    total_seq_len = get_packed_seq_len(actual_seq_len, get_context_parallel_for_hybrid_ring_world_size())
                else:
                    total_seq_len = get_seq_len("total")
                seq_len_per_ring = total_seq_len // get_context_parallel_for_hybrid_ring_world_size()

                # ulysses a2a
                query_states, key_states, value_states = gather_seq_scatter_heads_qkv(
                    query_states,
                    key_states,
                    value_states,
                    seq_dim=seq_dim,
                    head_dim=head_dim,
                    gather_size=seq_len_per_ring,
                    group=get_context_parallel_group_for_hybrid_ulysses()
                )

                # ring attention
                if layout == "TND":
                    query_states = query_states.view(-1, *query_states.shape[2:])
                    key_states = key_states.view(-1, *key_states.shape[2:])
                    value_states = value_states.view(-1, *value_states.shape[2:])

                attn_output = do_llm_ring_context_parallel(
                    query_states,
                    key_states,
                    value_states,
                    self.config.num_attention_heads // get_context_parallel_for_hybrid_ulysses_world_size(),  # Num of heads per ring rank
                    softmax_scale=self.scaling,
                    attn_mask=None,
                    dropout_p=0.,
                    pse=None,
                    pse_type=None,
                    shapes=None,  # LLM inputs are padded to be divisible by 2*cp_size
                    layout=layout
                )
                if layout == "TND":
                    attn_output = rearrange(attn_output, '(b s) n d -> b s n d', b=batch_size).contiguous()
                else:
                    attn_output = rearrange(attn_output, "s b (n d) -> b s n d", d=self.head_dim).contiguous()

                attn_output = gather_heads_scatter_seq(
                    attn_output,
                    seq_dim=seq_dim,
                    head_dim=head_dim,
                    gather_size=self.config.num_attention_heads,
                    group=get_context_parallel_group_for_hybrid_ulysses()
                )
                attn_output = attn_output.reshape(batch_size, seqlen, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output
            else:
                raise NotImplementedError(f"Only support `ulysses_cp_algo`,`megatron_cp_algo`,`hybrid_cp_algo`, but got {megatron_args.context_parallel_algo}")

        if layout == "BNSD":
            query_states = query_states.transpose(1, 2)  # BNSD
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attention_kwargs["attention_mask"] = attention_mask
        elif layout == "BSND":
            attention_kwargs["attention_mask"] = attention_mask
        elif layout == "TND":
            attention_kwargs["actual_seq_qlen"] = kwargs["cu_seqlens"]
            attention_kwargs["actual_seq_kvlen"] = kwargs["cu_seqlens"]
            indices = kwargs["indices"]
            # reshape BSND -> TND, and upad_input
            query_states = query_states.view(-1, *query_states.shape[2:])[indices]
            key_states = key_states.view(-1, *key_states.shape[2:])[indices]
            value_states = value_states.view(-1, *value_states.shape[2:])[indices]
        else:
            raise NotImplementedError(
                f"Unsupported Attention layout: {layout}, "
                "Qwen3VLTextAttention only support ['BNSD', 'BSND', 'TND'] now.")

        attn_output = attention_interface(
            query_states,
            key_states,
            value_states,
            **attention_kwargs,
        )

        if layout == "BNSD":
            attn_output = attn_output.transpose(1, 2)
        if layout == "TND":
            # pad output, and reshape to BSND
            attn_output = pad_out(attn_output, indices, batch_size, total_seq_len)
            attn_output = attn_output.view(batch_size, total_seq_len, *attn_output.shape[1:])

        if mpu.get_context_parallel_world_size() > 1:
            attn_output = gather_heads_scatter_seq(
                attn_output,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=self.num_heads
            )

        attn_output = attn_output.reshape(batch_size, seqlen, -1).contiguous()  # reshape to BSH
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Qwen3VLLMHead(nn.Linear):
    def forward(self, hidden_states: torch.Tensor, loss_ctx: callable = None):
        # Handle distributed tensor (DTensor) weights and biases by converting to local tensors.
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                if not isinstance(self.bias, DTensor):
                    raise TypeError(
                        f"Expected bias to be a DTensor when weight is a DTensor, "
                        f"but got bias of type {type(self.bias)}."
                    )
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias

        if loss_ctx is None:
            # If no loss context is provided, compute and return logits normally.
            logits = F.linear(hidden_states, w, b)
            return logits, None
        else:
            # Otherwise, delegate loss computation to the provided loss context function,
            # which typically enables memory-efficient or chunked loss calculation.
            return None, loss_ctx(hidden_states, w, b)
