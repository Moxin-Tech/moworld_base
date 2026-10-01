# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from typing import Optional

import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig

from megatron.core import mpu
from megatron.training import get_args
from ..attention_utils import ALL_ATTENTION_FUNCTIONS, pad_out
from ..cp_utils import get_seq_len, gather_seq_scatter_heads_qkv, gather_heads_scatter_seq


class Qwen3OmniMoeAudioAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_decoder = False
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        seq_length = hidden_states.size(0) # [seq_length, d_model=1280]
        total_audio_seqlen = int(cu_seqlens[-1])
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1) # [s,n,]
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        attention_kwargs = {
            "scale": self.scaling,
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "is_causal": self.is_causal,
            "layout": self.config.attn_layout,
        }
        seq_dim, head_dim = 0, 1
        if self.config._attn_implementation == "flash_attention_2" and self.config.attn_layout == "TND":
            attention_kwargs["actual_seq_qlen"] = cu_seqlens
            attention_kwargs["actual_seq_kvlen"] = cu_seqlens
        elif (
            self.config._attn_implementation in ["eager", "sdpa", "flash_attention_2"]
            and self.config.attn_layout == "BNSD"
        ):
            query_states = query_states.transpose(0, 1).unsqueeze(0) # [1,n,s,]
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            attention_kwargs["attention_mask"] = attention_mask
            seq_dim, head_dim = 2, 1
        else:
            raise NotImplementedError(
                f"Unsupported Attention: {self.config._attn_implementation}, or layout: {self.config.attn_layout}"
                "Qwen3OmniMoeAudioAttention only support ['eager', 'sdpa', 'flash_attention_2'], layout TND and BNSD")

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Support context parallel only when CP size < seq_len; reasons are as follows:
        # 1) Data is most likely fake when CP size ≥ seq_len; 2) Splitting provides no benefit for short real sequences
        if mpu.get_context_parallel_world_size() > 1 and mpu.get_context_parallel_world_size() < total_audio_seqlen:
            megatron_args = get_args()
            if megatron_args.context_parallel_algo == "ulysses_cp_algo":
                query_states, key_states, value_states = gather_seq_scatter_heads_qkv(
                    query_states,
                    key_states,
                    value_states,
                    seq_dim=seq_dim,
                    head_dim=head_dim,
                    gather_size=total_audio_seqlen
                )
            else:
                raise NotImplementedError(f"Only support `ulysses_cp_algo`, but got {megatron_args.context_parallel_algo}")

        attn_output = attention_interface(
            query_states,
            key_states,
            value_states,
            **attention_kwargs,
        )

        if mpu.get_context_parallel_world_size() > 1 and mpu.get_context_parallel_world_size() < total_audio_seqlen:
            attn_output = gather_heads_scatter_seq(
                attn_output,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=self.num_heads
            )

        if self.config.attn_layout == "BNSD":
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output


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


class Qwen3OmniMoeVisionAttention(nn.Module):
    def __init__(self, config: Qwen3OmniMoeVisionEncoderConfig) -> None:
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
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        seq_dim, head_dim = None, None
        attention_kwargs = {
            "scale": self.scaling,
            "dropout": 0.0 if not self.training else self.attention_dropout,
            "is_causal": self.is_causal,
            "attention_mask": None,
        }
        if self.config._attn_implementation == "flash_attention_2" and self.config.attn_layout == "TND":
            seq_dim, head_dim = 0, 1
            attention_kwargs["actual_seq_qlen"] = cu_seqlens
            attention_kwargs["actual_seq_kvlen"] = cu_seqlens
            attention_kwargs["layout"] = "TND"

        elif self.config._attn_implementation in ["eager", "sdpa", "flash_attention_2"] and self.config.attn_layout == "BNSD":
            # layout, TND --> BNSD
            query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1, N, T(B*S), D]
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            seq_dim, head_dim = 2, 1
            attention_kwargs["layout"] = "BNSD"
        else:
            raise NotImplementedError(
                f"Unsupported Attention: {self.config._attn_implementation}, or layout: {self.config.attn_layout}"
                "Qwen3OmniMoeVisionAttention only support ['eager', 'sdpa', 'flash_attention_2'], layout TND and BNSD")

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
            else:
                raise NotImplementedError(f"Only support `ulysses_cp_algo`, but got {megatron_args.context_parallel_algo}")

        if self.config.attn_layout == "TND":
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

        if self.config.attn_layout == "BNSD":
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3OmniMoeThinkerTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3OmniMoeThinkerTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
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


class Qwen3OmniMoeThinkerTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
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
        self.q_norm = Qwen3OmniMoeThinkerTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3OmniMoeThinkerTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
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

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        dropout = 0.0 if not self.training else self.attention_dropout
        attention_kwargs = {
            "scale": self.scaling,
            "dropout": dropout,
            "is_causal": self.is_causal,
            "layout": self.config.attn_layout,
            "enable_gqa": True
        }

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
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
            else:
                raise NotImplementedError(f"Only support `ulysses_cp_algo`, but got {megatron_args.context_parallel_algo}")

        if self.config.attn_layout == "BNSD":
            query_states = query_states.transpose(1, 2)  # BNSD
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attention_kwargs["attention_mask"] = attention_mask
        elif self.config.attn_layout == "BSND":
            attention_kwargs["attention_mask"] = attention_mask
        elif self.config.attn_layout == "TND":
            attention_kwargs["actual_seq_qlen"] = kwargs["cu_seqlens"]
            attention_kwargs["actual_seq_kvlen"] = kwargs["cu_seqlens"]
            indices = kwargs["indices"]
            # reshape BSND -> TND, and upad_input
            query_states = query_states.view(-1, *query_states.shape[2:])[indices]
            key_states = key_states.view(-1, *key_states.shape[2:])[indices]
            value_states = value_states.view(-1, *value_states.shape[2:])[indices]
        else:
            raise NotImplementedError(
                f"Unsupported Attention layout: {self.config.attn_layout}, "
                "Qwen3OmniMoeThinkerTextAttention only support ['BNSD', 'BSND', 'TND'] now.")

        attn_output = attention_interface(
            query_states,
            key_states,
            value_states,
            **attention_kwargs,
        )

        if self.config.attn_layout == "BNSD":
            attn_output = attn_output.transpose(1, 2)
        if self.config.attn_layout == "TND":
            # pad output, and reshape to BSND
            attn_output = pad_out(attn_output, indices, batch_size, total_seq_len)
            attn_output = attn_output.view(batch_size, total_seq_len, *attn_output.shape[1:])

        if mpu.get_context_parallel_world_size() > 1:
            attn_output = gather_heads_scatter_seq(
                attn_output,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=self.config.num_attention_heads
            )

        attn_output = attn_output.reshape(batch_size, seqlen, -1).contiguous()  # reshape to BSH
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3OmniLMHead(nn.Linear):
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
