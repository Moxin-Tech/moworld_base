# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.

from typing import Optional, Callable

import torch
from transformers import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from megatron.core import mpu
from mindspeed_mm.models.transformers.cp_utils import get_seq_len, gather_seq_scatter_heads_qkv, \
    gather_heads_scatter_seq


class MMMistralAttention(MistralAttention):
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Ulysses Context Parallel: Gather sequence dimension and scatter head dimension for QKV
        # This allows parallel processing across context sequence length while maintaining communication efficiency
        total_seq_len = get_seq_len("total")
        if mpu.get_context_parallel_world_size() > 1:
            seq_dim, head_dim = 2, 1
            query_states, key_states, value_states = gather_seq_scatter_heads_qkv(
                query_states,
                key_states,
                value_states,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=total_seq_len
            )
        # ---------ulysses-cp------------

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        # Ulysses Context Parallel: Gather head dimension and scatter sequence dimension for attention output
        # This reverses the earlier scattering operation, gathering results back along heads and scattering along sequence
        if mpu.get_context_parallel_world_size() > 1:
            seq_dim, head_dim = 1, 2
            attn_output = gather_heads_scatter_seq(
                attn_output,
                seq_dim=seq_dim,
                head_dim=head_dim,
                gather_size=self.config.num_attention_heads
            )
        # ---------ulysses-cp------------

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
