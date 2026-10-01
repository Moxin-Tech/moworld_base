# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
import os
import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import cached_property

import torch
import torch_npu
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.models.chameleon.modeling_chameleon import (
    ChameleonRotaryEmbedding,
    ChameleonMLP,
    ChameleonImageVocabularyMapping,
    repeat_kv
)

from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.normalize import LlamaRMSNorm


class RotaryEmbedding(ChameleonRotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
            self.inv_freq = inv_freq.to(self.inv_freq.device)
        # x is bnsd
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin  # 1, seq_len, dim


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
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
    q_embed = npu_rotary_position_embedding(q, cos, sin, 0)
    k_embed = npu_rotary_position_embedding(k, cos, sin, 0)
    return q_embed, k_embed


@dataclass
class ChameleonMLPConfig:
    hidden_size: int = 4096
    intermediate_size: int = 11008
    mlp_bias: bool = False
    hidden_act: str = "silu"


class ChameleonLayerNorm(nn.LayerNorm):
    """
    LayerNorm but computes stats only over the last dim because Chameleon applies gamma and beta
    from each shard separately to each head, instead of reducing. We can apply each head's own
    gamma/beta by repeat-interleaving weights from each shard, but the stats have to be computed
    in the last dimension. This module applies gamma/beta manually to fulfill this requirement.
    """

    def __init__(self, hidden_size, model_parallel_size, n_heads_per_mp, *args, **kwargs):
        if isinstance(hidden_size, int):
            hidden_size = (hidden_size,)
        super().__init__([model_parallel_size, *hidden_size], *args, **kwargs)
        self.normalized_shape = (hidden_size[-1],)
        self.n_heads_per_mp = n_heads_per_mp

    def repeat_param(self, param):
        return param.repeat_interleave(self.n_heads_per_mp, dim=0)

    def forward(self, hidden_states):
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, None, None, eps=1e-5)
        hidden_states = hidden_states * self.repeat_param(self.weight) + self.repeat_param(self.bias)
        return hidden_states


class ChameleonAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        num_attention_heads,
        num_key_value_heads,
        hidden_size,
        attention_dropout,
        attention_bias,
        rope_theta,
        max_position_embeddings,
        layer_idx: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = True
        self.model_parallel_size = 1

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=attention_bias)
        self.q_norm = ChameleonLayerNorm(
            self.head_dim, self.model_parallel_size, self.num_heads // self.model_parallel_size
        )
        self.k_norm = ChameleonLayerNorm(
            self.head_dim, self.model_parallel_size, self.num_key_value_heads // self.model_parallel_size
        )
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=-2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)  # BNSD
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[1]]

        # upcast attention to fp32
        attn_output = torch_npu.npu_fusion_attention(
            query_states,
            key_states,
            value_states.to(query_states.dtype),
            atten_mask=causal_mask,
            input_layout="BSND",
            scale=self.head_dim ** -0.5,
            head_num=self.num_heads,
            keep_prob=1 - self.attention_dropout,
        )[0]

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, None


class ChameleonDecoderLayer(nn.Module):
    def __init__(self,
        hidden_size,
        intermediate_size,
        mlp_bias,
        hidden_act,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        rms_norm_eps,
        dropout,
        rope_theta,
        max_position_embeddings,
        layer_idx: int
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = ChameleonAttention(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            layer_idx=layer_idx,
        )

        self.mlp = ChameleonMLP(ChameleonMLPConfig(hidden_size, intermediate_size, mlp_bias, hidden_act))
        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[Union[torch.Tensor, bool]] = True,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if isinstance(output_attentions, torch.Tensor):
            output_attentions = output_attentions.item()

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class ChameleonModel(nn.Module):
    def __init__(
        self,
        pad_token_id,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        mlp_bias,
        hidden_act,
        vocabulary_map_path,
        swin_norm,
        rms_norm_eps,
        dropout,
        rope_theta,
        max_position_embeddings,
        **kwargs,
    ):
        super().__init__()
        args = get_args()
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)

        vocab_map = {}
        configs_path = os.path.join(vocabulary_map_path, 'config.json')
        with open(configs_path, 'r', encoding='utf-8') as file:
            configs = json.load(file)
            vocab_map = configs.get("vocabulary_map")
        self.vocabulary_mapping = ChameleonImageVocabularyMapping(vocab_map)

        self.layers = nn.ModuleList(
            [
                ChameleonDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    mlp_bias=mlp_bias,
                    hidden_act=hidden_act,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    attention_dropout=attention_dropout,
                    attention_bias=attention_bias,
                    rms_norm_eps=rms_norm_eps,
                    dropout=dropout,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    layer_idx=layer_idx
                ) for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.gradient_checkpointing = args.recompute_granularity

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs_embeds = self.embed_tokens(input_ids)
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = tensor_parallel.checkpoint(
                    decoder_layer,
                    False,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    torch.tensor(output_attentions) if output_attentions is not None else None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask.to(torch.bool)


class ChameleonForConditionalGeneration(MultiModalModule):
    def __init__(
        self,
        max_position_embeddings,
        vocab_size,
        hidden_size,
        intermediate_size,
        pad_token_id,
        num_layers,
        num_heads,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        mlp_bias,
        hidden_act,
        vocabulary_map_path,
        swin_norm,
        rms_norm_eps,
        dropout,
        rope_theta,
        **kwargs
    ):
        super().__init__(config=None)
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.output_attentions = kwargs.get("output_attentions", True)
        self.output_hidden_states = kwargs.get("output_hidden_states", False)

        self.model = ChameleonModel(
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            hidden_act=hidden_act,
            vocabulary_map_path=vocabulary_map_path,
            swin_norm=swin_norm,
            rms_norm_eps=rms_norm_eps,
            dropout=dropout,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_call_times = 0
        self.mask_image_logits = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.mask_image_logits: # False
            # Disallow image tokens which does not include special begin-image and end-image tokens
            image_tokens = self.model.vocabulary_mapping.image_tokens
            logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @property
    def device(self) -> torch.device:
        """The device of the module (assuming that all the module parameters are in the same device)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].device
        else:
            buffers = tuple(self.buffers())
            return buffers[0].device
