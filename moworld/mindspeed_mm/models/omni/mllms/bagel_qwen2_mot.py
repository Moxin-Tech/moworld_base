# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
#
# This modified file is released under the same license.

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch_npu
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.utils import logging

from mindspeed_mm.data.datasets.bagel_dataset import create_sparse_mask
from mindspeed_mm.models.common.embeddings.pos_embeddings import PositionEmbedding
from mindspeed_mm.models.common.embeddings.time_embeddings import timestep_embedding
from mindspeed_mm.models.common.normalize import LlamaRMSNorm
from mindspeed_mm.utils.utils import is_npu_available


class TimeStepEmbedding(nn.Module):
    """Time step embedding module for diffusion models."""
    def __init__(self, hidden_size, time_embed_dim=256, max_period=10000, repeat_only=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.max_period = max_period
        self.repeat_only = repeat_only
        # MLP for time embedding transformation
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the module parameters."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def forward(self, timesteps):
        """Generate time embeddings for diffusion steps."""
        emb = timestep_embedding(timesteps, self.time_embed_dim, self.max_period, self.repeat_only, dtype=self.dtype)
        return self.time_embed(emb)


class PackedAttention(Qwen2Attention):
    """Attention module for packed sequence processing with efficient batching."""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # Query-Key normalization for stability
        if self.config.qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
            self,
            packed_sequence: torch.Tensor,
            sample_lens: List[int],
            attention_mask: List[torch.Tensor],
            packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        # Project input to query, key, value states
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        # Apply normalization if enabled
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        # Apply rotary position embeddings
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        # Expand key-value states for group query attention
        packed_key_states = packed_key_states[:, :, None, :].repeat_interleave(self.num_key_value_groups, dim=2)
        packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
        packed_value_states = packed_value_states[:, :, None, :].repeat_interleave(self.num_key_value_groups, dim=2)
        packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

        # Split packed sequences into individual samples
        unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)

        # Process each sample individually with NPU optimized attention
        upacked_attn_output = []
        for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
        ):
            # Prepare attention mask for NPU kernel
            if attention_mask_per_sample.dtype == torch.bool:
                atten_mask_npu = torch.logical_not(attention_mask_per_sample.bool()).to(packed_sequence.device)
            else:
                atten_mask_npu = attention_mask_per_sample.bool().to(packed_sequence.device)

            # Use NPU optimized attention kernel
            head_num = query_states.shape[0]
            attn_output = torch_npu.npu_fusion_attention(
                query_states.unsqueeze(0),
                key_states.unsqueeze(0),
                value_states.unsqueeze(0),
                head_num, input_layout="BNSD",
                pse=None,
                atten_mask=atten_mask_npu,
                scale=1.0 / math.sqrt(query_states.shape[-1]),
                pre_tockens=2147483647,
                next_tockens=2147483647,
                keep_prob=1
            )[0]
            upacked_attn_output.append(attn_output.squeeze(0))
        packed_attn_output = torch.cat(upacked_attn_output, dim=1)

        # Project attention output back to hidden size
        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output


class PackedAttentionMoT(Qwen2Attention):
    """Mixture of Tokens (MoT) attention with separate pathways for understanding and generation tokens."""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.config.qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            packed_sequence: torch.Tensor,
            sample_lens: List[int],
            attention_mask,
            packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            packed_und_token_indexes: torch.LongTensor,
            packed_gen_token_indexes: torch.LongTensor,
    ):
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros(
            (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        # Separate understanding and generation tokens
        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

        # Apply different projections based on token type
        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        # Reshape to multi-head format
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Freeze understanding tokens if configured
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        # Apply separate normalization for different token types
        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(
            packed_query_states[packed_gen_token_indexes])

        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        # Apply rotary position embeddings
        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_key_states_ = packed_key_states_[:, :, None, :].repeat_interleave(self.num_key_value_groups, dim=2)
        packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
        packed_value_states = packed_value_states[:, :, None, :].repeat_interleave(self.num_key_value_groups, dim=2)
        packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

        unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
        unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
        upacked_attn_output = []
        for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
        ):
            if attention_mask_per_sample.dtype == torch.bool:
                atten_mask_npu = torch.logical_not(attention_mask_per_sample.bool()).to(packed_sequence.device)
            else:
                atten_mask_npu = attention_mask_per_sample.bool().to(packed_sequence.device)
            head_num = query_states.shape[0]
            attn_output = torch_npu.npu_fusion_attention(
                query_states.unsqueeze(0),
                key_states.unsqueeze(0),
                value_states.unsqueeze(0),
                head_num, input_layout="BNSD",
                pse=None,
                atten_mask=atten_mask_npu,
                scale=1.0 / math.sqrt(query_states.shape[-1]),
                pre_tockens=2147483647,
                next_tockens=2147483647,
                keep_prob=1
            )[0]
            upacked_attn_output.append(attn_output.squeeze(0))
        packed_attn_output = torch.cat(upacked_attn_output, dim=1)

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(
            packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_


class Qwen2DecoderLayer(nn.Module):
    """Standard transformer decoder layer with packed attention."""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            packed_sequence: torch.Tensor,
            sample_lens: List[int],
            attention_mask,
            packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention with residual connection
        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # MLP with residual connection
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence


class Qwen2MoTDecoderLayer(nn.Module):
    """Mixture of Tokens decoder layer with separate pathways for understanding and generation."""
    def __init__(
            self,
            config,
            layer_idx: Optional[int] = None,
            attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            packed_sequence: torch.Tensor,
            sample_lens: List[int],
            attention_mask,
            packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            packed_und_token_indexes: torch.LongTensor,
            packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(
            packed_sequence[packed_gen_token_indexes])
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        # Freeze understanding tokens if configured
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected layer with separate MLPs
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()

        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence


# Registry for different decoder layer types
Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    """Main Qwen2 model with support for packed sequence processing and MoT."""
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = 'Mo' in config.layer_module  # Check if using Mixture of Tokens

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Select decoder layer type from registry
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Final normalization layers
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            packed_sequence: torch.Tensor,
            sample_lens: List[int],
            attention_mask,
            packed_position_ids: torch.Tensor,
            packed_und_token_indexes: Optional[torch.LongTensor] = None,
            packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()

        # Generate rotary position embeddings shared across layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        # Prepare extra inputs for MoT layers
        extra_inputs = {}
        if self.use_moe:
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        # Process through all decoder layers
        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        # Apply separate final normalization for MoT
        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    """Qwen2 model for causal language modeling with multi-modal support."""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # Extract and remove multi-modal configs from main config
        self.image_encoder_config = None
        if hasattr(config, 'image_encoder'):
            self.image_encoder_config = config.image_encoder
            del config.image_encoder
        self.ae_config = None
        if hasattr(config, 'ae'):
            self.ae_config = config.ae
            del config.ae

        self.config = config
        self.decoder = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize multi-modal components
        self._init_core_parameters()
        self._init_position_embeddings()
        if self.ae_config:
            self._init_time_embedder()
            self._init_latent_projection_layers()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_core_parameters(self):
        self.hidden_size = self.config.hidden_size
        if self.image_encoder_config:
            self.vit_max_num_patch_per_side = self.image_encoder_config.vit_max_num_patch_per_side

        if self.ae_config:
            self.latent_patch_size = self.ae_config.latent_patch_size
            self.timestep_shift = self.ae_config.timestep_shift
            self.latent_downsample = self.ae_config.downsample * self.latent_patch_size
            self.max_latent_size = self.ae_config.max_latent_size
            self.latent_channel = self.ae_config.z_channels
            self.patch_latent_dim = (self.latent_patch_size ** 2) * self.latent_channel

    def _init_position_embeddings(self):
        if self.image_encoder_config:
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
        if self.ae_config:
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

    def _init_time_embedder(self):
        self.time_embedder = TimeStepEmbedding(self.hidden_size)

    def _init_latent_projection_layers(self):
        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.decoder = decoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            sequence_length: int,
            packed_text_ids: torch.LongTensor,
            packed_text_indexes: torch.LongTensor,
            sample_lens: List[int],
            packed_position_ids: torch.LongTensor,
            nested_attention_masks: List[torch.Tensor] = None,
            split_lens: List[int] = None,
            attn_modes: List[str] = None,
            # for visual understanding
            ce_loss_indexes: Optional[torch.BoolTensor] = None,
            packed_label_ids: Optional[torch.LongTensor] = None,
            packed_vit_token_indexes: Optional[torch.LongTensor] = None,
            packed_vit_position_ids: Optional[torch.LongTensor] = None,
            vit_inputs_embeds: Optional[torch.IntTensor] = None,
            # for visual generation
            padded_latent: Optional[torch.Tensor] = None,
            patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
            packed_latent_position_ids: Optional[torch.LongTensor] = None,
            packed_vae_token_indexes: Optional[torch.LongTensor] = None,
            packed_timesteps: Optional[torch.LongTensor] = None,
            mse_loss_indexes: Optional[torch.BoolTensor] = None,
            **kwargs
    ) -> torch.Tensor:
        packed_text_embedding = self.decoder.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if self.image_encoder_config:
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = vit_inputs_embeds + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        if self.ae_config:
            p = self.latent_patch_size
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)
            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps)
            packed_timesteps = self.timestep_shift * packed_timesteps / (
                        1 + (self.timestep_shift - 1) * packed_timesteps)
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
                device=packed_text_embedding.device
            )
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask=sparse_mask,
                B=1,
                H=self.num_heads,
                Q_LEN=seqlen,
                KV_LEN=seqlen,
                device=packed_text_embedding.device,
                BLOCK_SIZE=128,
                _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        extra_inputs = {}
        if self.config.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update({
                "packed_und_token_indexes": packed_und_token_indexes,
                "packed_gen_token_indexes": packed_vae_token_indexes
            })
        last_hidden_state = self.decoder(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            **extra_inputs
        )
        mse = None
        if self.ae_config:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
            target = noise - packed_latent_clean  # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        if self.image_encoder_config:
            packed_ce_preds = self.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        loss = 0
        if ce is not None:
            total_ce_tokens = torch.tensor(len(ce_loss_indexes), device=self.device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss = loss + ce

        if mse is not None:
            total_mse_tokens = torch.tensor(len(mse_loss_indexes), device=self.device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss = loss + mse

        return [loss]
