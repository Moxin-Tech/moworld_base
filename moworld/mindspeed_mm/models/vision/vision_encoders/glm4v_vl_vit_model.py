# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

from typing import Optional, Tuple
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType
from megatron.training.global_vars import get_args
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
from mindspeed.core.megatron_basic.megatron_basic import PTNorm

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import VisionRotaryEmbedding, PatchEmbed


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Glm4vRotaryEmbedding_llm(Qwen2VLRotaryEmbedding):
    def __init__(self, config: Optional[TransformerConfig] = None):
        super().__init__(config=config)
        # head_dim default: hidden_size // num_attention_heads, pass kv_channels to override it
        self.config.head_dim = self.config.kv_channels
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)


    @torch.no_grad()
    def forward(self, x_device, x_dtype, position_ids, mrope_section):

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x_device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = (cos * self.attention_scaling).contiguous()
        sin = (sin * self.attention_scaling).contiguous()

        # Extract the repetitive calculation from the apply_multimodal_rotary_pos_emb function.
        unsqueeze_dim = 1
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

        # Interleave them instead of usual shape
        cos = cos[..., : cos.shape[-1] // 2].transpose(0, -1).repeat_interleave(2, dim=0).transpose(0, -1).permute(2, 0, 1, 3).contiguous()
        sin = sin[..., : sin.shape[-1] // 2].transpose(0, -1).repeat_interleave(2, dim=0).transpose(0, -1).permute(2, 0, 1, 3).contiguous()
        return torch.concat((cos, sin), dim=-1).to(dtype=x_dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, use_fused_rope=True):

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    if use_fused_rope:
        q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin, rotary_mode='interleave')
        k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin, rotary_mode='interleave')
    else:
        q_embed = (q_rot * cos) + (rotate_half_llm(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half_llm(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class Glm4vSelfAttention(SelfAttention):
    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

        self.mrope_section = config.mrope_section

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_params=None,
    ):
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)  # s b h d

        if self.config.context_parallel_size > key.shape[2]:
            key = key.repeat_interleave(
                query.shape[2] // key.shape[2], dim=2
            )
            value = value.repeat_interleave(
                query.shape[2] // value.shape[2], dim=2
            )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        if rotary_pos_emb is not None:
            half_dim = rotary_pos_emb.shape[-1] // 2
            cos, sin = rotary_pos_emb[..., :half_dim], rotary_pos_emb[..., half_dim:]
            query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin,
                                                         use_fused_rope=self.config.use_fused_rotary_pos_emb)  # b h s d
        # ===================================================
        # Adjust key, value for inference
        # ===================================================
        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            None,
        )
        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # from (t, np, hn) to (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Glm4vVisionAttention(SelfAttention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        inference_params=None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        packed_seq_params=None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        cos, sin = rotary_pos_emb

        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(1)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        attention_interface = ALL_ATTENTION_FUNCTIONS['sdpa']
        core_attn_out, _ = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scale,
            is_causal=False,
            **kwargs,
        )

        attn_output = core_attn_out.squeeze(0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output, None


class Glm4vVisionEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords) -> torch.Tensor:
        """
        Forward pass with integrated position encoding adaptation using 2D interpolation.

        Args:
            embeddings: Input embeddings tensor
            lengths (torch.Tensor): Sequence lengths for each image in the batch.
            image_shapes (torch.Tensor): Tensor of shape [batch_size, 3] representing the image shapes (t, h, w).
            h_coords (torch.Tensor): Tensor of shape [total_seq] representing the h coordinate for each patch.
            w_coords (torch.Tensor): Tensor of shape [total_seq] representing the w coordinate for each patch.

        Returns:
            torch.Tensor: Embeddings with adapted position encoding added.
        """
        # Get position embedding parameters
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Move coordinates to correct device
        h_coords, w_coords = h_coords.to(device), w_coords.to(device)

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            # Convert inputs to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size).permute(2, 0, 1).unsqueeze(0).float()
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).float()
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).float()

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(dtype=torch.float32)
            w_coords = w_coords.to(dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bicubic", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class GlmTransformerBlock(TransformerBlock):
    def _build_layers(self):
        super()._build_layers()

        if self.post_process and self.post_layer_norm:
            self.final_layernorm = PTNorm(config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon)
        else:
            self.final_layernorm = None


class GlmViT(MultiModalModule):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
        patch_size: int = 14,
        pre_process: bool = True,
        post_process: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(config=config)
        config.layernorm_epsilon = config.rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.pre_process = pre_process
        self.post_process = post_process

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            bias=True,
        )
        self.post_conv_layernorm = PTNorm(config=self.config, hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.embeddings = Glm4vVisionEmbeddings(config)

        self.blocks = GlmTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            post_layer_norm=True,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        self.downsample = nn.Conv2d(
            in_channels=config.hidden_size,
            out_channels=config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )

        self.gradient_checkpointing = False

    def set_input_tensor(self, input_tensor):
        self.blocks.set_input_tensor(input_tensor)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb, pos_ids

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        if self.pre_process:
            if pixel_values is None or grid_thw is None:
                raise ValueError('You have to specify pixel_values and grid_thw')
            else:
                hidden_states = self.patch_embed(pixel_values)
        else:
            hidden_states = None
        hidden_states = self.post_conv_layernorm(hidden_states)

        rotary_pos_emb, image_type_ids = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        hidden_states = self.embeddings(hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1])

        seq_length = hidden_states.shape[0]
        attention_mask = torch.zeros([1, seq_length, seq_length], device=hidden_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = True

        hidden_states = self.blocks(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=position_embeddings
        )

        if self.post_process:
            hidden_states = hidden_states.view(
                -1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1]
            )
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)

        return hidden_states
