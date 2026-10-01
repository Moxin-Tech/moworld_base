# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.

__all__ = ["Glm4vFusedMoeForConditionalGeneration"]

import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

import transformers
from transformers.activations import ACT2FN
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeTextConfig, Glm4vMoeVisionConfig
from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeForConditionalGeneration, Glm4vMoeTextDecoderLayer, Glm4vMoeTextMLP, \
    Glm4vMoeVisionModel, Glm4vMoeRMSNorm, Glm4vMoeTextTopkRouter, Glm4vMoeVisionBlock, Glm4vMoeTextModel, Glm4vMoeVisionEmbeddings
from megatron.training import get_args
from mindspeed_mm.models.common.gmm import npu_group_gemm


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
    """NPU fused rope"""
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed = q_embed.squeeze(0)
    k_embed = k_embed.squeeze(0)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
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

    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    # npu fused rope
    q_embed = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


transformers.models.glm4v_moe.modeling_glm4v_moe.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
transformers.models.glm4v_moe.modeling_glm4v_moe.apply_multimodal_rotary_pos_emb = apply_multimodal_rotary_pos_emb


class Glm4vMoeTextExperts(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.num_experts = config.n_routed_experts
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts * self.hidden_size, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts * self.intermediate_size, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def _view_experts_weight(self):
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        gate_up_proj = gate_up_proj.view(self.num_experts, self.hidden_size, -1)

        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        down_proj = down_proj.view(self.num_experts, self.intermediate_size, -1)
        return gate_up_proj, down_proj

    def forward(
        self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            topk_weights (torch.Tensor): (batch_size * token_num, num_experts)
            topk_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        gate_up_proj, down_proj = self._view_experts_weight()
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 0, 1)
            # we sum on the top_k and on the sequence length to get which experts
            # are hit this time around
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit[:]:
            with torch.no_grad():
                token_idx, weight_idx = torch.where(expert_mask[expert_idx[0]])
            gate_up = hidden_states[token_idx] @ gate_up_proj[expert_idx]
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = (self.act_fn(gate) * up) @ down_proj[expert_idx]
            weighted_output = expert_output[0] * topk_weights[token_idx, weight_idx, None]
            final_hidden_states.index_add_(0, token_idx, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)


class Glm4vFusedMoeTextExperts(Glm4vMoeTextExperts):
    """NPU fusd Moe"""

    def forward(
        self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        gate_up_proj, down_proj = self._view_experts_weight()

        batch_size = hidden_states.shape[0]
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, topk_indices.to(torch.int32))
        tokens_per_expert = torch.histc(topk_indices, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, gate_up_proj, tokens_per_expert)
        intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
        output = npu_group_gemm(intermediate_activations, down_proj, tokens_per_expert)
        final_hidden_states = torch_npu.npu_moe_token_unpermute(output.to(topk_weights.dtype), row_ids_map, probs=topk_weights)
        final_hidden_states = final_hidden_states.view(batch_size, -1, self.hidden_size)
        return final_hidden_states.type(orig_dtype)


class Glm4vFusedMoeTextMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__()
        self.config = config
        self.use_npu_fused_moe = getattr(get_args().mm.model.text_decoder, "use_npu_fused_moe", True)
        if self.use_npu_fused_moe:
            self.experts = Glm4vFusedMoeTextExperts(config, intermediate_size=config.moe_intermediate_size)
        else:
            self.experts = Glm4vMoeTextExperts(config, intermediate_size=config.moe_intermediate_size)
        self.gate = Glm4vMoeTextTopkRouter(config)
        self.shared_experts = Glm4vMoeTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states):
        torch.npu.synchronize()
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, topk_weights, topk_indices).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class Glm4vFusedMoeRMSNorm(Glm4vMoeRMSNorm):
    def forward(self, hidden_states):
        """NPU fused rms_norm"""
        return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


class Glm4vFusedMoeVisionEmbeddings(Glm4vMoeVisionEmbeddings):
    """Since the NPU dose not yet support "bicubic" mode for F.griad_sample, the "bilinear" mode is used as a temporary replacement here."""

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
            orig_size = int(orig_size_sq ** 0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )

            # Normalize coordinates to [-1, 1] range for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Perform bicubic interpolation
            interpolated_embed_fp32 = F.grid_sample(
                pos_embed_2d, grid, mode="bilinear", align_corners=False, padding_mode="border"
            )

            # Reshape and convert back to original dtype
            adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
            adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)

        # Add adapted position encoding to embeddings
        embeddings = embeddings + adapted_pos_embed
        return embeddings


class Glm4vFusedMoeVisionBlock(Glm4vMoeVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.norm1 = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4vFusedMoeVisionModel(Glm4vMoeVisionModel):
    config: Glm4vMoeVisionConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.embeddings = Glm4vFusedMoeVisionEmbeddings(config)
        self.blocks = nn.ModuleList([Glm4vFusedMoeVisionBlock(config) for _ in range(config.depth)])
        self.post_conv_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class Glm4vFusedMoeTextDecoderLayer(Glm4vMoeTextDecoderLayer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4vFusedMoeTextMoE(config)
        else:
            self.mlp = Glm4vMoeTextMLP(config)

        self.input_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Glm4vFusedMoeTextModel(Glm4vMoeTextModel):
    config: Glm4vMoeTextConfig

    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Glm4vFusedMoeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Glm4vFusedMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class Glm4vFusedMoeForConditionalGeneration(Glm4vMoeForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model.visual = Glm4vFusedMoeVisionModel._from_config(config.vision_config)
        self.model.language_model = Glm4vFusedMoeTextModel._from_config(config.text_config)
        self.post_init()
