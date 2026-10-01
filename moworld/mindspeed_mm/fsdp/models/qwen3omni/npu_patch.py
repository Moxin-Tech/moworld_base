# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

import torch
from torch import nn

from transformers.activations import ACT2FN
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.ops.swiglu import swiglu
from . import modeling_qwen3_omni_moe


class Qwen3OmniMoeThinkerTextExpertsGemm(nn.ModuleList):
    """
    ModuleList of experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size \
            if config.moe_intermediate_size is None else config.moe_intermediate_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size * sequence_length, hidden_dim)
            selected_experts: (batch_size * sequence_length, top_k)
            routing_weights: (batch_size * sequence_length, top_k)
        Returns:
            (batch_size * sequence_length, hidden_dim)
        """

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        permuted_hidden_states, row_ids_map = permute(hidden_states, top_k_index.to(torch.int32), fused=True)
        tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
        intermediate_hidden_states = grouped_matmul(permuted_hidden_states, self.gate_up_proj, tokens_per_expert, fused=True)
        intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=True)
        output = grouped_matmul(intermediate_activations, self.down_proj, tokens_per_expert, fused=True)
        next_states = unpermute(output, row_ids_map, probs=top_k_weights, fused=True)
        next_states = next_states.view(batch_size, -1, self.hidden_size)
        return next_states


def apply_qwen3_omni_moe_npu_patch():
    import torch.nn.functional as F
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator

    # Patches for Qwen3-Omni Model
    modeling_qwen3_omni_moe.apply_rotary_pos_emb_vision = npu_fused_operator.apply_transformers_vision_rope_half_npu
    modeling_qwen3_omni_moe.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
    modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    F.gelu = npu_fused_operator.apply_gelu_npu
