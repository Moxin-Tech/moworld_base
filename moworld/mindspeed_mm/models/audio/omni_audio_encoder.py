# Copyright 2025 The Qwen team; Alibaba Group and the HuggingFace Inc. team. All rights reserved.

import numpy as np
import torch
from torch import nn as nn

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlVitSelfAttention


class AudioLinear(torch.nn.Linear):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.matmul(input_, self.weight.T) + self.bias
        else:
            return torch.matmul(input_, self.weight.T)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp((-log_timescale_increment * torch.arange(channels // 2).to(torch.bfloat16))).float()
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class QwenOmniAudioSelfAttention(Qwen2vlVitSelfAttention):
    """Omni Audio模块的q_bias/v_bias为True,k_bias为False，Megatron的SelfAttention是一个统一的linear_qkv.bias
    这里为了迁移到Megatron的SelfAttention适配tp，将linear_qkv.bias中的k_bias初始权重置0并在反向更新时将k_bias部分拆出来对应的梯度置0
    """

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int,
                 attn_mask_type=AttnMaskType.padding):
        super().__init__(config, submodules, layer_number, attn_mask_type)

        def freeze_k_bias_grad_hook(grad):
            grad_clone = grad.clone()
            head_size = self.hidden_size_per_attention_head
            num_heads = self.num_attention_heads_per_partition
            # 遍历每个注意力头，冻结其对应的 K 部分
            for i in range(num_heads):
                start = i * QKV_SIZE * head_size + head_size
                end = start + head_size
                grad_clone[start:end, ...] = 0  # 置零梯度
            return grad_clone

        self.linear_qkv.bias.register_hook(freeze_k_bias_grad_hook)

    def apply_rotary_pos_emb_qk(self, rotary_pos_emb, query, key):
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config,
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config,
        )

        return query, key


QKV_SIZE = 3
