# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

from dataclasses import dataclass
from typing import Optional
import torch

from transformers.modeling_outputs import MoeCausalLMOutputWithPast


@dataclass
class Qwen3OmniMoeThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None
