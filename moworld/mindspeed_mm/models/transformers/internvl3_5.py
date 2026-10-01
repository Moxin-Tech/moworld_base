# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
import torch

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    Qwen3MoeMLP,
    Qwen3MoeRMSNorm
)

from mindspeed_mm.models.transformers.base_model import FSDP2Mixin
from mindspeed_mm.models.transformers.qwen3vl.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
from mindspeed_mm.models.transformers.custom_model_registry import register_model

try:
    from internvl.modeling_internvl_chat import InternVLChatModel
except ModuleNotFoundError:
    class InternVLChatModel:
        def __init__(self, config):
            raise Exception("Cannot read modeling_internvl_chat.py")


def internvl_moe_decoder_layer_init(self, config, layer_idx: int):
    super(Qwen3MoeDecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size

    self.self_attn = Qwen3MoeAttention(config, layer_idx)

    if (layer_idx not in config.mlp_only_layers) and (
        config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
    ):
        self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
    else:
        self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

    self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def internvl_moe_decoder_layer_forward(self, hidden_states, **kwargs):
    # Memory bloat exists, synchronization can be used to avoid it.
    torch.npu.synchronize()
    outputs = self.forward_before_patch(hidden_states, **kwargs)
    return outputs


@register_model("internvl")
class InternVLChatModelGeneration(InternVLChatModel, FSDP2Mixin):
    def __init__(self, config, vision_model=None, language_model=None, use_flash_attn=True):
        Qwen3MoeDecoderLayer.__init__ = internvl_moe_decoder_layer_init
        Qwen3MoeDecoderLayer.forward_before_patch = Qwen3MoeDecoderLayer.forward
        Qwen3MoeDecoderLayer.forward = internvl_moe_decoder_layer_forward
        super().__init__(config)
