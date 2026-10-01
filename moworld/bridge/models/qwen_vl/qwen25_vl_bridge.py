# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.core.transformer.module import MegatronModule
from bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    VisionEncoderQKVMapping,
)
from mindspeed_mm.models.vlm_model import VLMModel


class Qwen2_5_VLForConditionalGeneration():
    pass


@MegatronModelBridge.register_bridge(source=Qwen2_5_VLForConditionalGeneration, target=VLMModel)
class Qwen25VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2.5-VL Conditional Generation.
    """

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "text_decoder.embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "text_decoder.decoder.final_layernorm.weight": "model.norm.weight",
            "text_decoder.output_layer.weight": "lm_head.weight",
            "text_decoder.decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "text_decoder.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "text_decoder.decoder.layers.*.mlp.linear_fc2.bias": "model.layers.*.mlp.down_proj.bias",
            "text_decoder.decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "text_decoder.decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "text_decoder.decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",

            "image_encoder.encoder.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
            "image_encoder.projector.layernorm.weight": "visual.merger.ln_q.weight",
            "image_encoder.projector.encoder.linear_fc1.weight": "visual.merger.mlp.0.weight",
            "image_encoder.projector.encoder.linear_fc1.bias": "visual.merger.mlp.0.bias",
            "image_encoder.projector.encoder.linear_fc2.weight": "visual.merger.mlp.2.weight",
            "image_encoder.projector.encoder.linear_fc2.bias": "visual.merger.mlp.2.bias",
            "image_encoder.encoder.blocks.layers.*.self_attention.linear_proj.weight": "visual.blocks.*.attn.proj.weight",
            "image_encoder.encoder.blocks.layers.*.self_attention.linear_proj.bias": "visual.blocks.*.attn.proj.bias",
            "image_encoder.encoder.blocks.layers.*.self_attention.linear_qkv.layer_norm_weight": "visual.blocks.*.norm1.weight",
            "image_encoder.encoder.blocks.layers.*.mlp.linear_fc2.weight": "visual.blocks.*.mlp.down_proj.weight",
            "image_encoder.encoder.blocks.layers.*.mlp.linear_fc2.bias": "visual.blocks.*.mlp.down_proj.bias",
            "image_encoder.encoder.blocks.layers.*.pre_mlp_layernorm.weight": "visual.blocks.*.norm2.weight",
            "image_encoder.encoder.blocks.layers.*.input_layernorm.weight": "visual.blocks.*.norm1.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="visual.**",
                ),
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="text_decoder.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # QKV bias: Combine separate Q, K, V biases into single QKV bias (Qwen2 specific)
                QKVMapping(
                    megatron_param="text_decoder.decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="text_decoder.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="text_decoder.decoder.layers.*.mlp.linear_fc1.bias",
                    gate="model.layers.*.mlp.gate_proj.bias",
                    up="model.layers.*.mlp.up_proj.bias",
                ),
                GatedMLPMapping(
                    megatron_param="image_encoder.encoder.blocks.layers.*.mlp.linear_fc1.weight",
                    gate="visual.blocks.*.mlp.gate_proj.weight",
                    up="visual.blocks.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="image_encoder.encoder.blocks.layers.*.mlp.linear_fc1.bias",
                    gate="visual.blocks.*.mlp.gate_proj.bias",
                    up="visual.blocks.*.mlp.up_proj.bias",
                ),
                VisionEncoderQKVMapping(
                    megatron_param="image_encoder.encoder.blocks.layers.*.self_attention.linear_qkv.bias",
                    hf_param="visual.blocks.*.attn.qkv.bias",
                ),
                VisionEncoderQKVMapping(
                    megatron_param="image_encoder.encoder.blocks.layers.*.self_attention.linear_qkv.weight",
                    hf_param="visual.blocks.*.attn.qkv.weight",
                )

            ]
        )

        return MegatronMappingRegistry(*mapping_list)
