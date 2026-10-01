# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from megatron.core.transformer.module import MegatronModule
from bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.conversion.param_mapping import AutoMapping
from mindspeed_mm.models.sora_model import SoRAModel


class WanTransformer3DModel():
    pass


@MegatronModelBridge.register_bridge(source=WanTransformer3DModel, target=SoRAModel)
class Wan21Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2.5-VL Conditional Generation.

    This bridge handles the conversion between HuggingFace Qwen2_5_VLForConditionalGeneration
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation for vision-language models.

    Example:
        >>> from bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        >>> provider = bridge.to_megatron_provider()
    """

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
            "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
            "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
            "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
            "time_projection.1.bias": "condition_embedder.time_proj.bias",
            "time_projection.1.weight": "condition_embedder.time_proj.weight",
            "text_embedding.linear_1.weight": "condition_embedder.text_embedder.linear_1.weight",
            "text_embedding.linear_1.bias": "condition_embedder.text_embedder.linear_1.bias",
            "text_embedding.linear_2.bias": "condition_embedder.text_embedder.linear_2.bias",
            "text_embedding.linear_2.weight": "condition_embedder.text_embedder.linear_2.weight",
            "patch_embedding.weight": "patch_embedding.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "head.modulation": "scale_shift_table",
            "head.head.weight": "proj_out.weight",
            "head.head.bias": "proj_out.bias",


            "blocks.*.modulation": "blocks.*.scale_shift_table",
            "blocks.*.self_attn.proj_q.weight": "blocks.*.attn1.to_q.weight",
            "blocks.*.self_attn.proj_q.bias": "blocks.*.attn1.to_q.bias",
            "blocks.*.self_attn.proj_k.weight": "blocks.*.attn1.to_k.weight",
            "blocks.*.self_attn.proj_k.bias": "blocks.*.attn1.to_k.bias",
            "blocks.*.self_attn.proj_v.weight": "blocks.*.attn1.to_v.weight",
            "blocks.*.self_attn.proj_v.bias": "blocks.*.attn1.to_v.bias",
            "blocks.*.self_attn.q_norm.weight": "blocks.*.attn1.norm_q.weight",
            "blocks.*.self_attn.k_norm.weight": "blocks.*.attn1.norm_k.weight",
            "blocks.*.self_attn.proj_out.weight": "blocks.*.attn1.to_out.0.weight",
            "blocks.*.self_attn.proj_out.bias": "blocks.*.attn1.to_out.0.bias",
            "blocks.*.norm3.weight": "blocks.*.norm2.weight",
            "blocks.*.norm3.bias": "blocks.*.norm2.bias",
            "blocks.*.cross_attn.proj_q.weight": "blocks.*.attn2.to_q.weight",
            "blocks.*.cross_attn.proj_q.bias": "blocks.*.attn2.to_q.bias",
            "blocks.*.cross_attn.proj_k.weight": "blocks.*.attn2.to_k.weight",
            "blocks.*.cross_attn.proj_k.bias": "blocks.*.attn2.to_k.bias",
            "blocks.*.cross_attn.proj_v.weight": "blocks.*.attn2.to_v.weight",
            "blocks.*.cross_attn.proj_v.bias": "blocks.*.attn2.to_v.bias",
            "blocks.*.cross_attn.q_norm.weight": "blocks.*.attn2.norm_q.weight",
            "blocks.*.cross_attn.k_norm.weight": "blocks.*.attn2.norm_k.weight",
            "blocks.*.cross_attn.proj_out.weight": "blocks.*.attn2.to_out.0.weight",
            "blocks.*.cross_attn.proj_out.bias": "blocks.*.attn2.to_out.0.bias",
            "blocks.*.ffn.0.weight": "blocks.*.ffn.net.0.proj.weight",
            "blocks.*.ffn.0.bias": "blocks.*.ffn.net.0.proj.bias",
            "blocks.*.ffn.2.weight": "blocks.*.ffn.net.2.weight",
            "blocks.*.ffn.2.bias": "blocks.*.ffn.net.2.bias",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        return MegatronMappingRegistry(*mapping_list)
