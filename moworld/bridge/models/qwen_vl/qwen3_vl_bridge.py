import torch
from megatron.core.transformer.module import MegatronModule
from bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    WeightReshapeMapping,
)
from mindspeed_mm.models.vlm_model import VLMModel


class Qwen3VLForConditionalGeneration():
    pass


@MegatronModelBridge.register_bridge(source=Qwen3VLForConditionalGeneration, target=VLMModel)
class Qwen3VLBridge(MegatronModelBridge):

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="model.lm_head.weight",
                    hf_param="lm_head.weight",
                ),
                ReplicatedMapping(
                    megatron_param="model.model.**",
                    hf_param="model.**",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


class Qwen3VLMoeForConditionalGeneration():
    pass


@MegatronModelBridge.register_bridge(source=Qwen3VLMoeForConditionalGeneration, target=VLMModel)
class Qwen3VLMoEBridge(MegatronModelBridge):

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="model.lm_head.weight",
                    hf_param="lm_head.weight",
                ),
                WeightReshapeMapping(
                    megatron_param="model.model.language_model.layers.*.mlp.experts.down_proj",
                    hf_param="model.language_model.layers.*.mlp.experts.down_proj",
                ),
                WeightReshapeMapping(
                    megatron_param="model.model.language_model.layers.*.mlp.experts.gate_up_proj",
                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
                ),
                ReplicatedMapping(
                    megatron_param="model.model.**",
                    hf_param="model.**",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
