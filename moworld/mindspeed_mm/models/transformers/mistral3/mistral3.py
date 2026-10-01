
# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

from transformers import Mistral3ForConditionalGeneration

from megatron.training import print_rank_0
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin, WeightInitMixin
from mindspeed_mm.models.transformers.custom_model_registry import register_model
from mindspeed_mm.models.transformers.mistral3.modules import MMMistralAttention
from mindspeed_mm.models.transformers.mistral3.modeling_mistral import MMMistralModel


@register_model("mistral3")
class MultiModelMistral3ForConditionalGeneration(Mistral3ForConditionalGeneration, FSDP2Mixin, WeightInitMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model.language_model = MMMistralModel(config=config.text_config)
        for idx, layer in enumerate(self.model.language_model.layers):
            layer.self_attn = MMMistralAttention(config=config.text_config, layer_idx=idx)

    def freeze(self, config):
        forbidden_modules = set()
        if config.vision_encoder.freeze:
            vision_model_keys = ["vision_tower"]
            print_rank_0(f"Set vision model not trainable: {vision_model_keys}")
            forbidden_modules.update(vision_model_keys)

        if config.vision_projector.freeze:
            projector_keys = ["multi_modal_projector"]
            print_rank_0(f"Set vision model not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["language_model", "lm_head"]
            print_rank_0(f"Set vision model not trainable: {language_model_keys}")
            forbidden_modules.update(language_model_keys)

        for name, param in self.model.named_parameters():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                param.requires_grad_(False)
