from typing import Any, List

from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)

from megatron.training import print_rank_0, get_args
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin
from mindspeed_mm.models.transformers.custom_model_registry import register_model
from .modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration as HFQwen3OmniMoeThinkerForConditionalGeneration
from ..attention_utils import verify_attn_layout


class Qwen3OmniFSDP2Mixin(FSDP2Mixin):
    """
    Mixin class for FSDP2 of the Qwen3Omni-series
    """

    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        # recompute
        for i, layer in enumerate(self.audio_tower.layers):
            self.audio_tower.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        for i, block in enumerate(self.visual.blocks):
            self.visual.blocks[i] = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)

        for i, layer in enumerate(self.model.layers):
            self.model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        # fully_shard
        fully_shard(self.audio_tower.positional_embedding, **fsdp2_kwargs)
        for layer in self.audio_tower.layers:
            fully_shard(layer, **fsdp2_kwargs)

        for block in self.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.visual.merger, **fsdp2_kwargs)
        for merger in self.visual.merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.visual, **fsdp2_kwargs)

        fully_shard(self.model.embed_tokens, **fsdp2_kwargs)
        for layer in self.model.layers:
            fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **fsdp2_kwargs)
        fully_shard(self, **fsdp2_kwargs)

        # forward prefetch
        if fsdp2_config.num_to_forward_prefetch > 0:
            for curr_layer, next_layer in zip(self.audio_tower.layers[:-1], self.audio_tower.layers[1:]):
                curr_layer.set_modules_to_forward_prefetch([next_layer])
            self.audio_tower.layers[-1].set_modules_to_forward_prefetch([self.visual.blocks[0]])

            for i, (curr_block, next_block) in enumerate(zip(self.visual.blocks[:-1], self.visual.blocks[1:])):
                prefetch_modules: List[nn.Module] = []
                if i in self.visual.deepstack_visual_indexes:
                    prefetch_modules.append(self.visual.deepstack_merger_list[self.visual.deepstack_visual_indexes.index(i)])
                prefetch_modules.append(next_block)
                curr_block.set_modules_to_forward_prefetch(prefetch_modules)
            self.visual.blocks[-1].set_modules_to_forward_prefetch([self.visual.merger])
            self.visual.merger.set_modules_to_forward_prefetch([self.model.embed_tokens])

            self.model.embed_tokens.set_modules_to_forward_prefetch([self.model.layers[0]])
            for curr_layer, next_layer in zip(self.model.layers[:-1], self.model.layers[1:]):
                curr_layer.set_modules_to_forward_prefetch([next_layer])
            self.model.layers[-1].set_modules_to_forward_prefetch([self.lm_head])

    def freeze(self, config):
        forbidden_modules = set()
        if config.image_encoder.vision_encoder.freeze:
            vision_model_keys = ['visual.patch_embed', 'visual.blocks', 'visual.merger_list', 'visual.pos_embed']
            print_rank_0(f"Set vision model not trainable: {vision_model_keys}")
            forbidden_modules.update(vision_model_keys)

        if config.image_encoder.vision_projector.freeze:
            projector_keys = ["visual.merger"]
            print_rank_0(f"Set multi model projector not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.audio_encoder.audio_encoder.freeze:
            projector_keys = ["audio_tower"]
            print_rank_0(f"Set audio model not trainable: {projector_keys}")
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["model", "lm_head"]
            print_rank_0(f"Set language model not trainable: {language_model_keys}")
            forbidden_modules.update(language_model_keys)

        for name, param in self.named_parameters():
            if any(forbidden_module in name for forbidden_module in forbidden_modules):
                param.requires_grad_(False)

    @staticmethod
    def overwrite_transformer_config(transformer_config: Any) -> Any:
        args = get_args()
        model_cfg = args.mm.model

        # attn_implementation support eager, sdpa, flash_attention_2, default flash_attention_2
        vit_attn_implementation = getattr(model_cfg.image_encoder.vision_encoder, "attn_implementation", "flash_attention_2")
        aud_attn_implementation = getattr(model_cfg.audio_encoder.audio_encoder, "attn_implementation", "flash_attention_2")
        llm_attn_implementation = getattr(model_cfg.text_decoder, "attn_implementation", "flash_attention_2")

        # layout support BNSD, TND, default TND
        vit_attn_layout = getattr(model_cfg.image_encoder.vision_encoder, "attn_layout", "TND")
        aud_attn_layout = getattr(model_cfg.audio_encoder.audio_encoder, "attn_layout", "TND")
        # layout support BNSD, TND, BSND, default TND
        llm_attn_layout = getattr(model_cfg.text_decoder, "attn_layout", "TND")

        verify_attn_layout(vit_attn_implementation, vit_attn_layout)
        verify_attn_layout(aud_attn_implementation, aud_attn_layout)
        verify_attn_layout(llm_attn_implementation, llm_attn_layout)

        # set attn type configuration, layout configuration
        vision_cfg = getattr(transformer_config.thinker_config, "vision_config", None)
        audio_cfg = getattr(transformer_config.thinker_config, "audio_config", None)
        text_cfg = getattr(transformer_config.thinker_config, "text_config", None)

        if vision_cfg is not None:
            setattr(vision_cfg, "_attn_implementation", vit_attn_implementation)
            setattr(vision_cfg, "attn_layout", vit_attn_layout)

        if audio_cfg is not None:
            setattr(audio_cfg, "_attn_implementation", aud_attn_implementation)
            setattr(audio_cfg, "attn_layout", aud_attn_layout)

        if text_cfg is not None:
            setattr(text_cfg, "_attn_implementation", llm_attn_implementation)
            setattr(text_cfg, "attn_layout", llm_attn_layout)

            # ---- activation offload (text only) ----
            llm_activation_offload = getattr(model_cfg.text_decoder, "activation_offload", False)
            setattr(text_cfg, "activation_offload", llm_activation_offload)

        return transformer_config


@register_model("qwen3_omni_moe")
class Qwen3OmniMoeThinkerForConditionalGeneration(HFQwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniFSDP2Mixin):
    pass
