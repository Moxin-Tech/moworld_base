# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

import transformers
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl
)
from megatron.training import print_rank_0, get_args
from mindspeed_mm.models.transformers.base_model import FSDP2Mixin

from mindspeed_mm.models.transformers.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration
from mindspeed_mm.models.transformers.qwen3vl.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration as HFQwen3VLMoeForConditionalGeneration
from mindspeed_mm.models.transformers.custom_model_registry import register_model
from ..attention_utils import verify_attn_layout


class Qwen3VLFSDP2Mixin(FSDP2Mixin):
    """
    Mixin class for FSDP2 of the Qwen3VL-series
    """
    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        # recompute
        for i, block in enumerate(self.model.visual.blocks):
            self.model.visual.blocks[i] = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)

        for i, layer in enumerate(self.model.language_model.layers):
            self.model.language_model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)

        last_module_kwargs = fsdp2_kwargs.copy()
        last_module_kwargs["reshard_after_forward"] = False

        # fully_shard
        for block in self.model.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.model.visual.merger, **fsdp2_kwargs)
        for merger in self.model.visual.deepstack_merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.model.visual, **fsdp2_kwargs)

        if fsdp2_config.align_fsdp_param_groups:
            # Each FSDP parameter group within a block contains only Linear layer parameters,
            # enabling aligned sharding for improved communication efficiency.
            norm_gate_params = []
            norm_gate_params.append(self.model.language_model.norm_hook_module)
            # Group all layers norm and gate parameters into a separate FSDP parameter group.
            for layer in self.model.language_model.layers:
                norm_gate_params.append(layer.input_layernorm)
                norm_gate_params.append(layer.self_attn.q_norm)
                norm_gate_params.append(layer.self_attn.k_norm)
                norm_gate_params.append(layer.post_attention_layernorm)
                # moe
                if hasattr(layer.mlp, "gate"):
                    norm_gate_params.append(layer.mlp.gate)
            norm_gate_params.append(self.model.language_model.norm)

            fully_shard(norm_gate_params, **last_module_kwargs)

        llm_num_layers = len(self.model.language_model.layers)
        fully_shard(self.model.language_model.embed_tokens, **fsdp2_kwargs)
        for idx, layer in enumerate(self.model.language_model.layers):
            if idx == (llm_num_layers - 1) and fsdp2_config.num_to_forward_prefetch > 0:
                # Skip resharding after forward for the last layer if prefetching is enabled
                fully_shard(layer, **last_module_kwargs)
            else:
                fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **last_module_kwargs)
        fully_shard(self, **fsdp2_kwargs)

        # prefetch
        if fsdp2_config.num_to_forward_prefetch > 0:
            for i, (curr_block, next_block) in enumerate(zip(self.model.visual.blocks[:-1], self.model.visual.blocks[1:])):
                prefetch_modules = []
                if i in self.model.visual.deepstack_visual_indexes:
                    prefetch_modules.append(self.model.visual.deepstack_merger_list[self.model.visual.deepstack_visual_indexes.index(i)])
                prefetch_modules.append(next_block)
                curr_block.set_modules_to_forward_prefetch(prefetch_modules)

            self.model.visual.blocks[-1].set_modules_to_forward_prefetch([self.model.visual.merger])
            self.model.visual.merger.set_modules_to_forward_prefetch([self.model.language_model.embed_tokens])
            self.model.language_model.embed_tokens.set_modules_to_forward_prefetch([self.model.language_model.layers[0]])

            for curr_layer, next_layer in zip(self.model.language_model.layers[:-1], self.model.language_model.layers[1:]):
                curr_layer.set_modules_to_forward_prefetch([next_layer])
            self.model.language_model.layers[-1].set_modules_to_forward_prefetch([self.lm_head])

    def freeze(self, config):
        forbidden_modules = set()
        if config.image_encoder.vision_encoder.freeze:
            vision_model_keys = ["visual.patch_embed", "visual.blocks", "visual.pos_embed"]
            forbidden_modules.update(vision_model_keys)

        if config.image_encoder.vision_projector.freeze:
            projector_keys = ["visual.merger"]
            forbidden_modules.update(projector_keys)

        if config.text_decoder.freeze:
            language_model_keys = ["language_model", "lm_head"]
            forbidden_modules.update(language_model_keys)

        # modules that are finally frozen
        final_forbidden_modules = set()
        for name, param in self.named_parameters():
            for forbidden_module in forbidden_modules:
                if forbidden_module in name:
                    param.requires_grad_(False)
                    final_forbidden_modules.add(forbidden_module)
        print_rank_0(f"Set modules not trainable: {final_forbidden_modules}")

    @staticmethod
    def overwrite_transformer_config(transformer_config):
        args = get_args()
        model_cfg = args.mm.model

        # attn_implementation support eager, sdpa(layout BNSD), flash_attention_2(layout BNSD), var_len_fa(layout TND), default flash_attention_2
        vit_attn_implementation = getattr(model_cfg.image_encoder.vision_encoder, "attn_implementation", "flash_attention_2")
        llm_attn_implementation = getattr(model_cfg.text_decoder, "attn_implementation", "flash_attention_2")
        # set attn type configuration
        transformer_config.vision_config._attn_implementation = vit_attn_implementation
        transformer_config.text_config._attn_implementation = llm_attn_implementation
        # set attn layout configuration
        vit_attn_layout = getattr(model_cfg.image_encoder.vision_encoder, "attn_layout", "TND")
        llm_attn_layout = getattr(model_cfg.text_decoder, "attn_layout", "TND")
        verify_attn_layout(vit_attn_implementation, vit_attn_layout)
        verify_attn_layout(llm_attn_implementation, llm_attn_layout)
        setattr(transformer_config.vision_config, "attn_layout", vit_attn_layout)
        setattr(transformer_config.text_config, "attn_layout", llm_attn_layout)
        # set attn mask type
        llm_is_causal = getattr(model_cfg.text_decoder, "is_causal", False)
        setattr(transformer_config.text_config, "is_causal", llm_is_causal)

        # set synchronize per layer, for memory reuse between streams when using FSDP2
        vit_synchronize_per_layer = getattr(model_cfg.image_encoder.vision_encoder, "synchronize_per_layer", False)
        llm_synchronize_per_layer = getattr(model_cfg.text_decoder, "synchronize_per_layer", False)
        setattr(transformer_config.vision_config, "synchronize_per_layer", vit_synchronize_per_layer)
        setattr(transformer_config.text_config, "synchronize_per_layer", llm_synchronize_per_layer)

        # set activation offload, only suppport offload text activations now
        llm_activation_offload = getattr(model_cfg.text_decoder, "activation_offload", False)
        setattr(transformer_config.text_config, "activation_offload", llm_activation_offload)

        # set router_aux_loss_coef, for moe model
        router_aux_loss_coef = getattr(model_cfg.loss_cfg, "router_aux_loss_coef", 0.0)
        transformer_config.text_config.router_aux_loss_coef = router_aux_loss_coef

        # set encoder mbs data balance for qwen3vl moe model
        transformer_config.vision_config.use_image_mbs_data_balance = args.use_image_mbs_data_balance
        transformer_config.vision_config.mbs_data_balance_sorting_algo = args.mbs_data_balance_sorting_algo

        return transformer_config


@register_model("qwen3_vl")
class Qwen3VLForConditionalGeneration(HFQwen3VLForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass


@register_model("qwen3_vl_moe")
class Qwen3VLMoeForConditionalGeneration(HFQwen3VLMoeForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass
