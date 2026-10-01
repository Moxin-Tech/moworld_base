from typing import Optional

import torch
from transformers import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    MistralModel
)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, TransformersKwargs
from transformers.utils.generic import check_model_inputs

from megatron.core import mpu
from mindspeed_mm.models.common.communications import split_forward_gather_backward_with_cp
from mindspeed_mm.models.transformers.cp_utils import set_seq_len


@auto_docstring
class MMMistralModel(MistralModel):
    @check_model_inputs
    @auto_docstring
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Ulysses Context Parallelism (CP) implementation
        # Split the sequence dimension across context parallel group during forward pass
        # and gather the results back during backward pass to reduce memory usage
        total_seq_len = inputs_embeds.shape[1]
        set_seq_len("total", total_seq_len)
        if mpu.get_context_parallel_world_size() > 1:
            inputs_embeds = split_forward_gather_backward_with_cp(inputs_embeds, dim=1)
            position_ids = split_forward_gather_backward_with_cp(position_ids, dim=1)
        # ---------ulysses-cp------------

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )
