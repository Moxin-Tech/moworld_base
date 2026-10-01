# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.masking_utils import create_causal_mask

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import split_forward_gather_backward_with_cp, gather_forward_split_backward
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes, generate_ulysses_cu_seqlen_params

def shift_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    shifted = torch.roll(tensor, shifts=shifts, dims=dims)
    shifted.select(dims, shifts).fill_(fill_value)
    return shifted


class MultiTokenPredictionBlock(nn.Module):
    def __init__(
        self,
        config,
        layer_cls,
        norm_cls,
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            layer_cls(config, i, is_mtp=True)
            for i in range(config.mtp_num_layers)
        ])
        self.pre_fc_norm_embedding = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.LongTensor],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor]
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        return position_ids, text_position_ids, cache_position

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        embed_tokens: Optional[nn.Module] = None,
        rotary_emb: Optional[nn.Module] = None,
        output_layer: Optional[nn.Module] = None,
        loss_function: Optional[Callable] = None,
        logits_to_keep: int | torch.Tensor = 0,
        seq_len: Optional[int] = 1,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if not self.layers:
            return None
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        all_mtp_loss = None
        for decoder_layer in self.layers:
            input_ids = shift_tensor(input_ids, shifts=-1, dims=-1)
            labels = shift_tensor(labels, shifts=-1, dims=-1, fill_value=-100)
            inputs_embeds = embed_tokens(input_ids)

            position_ids, text_position_ids, cache_position = self._prepare_position_ids(
                position_ids, inputs_embeds, past_key_values, cache_position
            )

            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
            ps = get_parallel_state()
            if ps.is_ulysses_enable():
                kwargs.update(generate_ulysses_cu_seqlen_params(text_position_ids))
                position_ids = split_forward_gather_backward_with_cp(position_ids, dim=2)
                text_position_ids = split_forward_gather_backward_with_cp(text_position_ids, dim=1)
                inputs_embeds = split_forward_gather_backward_with_cp(inputs_embeds, dim=1)

            position_embeddings = rotary_emb(hidden_states, position_ids)

            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = self.norm(hidden_states)

            if getattr(self.config, "enable_chunk_loss", False) or getattr(self.config, "enable_dynamic_chunk_loss", False):
                mtp_loss = output_layer(hidden_states[:, slice_indices, :], loss_function, labels)
                if all_mtp_loss is None:
                    all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)
            else:
                mtp_logits = output_layer(hidden_states[:, slice_indices, :])
                if labels is not None:
                    mtp_loss = loss_function(mtp_logits, labels, vocab_size=self.config.vocab_size)
                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)
            if ps.is_ulysses_enable():
                gather_sizes = cal_split_sizes(seq_len, ps.get_ulysses_group_size())
                position_ids = gather_forward_split_backward(position_ids, ps.get_ulysses_group(), dim=2, grad_scale="up", gather_sizes=gather_sizes)
                text_position_ids = gather_forward_split_backward(text_position_ids, ps.get_ulysses_group(), dim=1, grad_scale="up", gather_sizes=gather_sizes)

        if all_mtp_loss is not None and ps.is_cp_enable():
            for i in range(len(all_mtp_loss)):
                all_mtp_loss[i] = gather_forward_split_backward(all_mtp_loss[i].unsqueeze(0), ps.get_cp_group(), dim=0)
                all_mtp_loss[i] = all_mtp_loss[i].sum()
        return all_mtp_loss
