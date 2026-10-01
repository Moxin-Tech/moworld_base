# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
# Copyright 2025 The KwaiVGI team. All rights reserved.
from typing import Optional, Dict, Union
import torch
from torch import nn
import torch.nn.init as init

from megatron.core import InferenceParams
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region

from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import prepare_positionsids_mask_for_llm
from mindspeed_mm.models.vlm_model import VLMModel


class Qwen2VLRewardModelBT(VLMModel):
    def __init__(self, config, extra_config):
        super().__init__(config=config)
        self.output_dim = config.output_dim
        self.reward_token = config.reward_token
        self.loss_type = config.loss_type
        self.loss_dtype = torch.bfloat16 if getattr(config.text_decoder, 'bf16', False) else torch.float32
        self.use_remove_padding = getattr(config.text_decoder, 'use_remove_padding', False)
        self.rm_head = nn.Linear(config.text_decoder.hidden_size, self.output_dim, bias=False)

        self.pad_token_id = extra_config['pad_token_id']
        self.special_token_ids = extra_config['special_token_ids']
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            inference_params: Optional[InferenceParams] = None,
            decoder_input: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            packed_seq_params: Optional[PackedSeqParams] = None,
            extra_block_kwargs: Optional[dict] = None,
            cache_position: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            *args, **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if self.add_image_encoder and pixel_values is not None:
            pixel_values = pixel_values.to(self.loss_dtype)
            vit_embeds = self.image_encoder(pixel_values, image_grid_thw).to(pixel_values.dtype)

            if image_flags is not None:
                if self.image_encoder.post_process:
                    image_flags = image_flags.squeeze(-1)
                    vit_embeds = vit_embeds[image_flags == 1]
                    vit_embeds = vit_embeds.reshape(-1, 1, vit_embeds.shape[-1]).clone()
            else:
                vit_embeds = vit_embeds.reshape(-1, 1, vit_embeds.shape[-1]).clone()
            output = vit_embeds
        else:
            vit_embeds = self.input_tensor

        if self.add_text_decoder:
            input_embeds = None
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                _input_ids = input_ids
                if self.config.sequence_parallel:
                    _input_ids = scatter_to_sequence_parallel_region(_input_ids.transpose(0, 1)).transpose(0, 1)
                if vit_embeds is not None:
                    input_embeds = input_embeds.transpose(0, 1)  # bsh
                    image_mask = torch.eq(_input_ids, self.img_context_token_id).unsqueeze(-1).expand_as(input_embeds)
                    vit_embeds = vit_embeds[:, 0, :]
                    input_embeds = input_embeds.masked_scatter(image_mask, vit_embeds)
                    input_embeds = input_embeds.transpose(0, 1).clone()

            attention_mask, position_ids = prepare_positionsids_mask_for_llm(config=self.config,
                                                                             input_ids=input_ids,
                                                                             inference_params=inference_params,
                                                                             attention_mask=None if self.use_remove_padding else attention_mask,
                                                                             position_ids=position_ids,
                                                                             image_grid_thw=None,
                                                                             rope_deltas=rope_deltas,
                                                                             inputs_embeds=input_embeds,
                                                                             cache_position=cache_position,
                                                                             **kwargs)


            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None,
                inference_params=inference_params,
            )

            hidden_states = output.transpose(0, 1)
            logits = self.rm_head(hidden_states)  # [B, L, N]

            if input_ids is not None:
                batch_size = input_ids.shape[0]
            else:
                batch_size = input_embeds.shape[0]

            # get sequence length
            if self.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                    sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            # get the last token's logits
            if self.reward_token == "last":
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            elif self.reward_token == "mean":
                # get the mean of all valid tokens' logits
                valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
                pooled_logits = torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
            elif self.reward_token == "special":
                # create a mask for special tokens
                special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for special_token_id in self.special_token_ids:
                    special_token_mask = special_token_mask | (input_ids == special_token_id)
                pooled_logits = logits[special_token_mask, ...]
                pooled_logits = pooled_logits.view(batch_size, 3, -1)  # [B, 3, N] asslert 3 attributes
                if self.output_dim == 3:
                    pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
                pooled_logits = pooled_logits.view(batch_size, -1)
            else:
                raise ValueError("Invalid reward_token")
        return pooled_logits
