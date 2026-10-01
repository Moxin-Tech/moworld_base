import inspect
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import ModelOutput


@dataclass
class HunyuanMLLmModelOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class HunyuanMLLmModel(nn.Module):
    def __init__(
        self,
        model,
        template_info,
        image_embed_interleave=2,
    ):
        super().__init__()
        self.model = model.to(model.dtype)
        self.template_info = template_info
        self.image_embed_interleave = image_embed_interleave

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs
    ):
        crop_start = self.template_info.get("crop_start", None)
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True
        }
        if pixel_values is not None:
            model_kwargs["pixel_values"] = pixel_values.to(self.model.dtype)
        prompt_embeds = self.model(**model_kwargs).hidden_states[-(self.hidden_state_skip_layer + 1)]

        if pixel_values is None:
            if crop_start is not None and crop_start > 0:
                prompt_embeds = prompt_embeds[:, crop_start:]
                if attention_mask is not None:
                    attention_mask.set_(attention_mask[:, crop_start:].contiguous())
        else:
            image_emb_len = self.template_info.get("image_emb_len", 576)
            image_emb_start = self.template_info.get("image_emb_start", 5)
            image_emb_end = self.template_info.get("image_emb_end", 581)
            double_return_token_id = self.template_info.get("double_return_token_id", 271)
            if crop_start is not None and crop_start > 0:
                text_crop_start = crop_start - 1 + image_emb_len
                batch_indices, last_double_return_token_indices = torch.where(input_ids == double_return_token_id)

                if last_double_return_token_indices.shape[0] == 3:
                    last_double_return_token_indices = torch.cat(
                        (last_double_return_token_indices, torch.tensor([input_ids.shape[-1]]).to(device=last_double_return_token_indices.device)
                    )

                    )

                last_double_return_token_indices = last_double_return_token_indices.reshape(input_ids.shape[0], -1)[:, -1]

                assistant_crop_start = last_double_return_token_indices - 1 + image_emb_len - 4
                assistant_crop_end = last_double_return_token_indices - 1 + image_emb_len
                attention_mask_assistant_crop_start = last_double_return_token_indices - 4
                attention_mask_assistant_crop_end = last_double_return_token_indices

                prompt_embed_list = []
                prompt_attention_mask_list = []
                image_embed_list = []
                image_attention_mask_list = []

                for i in range(input_ids.shape[0]):
                    prompt_embed_list.append(
                        torch.cat(
                            (
                                prompt_embeds[i, text_crop_start:assistant_crop_start[i].item()],
                                prompt_embeds[i, assistant_crop_end[i].item():]
                            )
                        )
                    )
                    prompt_attention_mask_list.append(
                        torch.cat(
                            (
                                attention_mask[i, crop_start:attention_mask_assistant_crop_start[i].item()],
                                attention_mask[i, attention_mask_assistant_crop_end[i].item():]
                            )
                        )
                    )
                    image_embed_list.append(
                        prompt_embeds[i, image_emb_start:image_emb_end]
                    )
                    image_attention_mask_list.append(
                        torch.ones(image_embed_list[-1].shape[0]).to(prompt_embeds.device).to(attention_mask.dtype)
                    )

                prompt_embed_list = torch.stack(prompt_embed_list)
                prompt_attention_mask_list = torch.stack(prompt_attention_mask_list)
                image_embed_list = torch.stack(image_embed_list)
                image_attention_mask_list = torch.stack(image_attention_mask_list)

                if 0 < self.image_embed_interleave < 6:
                    image_embed_list = image_embed_list[:, ::self.image_embed_interleave, :]
                    image_attention_mask_list = image_attention_mask_list[:, ::self.image_embed_interleave]

                prompt_embeds = torch.cat((image_embed_list, prompt_embed_list), dim=1)
                prompt_attention_mask = torch.cat((image_attention_mask_list, prompt_attention_mask_list), dim=1)
                attention_mask.set_(prompt_attention_mask.contiguous())

        return HunyuanMLLmModelOutput(
            hidden_states=(prompt_embeds,) * (self.hidden_state_skip_layer + 1),
        )

    def __getattr__(self, name):
        if name in dir(self):
            return super().__getattr__(name)
        else:
            return getattr(self.model, name)

    @classmethod
    def from_pretrained(cls, **config):
        template_file_path = config.pop("template_file_path")
        template_id = config.pop("template_id", "hyv-llm-encode-video")
        with open(template_file_path, "r") as f:
            templates = json.load(f)
        image_embed_interleave = config.pop("image_embed_interleave", 4)
        model_type = config.pop("model_type", "AutoModel")
        model = getattr(transformers, model_type).from_pretrained(**config)
        return HunyuanMLLmModel(
            model=model,
            template_info=templates[template_id],
            image_embed_interleave=image_embed_interleave,
        )
