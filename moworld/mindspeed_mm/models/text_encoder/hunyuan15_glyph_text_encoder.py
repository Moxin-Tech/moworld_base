from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import ModelOutput

from mindspeed_mm.models.text_encoder.hunyuan15_glyph_tokenizer import Hunyuan15GlyphTokenizer


@dataclass
class HunyuanMLLmModelOutput(ModelOutput):
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Hunyuan15GlyphModel(nn.Module):
    def __init__(
            self,
            model,
            image_embed_interleave=2,
            color_ann_path=None,
            font_ann_path=None,
            byt5_max_length=None
    ):
        super().__init__()
        self.model = model.to(model.dtype)
        self.image_embed_interleave = image_embed_interleave
        self.color_ann_path = color_ann_path
        self.font_ann_path = font_ann_path
        self.byt5_max_length = byt5_max_length

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            **kwargs
    ):
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=self.model.device)
        if not torch.equal(input_ids, torch.zeros(input_ids.shape, dtype=input_ids.dtype, device=input_ids.device)):
            byt5_outputs = self.model(input_ids, attention_mask=attention_mask.float())
            byt5_embeddings = byt5_outputs[0]

        return HunyuanMLLmModelOutput(
            hidden_states=(byt5_embeddings,) * (self.hidden_state_skip_layer + 1),
        )

    def __getattr__(self, name):
        if name in dir(self):
            return super().__getattr__(name)
        else:
            return getattr(self.model, name)

    @classmethod
    def from_pretrained(cls, **config):
        image_embed_interleave = config.pop("image_embed_interleave", 4)
        model_type = config.pop("model_type", "T5ForConditionalGeneration")
        pretrained_model_name_or_path = config.get("pretrained_model_name_or_path", None)
        model = getattr(transformers, model_type).from_pretrained(pretrained_model_name_or_path,
                                                                  cache_dir=None).get_encoder()
        color_ann_path = config.get("color_ann_path", "")
        font_ann_path = config.get("font_ann_path", "")
        byt5_max_length = config.get("byt5_max_length", 256)

        Hunyuan15GlyphTokenizer.add_special_token(
            tokenizer=None,
            text_encoder=model,
            add_color=True,
            add_font=True,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            multilingual=True,
            token_add=False,
        )

        # Load custom checkpoint if provided
        ckpt_path = config.get('byT5_ckpt_path', None)
        if ckpt_path is not None:
            byt5_state_dict = torch.load(ckpt_path, map_location=model.device)
            if 'state_dict' in byt5_state_dict:
                sd = byt5_state_dict["state_dict"]
                newsd = {}
                for k, v in sd.items():
                    if k.startswith('module.text_tower.encoder.'):
                        newsd[k[len('module.text_tower.encoder.'):]] = v
                byt5_state_dict = newsd
            model.load_state_dict(byt5_state_dict)
        model.requires_grad_(False)

        return Hunyuan15GlyphModel(
            model=model,
            image_embed_interleave=image_embed_interleave,
            color_ann_path=color_ann_path,
            font_ann_path=font_ann_path,
            byt5_max_length=byt5_max_length,
        )
