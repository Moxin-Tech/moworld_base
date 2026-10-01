import json
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import transformers
from transformers.modeling_outputs import ModelOutput

from mindspeed_mm.models.text_encoder.hunyuan_mllm_text_encoder import HunyuanMLLmModel


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None
    image_features: Optional[list] = None


class Hunyuan15MLLmModel(HunyuanMLLmModel):
    def __init__(
            self,
            model,
            template_info,
            image_embed_interleave=2,
    ):
        super().__init__(model=model, template_info=template_info, image_embed_interleave=image_embed_interleave)

    @classmethod
    def from_pretrained(cls, **config):
        template_file_path = config.pop("template_file_path")
        template_id = config.pop("template_id", "hyv-llm-encode-video")
        with open(template_file_path, "r") as f:
            templates = json.load(f)
        image_embed_interleave = config.pop("image_embed_interleave", 4)
        model_type = config.pop("model_type", "AutoModel")
        model = getattr(transformers, model_type).from_pretrained(**config)
        if hasattr(model, 'language_model'):
            model = model.language_model
        model.final_layer_norm = model.norm
        # from_pretrained will ensure that the model is in eval mode.
        model.requires_grad_(False)
        return Hunyuan15MLLmModel(
            model=model,
            template_info=templates[template_id],
            image_embed_interleave=image_embed_interleave,
        )

    def encode(
            self,
            batch_encoding,
            use_attention_mask=True,
            output_hidden_states=False,
            hidden_state_skip_layer=2,
            device=None,
            use_template=True,
            apply_final_norm=False,
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        attention_mask = (
            batch_encoding["attention_mask"].to(device) if use_attention_mask else None
        )
        outputs = self.model(
            input_ids=batch_encoding["input_ids"].to(device),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
                                 or hidden_state_skip_layer is not None,
        )
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        crop_start = self.template_info.get("crop_start", None)
        if use_template and crop_start > 0:
            last_hidden_state = last_hidden_state[:, crop_start:]
            attention_mask = (
                attention_mask[:, crop_start:] if use_attention_mask else None
            )

        if output_hidden_states:
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, outputs.hidden_states
            )
        return TextEncoderModelOutput(last_hidden_state, attention_mask)
