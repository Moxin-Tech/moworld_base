import importlib

import torch
import torch_npu
import torch.nn as nn

from mindspeed_mm.models.text_encoder.hunyuan_mllm_text_encoder import HunyuanMLLmModel
from mindspeed_mm.models.text_encoder.hunyuan15_mllm_text_encoder import Hunyuan15MLLmModel
from mindspeed_mm.models.text_encoder.hunyuan15_glyph_text_encoder import Hunyuan15GlyphModel
from mindspeed_mm.models.text_encoder.stepllm_text_encoder import StepLLmModel
from mindspeed_mm.utils.utils import get_dtype

TRANSFORMERS_TEXT_ENCODER_MAPPING = {
    "T5": "T5EncoderModel",
    "MT5": "MT5EncoderModel",
    "UMT5": "UMT5EncoderModel",
    "CLIP": "CLIPTextModel",
    "Auto": "AutoModel",
    "BertModel": "BertModel",
    "CLIPWithProjection": "CLIPTextModelWithProjection",
}

CUSTOM_TEXT_ENCODER_MAPPING = {
    "StepLLmModel": StepLLmModel,
    "HunyuanMLLmModel": HunyuanMLLmModel,
    "Hunyuan15MLLmModel": Hunyuan15MLLmModel,
    "Hunyuan15GlyphModel": Hunyuan15GlyphModel,
}


class TextEncoder(nn.Module):
    """
    Configuration for initializing one or more Text Encoder Model instances.

    Args:
        config (Optional[dict, list(dict)]): the general config for Text Encoder Model.
        - If `config` is a dictionary, it specifies the parameters for a single Text Encoder Model instance.
            e.g.
            {
                (1) args for our feautrues
                "backend": type-str, "hf" or "om",
                "model_id": type-str, "AutoModel" or other automodel name,
                "dtype": type-str, dtype of text encoder

                (2) args for automodel.from_pretrained() of transformers or openmind
                "pretrained_model_name_or_path": type-str, local path or hub path,
                "local_files_only": type-bool,
                ...
            }
        - If `config` is a list of dictionaries, each dictionary in the list will be used to instantiate a separate Text Encoder Model instance,
            effectively allowing the creation of multiple Text Encoder based on different configurations.
    """

    def __init__(self, config):
        super().__init__()

        if isinstance(config, list) or isinstance(config, tuple):
            self.text_encoders = nn.ModuleList()
            for config_i in config:
                text_encoder_i = self._init_text_encoder(config_i)
                self.text_encoders.append(text_encoder_i)
        else:
            self.text_encoders = self._init_text_encoder(config)

    def get_model(self):
        return self.text_encoders

    def encode(self, input_ids, mask, offload_cpu=False, **kwargs):
        """
        Encode input sequences with encoder-wise offload in different scenarios.

        Scenarios:
        1. Single encoder + single input tensor -> (Tensor outputs, Tensor masks)
        2. Multiple encoders + single-step inputs (List[Tensor]) -> (List[Tensor] outputs, List[Tensor] masks)
        (Each encoder processes its corresponding input tensor)
        3. Single/multiple encoders + multi-step inputs (List[List[Tensor]]) -> (List[List[Tensor]] outputs, List[List[Tensor]] masks)
        (Each encoder processes its corresponding input tensor in each step)

        Args:
            input_ids (Union[Tensor, List[Tensor], List[List[Tensor]]]): Input token IDs matching scenario structures
            mask (Union[Tensor, List[Tensor], List[List[Tensor]]]): Attention masks matching input_ids structure
            **kwargs: Contains 'offload_cpu' (bool, optional) - only used in scenario 3

        Returns:
            Tuple[Union[Tensor, List[Tensor], List[List[Tensor]]], Union[Tensor, List[Tensor], List[List[Tensor]]]]:
                - outputs: Encoded outputs matching input structure
                - masks: Corresponding attention masks
        """
        encoders = self.text_encoders
        is_multi_encoder = isinstance(encoders, nn.ModuleList)
        num_encoders = len(encoders) if is_multi_encoder else 1

        # Scenario 1: Single encoder + single input tensor
        if not is_multi_encoder and isinstance(input_ids, torch.Tensor):
            return self._single_encode(encoders, input_ids, mask, **kwargs)

        # Scenario 2: Multiple encoders + single-step inputs (List[Tensor])
        if is_multi_encoder and isinstance(input_ids, list) and not isinstance(input_ids[0], list):
            outputs, masks = [], []
            for encoder, inp, msk in zip(encoders, input_ids, mask):
                out, att = self._single_encode(encoder, inp, msk, **kwargs)
                outputs.append(out)
                masks.append(att)
            return outputs, masks

        # Scenario 3: Multiple encoders + multi-step inputs (List[List[Tensor]])
        num_steps = len(input_ids)
        if not is_multi_encoder:
            encoders = [encoders]

        # Initialize outputs/masks with [num_steps][num_encoders] structure
        outputs = [[None for _ in range(num_encoders)] for _ in range(num_steps)]
        result_masks = [[None for _ in range(num_encoders)] for _ in range(num_steps)]

        # Process each encoder sequentially (handle all steps for current encoder)
        for encoder_idx, encoder in enumerate(encoders):
            # Move encoder to device once before processing all steps
            if offload_cpu:
                encoder.to(torch_npu.npu.current_device())

            # Process all steps for current encoder
            for step in range(num_steps):
                # Get the input/mask for this encoder in current step
                step_input = input_ids[step][encoder_idx]
                step_mask = mask[step][encoder_idx]
                out, att = self._single_encode(encoder, step_input, step_mask, **kwargs)
                outputs[step][encoder_idx] = out  # Fill [step][encoder] position
                result_masks[step][encoder_idx] = att  # Fill masks correspondingly

            # Offload encoder after processing all steps
            if offload_cpu:
                encoder.to(torch.device('cpu'))
                torch_npu.npu.empty_cache()

        return outputs, result_masks

    def _single_encode(self, text_encoder, input_ids, attention_mask, **kwargs):
        *BN, L = input_ids.shape
        input_ids = input_ids.to(text_encoder.device).view(-1, L)
        attention_mask = attention_mask.to(text_encoder.device).view(-1, L)
        model_attention_mask = attention_mask if text_encoder.use_attention_mask else None
        model_kwargs = {}
        if text_encoder.using_kwargs:
            for k in text_encoder.using_kwargs:
                if k in kwargs.keys():
                    model_kwargs[k] = kwargs[k]
                else:
                    raise ValueError(f"{k} is not in kwargs")
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            output_hidden_states=text_encoder.hidden_state_skip_layer is not None,
            **model_kwargs
        )

        emb = output[text_encoder.output_key]
        if text_encoder.hidden_state_skip_layer:
            emb = emb[-(text_encoder.hidden_state_skip_layer + 1)]

        if text_encoder.ucg_rate is not None and text_encoder.ucg_rate > 0.0:
            def expand_dims_like(x, y):
                while x.dim() != y.dim():
                    x = x.unsqueeze(-1)
                return x

            emb = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - text_encoder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device, dtype=emb.dtype)),
                    emb,
                )
                * emb
            )

        if text_encoder.output_key in ["last_hidden_state", "hidden_states"]:
            emb = emb.view(*BN, emb.shape[-2], -1)
        elif text_encoder.output_key in ["pooler_output", "text_embeds"]:
            emb = emb.view(*BN, -1)
        else:
            raise NotImplementedError(f"Text encoder output_key: {text_encoder.output_key} is not implemented! ")

        if text_encoder.use_attention_mask:
            attention_mask = model_attention_mask
        attention_mask = attention_mask.view(*BN, -1)

        return emb, attention_mask

    def _init_text_encoder(self, config):
        if not isinstance(config, dict):
            config = config.to_dict()

        backend = config.pop("hub_backend")
        use_attention_mask = config.pop("use_attention_mask", True)
        ucg_rate = config.pop("ucg_rate", None)
        output_key = config.pop("output_key", "last_hidden_state")
        hidden_state_skip_layer = config.pop("hidden_state_skip_layer", None)
        using_kwargs = config.pop("using_kwargs", None)

        config["pretrained_model_name_or_path"] = config.pop("from_pretrained")
        config["torch_dtype"] = get_dtype(config.pop("dtype"))
        config["local_files_only"] = True
        try:
            from megatron.training import get_args
            config["trust_remote_code"] = get_args().trust_remote_code
        except (ImportError, AssertionError, AttributeError):
            config["trust_remote_code"] = False

        # Only huggingface backend is supported, OpenMind backend will be supported soon.
        model_id = config.pop("model_id")
        # Adapted for transformers 5.0.0.
        config.pop("load_in_8bit", None)
        if model_id in TRANSFORMERS_TEXT_ENCODER_MAPPING:
            module = importlib.import_module("transformers")
            self.automodel_name = TRANSFORMERS_TEXT_ENCODER_MAPPING[model_id]
            automodel = getattr(module, self.automodel_name)
            text_encoder = automodel.from_pretrained(**config)
        elif model_id in CUSTOM_TEXT_ENCODER_MAPPING:
            automodel = CUSTOM_TEXT_ENCODER_MAPPING[model_id]
            text_encoder = automodel.from_pretrained(**config)
        else:
            raise ValueError(f"Model ID {model_id} is not supported for text encoder")

        setattr(text_encoder, "ucg_rate", ucg_rate)
        setattr(text_encoder, "use_attention_mask", use_attention_mask)
        setattr(text_encoder, "output_key", output_key)
        setattr(text_encoder, "hidden_state_skip_layer", hidden_state_skip_layer)
        setattr(text_encoder, "using_kwargs", using_kwargs)

        if hidden_state_skip_layer and output_key not in ["hidden_states"]:
            raise ValueError("If use hidden_state_skip, the output_keys must in [`hidden_states`]")
        return text_encoder
