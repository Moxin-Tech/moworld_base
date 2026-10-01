# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from typing import List, Optional, Tuple, Union

import torch
import transformers

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
)

from .configuration_internvlu_chat import InternVLUChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .constants import SPECIAL_TOKEN_LIST

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    """Compare two version strings using a provided operator."""
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLUChatModel(PreTrainedModel):
    """Multimodal chat model combining a vision encoder with a language model."""

    config_class = InternVLUChatConfig
    main_input_name = "pixel_values"
    base_model_prefix = ""
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ""

    def __init__(
        self,
        config: InternVLUChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=None,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.patch_aspect_ratio = 1.0
        self.ps_version = config.ps_version
        use_flash_attn = getattr(config.vision_config, "use_flash_attn", True) if use_flash_attn is None else use_flash_attn
        use_flash_attn = bool(use_flash_attn) and has_flash_attn
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == "Qwen2ForCausalLM":
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == "Qwen3MoeForCausalLM":
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == "Qwen3ForCausalLM":
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f"{architecture} is not implemented.")

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.im_start_token_id = None
        self.im_end_token_id = None
        self.img_context_token_id = None
        self.img_start_token_id = None
        self.img_end_token_id = None
        self.img_uncond_token_id = None
        self.img_line_break_token_id = None
        self.img_frame_break_token_id = None
        self.pad_token_id = None
        self.conv_template = get_conv_template(self.template)

        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message

        ##### ---- Special token embeddings ---- #####
        self.special_token_embedding = nn.Embedding(
            len(SPECIAL_TOKEN_LIST), config.llm_config.hidden_size
        )
        self.special_token_list = copy.deepcopy(SPECIAL_TOKEN_LIST)
        self.special_token_id_list = None  # Remember to initialize this in the training script after tokenizer is loaded

    def replace_img_special_tokens(self, input_embeds, input_ids):
        assert (
            self.special_token_id_list is not None
        ), "model's special_token_id_list is not initialized"
        for i, token_id in enumerate(self.special_token_id_list):
            token_pos = input_ids == token_id
            input_embeds[token_pos] = (
                input_embeds[token_pos] * 0.0 + self.special_token_embedding.weight[i]
            )

        return input_embeds

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        padding_type: Optional[str] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)

        if video_grid_thw is not None:
            grid_thw = video_grid_thw
        else:
            grid_thw = image_grid_thw

        vit_embeds = self.extract_feature(pixel_values, grid_thw)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = (
                input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]
            )

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            padding_type=padding_type,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle_v2(self, x, scale_factor=0.5, patch_aspect_ratio=1.0):
        # input shape: N, L, C or N, H, W, C
        # output shape: N, L * (scale_factor ** 2), C / (scale_factor ** 2)

        if x.ndim == 3:
            n, l, c = x.size()
            h = w = int(l**0.5)
            # N, L, C --> N, H, W, C
            x = x.reshape(n, h, w, c)

        n, h, w, c = x.size()

        h_scale_factor = scale_factor * (patch_aspect_ratio**0.5)
        w_scale_factor = scale_factor / (patch_aspect_ratio**0.5)

        # N, H, W, C --> N, H, W * w_scale_factor, C // w_scale_factor
        x = x.reshape(n, h, int(w * w_scale_factor), int(c / w_scale_factor))
        # N, H, W * w_scale_factor, C // w_scale_factor --> N, W * w_scale_factor, H, C // w_scale_factor
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, W * w_scale_factor, H, C // w_scale_factor -->
        # N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor)
        x = x.reshape(
            n,
            int(w * w_scale_factor),
            int(h * h_scale_factor),
            int(c / (w_scale_factor * h_scale_factor)),
        )
        # N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor) -->
        # N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor)
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor) -->
        # N, L * (scale_factor ** 2), C // (scale_factor ** 2)
        x = x.reshape(
            n,
            int(h * h_scale_factor * w * w_scale_factor),
            int(c / (h_scale_factor * w_scale_factor)),
        )

        return x

    def extract_feature(self, pixel_values, grid_thw=None):
        if not self.config.anyres_image_size:
            if self.select_layer == -1:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[self.select_layer]
            vit_embeds = vit_embeds[:, 1:, :]
        else:
            if grid_thw is not None:
                grid_thw = grid_thw.to(pixel_values.device)

            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
                grid_thw=grid_thw,
            ).last_hidden_state

        vit_embeds = self.pixel_shuffle_v2(
            vit_embeds,
            scale_factor=self.downsample_ratio,
            patch_aspect_ratio=self.patch_aspect_ratio,
        )
        vit_embeds_after_mlp = self.mlp1(vit_embeds)

        return vit_embeds_after_mlp

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [
            response.split(template.sep.strip())[0].strip() for response in responses
        ]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Generate text tokens from multimodal inputs.

        Args:
            pixel_values (`torch.FloatTensor`, *optional*):
                Image tensor of shape `(B, C, H, W)` used to extract visual features.
            input_ids (`torch.LongTensor`, *optional*):
                Token IDs for the language model inputs.
            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for the input tokens.
            visual_features (`torch.FloatTensor`, *optional*):
                Precomputed vision features to insert at image token positions.
            generation_config (`GenerationConfig`, *optional*):
                Generation configuration for the language model.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states from the language model.
            generate_kwargs:
                Additional kwargs forwarded to `language_model.generate`.

        Returns:
            `torch.LongTensor`: Generated token IDs.
        """

        assert self.img_context_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)
        B, N, C = input_embeds.shape

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate_hidden_states(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Return hidden states from a forward pass with multimodal inputs.

        Args:
            pixel_values (`torch.FloatTensor`, *optional*):
                Image tensor of shape `(B, C, H, W)` used to extract visual features.
            input_ids (`torch.LongTensor`, *optional*):
                Token IDs for the language model inputs.
            attention_mask (`torch.LongTensor`, *optional*):
                Attention mask for the input tokens.
            visual_features (`torch.FloatTensor`, *optional*):
                Precomputed vision features to insert at image token positions.
            generate_kwargs:
                Additional kwargs forwarded to the language model forward pass.

        Returns:
            `CausalLMOutputWithPast`: Output containing hidden states.
        """

        assert self.img_context_token_id is not None
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)
        B, N, C = input_embeds.shape

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)

        outputs = outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
            padding_type="pad",
        )
        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
