# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the ./LICENSE file.

import os
from typing import Optional, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashForCausalLM
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging

from .configuration_longcat_next import LongcatNextConfig
from .modeling_longcat_ngram import LongcatFlashNgramModel, NgramCache
from .modular_longcat_next import CasualDepthTransformerHead
from .modular_longcat_next_audio import LongcatNextAudioTokenizer
from .modular_longcat_next_visual import LongcatNextVisualTokenizer

from .cosy24k_vocoder import Cosy24kVocoder
from .image_refiner import ImageRefinerContainer
from .refiner_modules import FlowMatchEulerDiscreteScheduler

logger = logging.get_logger(__name__)


@dataclass
class LongcatNextForCausalLMOutputWithPast(CausalLMOutputWithPast):
    visual_loss: Optional[torch.FloatTensor] = None
    visual_logits: Optional[torch.FloatTensor] = None
    visual_ids: Optional[torch.LongTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    audio_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerateDecoderOnlyOutput(GenerateDecoderOnlyOutput):
    visual_ids: Optional[torch.LongTensor] = None
    audio_ids: Optional[torch.LongTensor] = None
    audio_text_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerateEncoderDecoderOutput(GenerateEncoderDecoderOutput):
    visual_ids: Optional[torch.LongTensor] = None
    audio_ids: Optional[torch.LongTensor] = None
    audio_text_ids: Optional[torch.LongTensor] = None


@dataclass
class LongcatNextForCausalLMGenerationStatus:
    mode: str = "text"
    current_image_token_num: int = -1
    audio_parallel_decoding: bool = False
    is_audio_text_end: bool = False
    is_audio_start: bool = False
    last_step_mode: str = None

    def __init__(self, visual_generation_config, audio_generation_config):
        self.visual_generation_config = visual_generation_config
        self.h = self.visual_generation_config.custom_params["token_h"]
        self.w = self.visual_generation_config.custom_params["token_w"]
        self.anyres_prefix = self.visual_generation_config.custom_params["anyres_prefix"].format(h=self.h, w=self.w)
        self.audio_generation_config = audio_generation_config
        self.audio_parallel_decoding = audio_generation_config.audio_parallel_decoding

    def switch_to(self, modal):
        self.mode = modal
        self.current_image_token_num = 0 if modal == "visual" else -1
        self.is_audio_text_end = False
        self.is_audio_start = False

    @property
    def is_img_newline(self):
        return ((self.current_image_token_num + 1) % (self.w + 1)) == 0 and not self.is_img_end

    @property
    def is_img_end(self):
        return (self.current_image_token_num + 1) / (self.w + 1) == self.h


class LongcatNextModel(LongcatFlashNgramModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatNextConfig

    def __init__(self, config):
        super().__init__(config)
        self.visual_tokenizer = LongcatNextVisualTokenizer(config)
        self.audio_tokenizer = LongcatNextAudioTokenizer(config)

        self._init_multimodal_constants(config)
        self.post_init()

    def _init_multimodal_constants(self, config):
        name2id_dict = {
            "image_newline_token_id": self.config.visual_config.image_newline_token_id,
            "image_end_token_id": self.config.visual_config.image_end_token_id,
            "image_pad_token_id": self.config.visual_config.image_pad_token_id,
            "audiotext_start_token_id": config.audio_config.audiotext_start_token_id,
            "audiotext_pad_token_id": self.config.audio_config.audiotext_pad_token_id,
            "audiogen_end_token_id": config.audio_config.audiogen_end_token_id,
            "audio_pad_token_id": self.config.audio_config.audio_pad_token_id,
        }
        for k, v in name2id_dict.items():
            self.register_buffer(k, torch.tensor([v], dtype=torch.long), persistent=False)
        visual_offset_list = [config.visual_offset] + config.visual_config.vq_config.codebook_sizes[:-1]
        visual_offset_vals = torch.cumsum(torch.tensor(visual_offset_list, dtype=torch.long), dim=0)
        self.register_buffer("visual_offset_vals", visual_offset_vals, persistent=False)
        audio_offset_list = [config.audio_offset] + config.audio_config.vq_config.codebook_sizes[:-1]
        audio_offset_vals = torch.cumsum(torch.tensor(audio_offset_list, dtype=torch.long), dim=0)
        self.register_buffer("audio_offset_vals", audio_offset_vals, persistent=False)
        print(f"{self.visual_offset_vals=}")
        print(f"{self.audio_offset_vals=}")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        visual_inputs=None,
        visual_ids=None,
        audio_inputs=None,
        audio_ids=None,
        audio_text_ids=None,
        multimodal_generation_status=None,
        **kwargs
    ) -> BaseModelOutputWithPast:

        if input_ids is None:
            raise ValueError("You must specify input_ids")

        # Extract N-gram context if available
        ngram_context = None
        if isinstance(past_key_values, NgramCache) and past_key_values.ngram_context is not None:
            ngram_context = past_key_values.ngram_context

        special_visual_mask, special_audio_mask, special_audio_text_start_mask, special_audio_text_pad_mask = self.get_placeholder_mask(input_ids[:1]) # seq-dim

        if inputs_embeds is None:
            input_ids[:, special_visual_mask | special_audio_mask | special_audio_text_pad_mask | special_audio_text_start_mask] = 0
            filled_text_pad_mask = torch.ones_like(special_audio_mask)
            audio_text_position_mask = (special_audio_text_pad_mask | special_audio_text_start_mask | special_audio_mask)

            if audio_text_ids is not None and audio_text_ids.size(1) > 0 and audio_text_position_mask.sum() > 0:
                filled_text = audio_text_ids[:, -audio_text_position_mask.sum():]
                filled_text_pad_mask = (filled_text == self.config.audio_config.audiotext_pad_token_id)[0]
                input_ids[:, audio_text_position_mask] = filled_text
                input_ids[input_ids == self.config.audio_config.audiotext_pad_token_id] = 0

            inputs_embeds = self.ngram_embeddings(input_ids, ngram_context=ngram_context)
            inputs_embeds[:, (special_visual_mask | (special_audio_mask & filled_text_pad_mask))] = 0

        if special_audio_text_start_mask.sum() > 0:
            audio_text_start_embedding = self.embed_tokens(self.audiotext_start_token_id)
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_audio_text_start_mask] += audio_text_start_embedding
            else:
                inputs_embeds[:, special_audio_text_start_mask] += audio_text_start_embedding

        if visual_inputs is not None:
            visual_ids = self.get_visual_ids(**visual_inputs) # [<bs=1>*seq, lev]

        if visual_ids is not None and special_visual_mask.sum() > 0:
            visual_embeddings = self.get_visual_embeddings(visual_ids[-special_visual_mask.sum():]) # -> [seq, dim]
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_visual_mask] = visual_embeddings.to(inputs_embeds.device)
            else:
                inputs_embeds[:, special_visual_mask] = visual_embeddings.to(inputs_embeds.device)

        if audio_inputs is not None:
            audio_ids = self.get_audio_ids(**audio_inputs) # -> [<bs=1>*seq, lev]

        if audio_ids is not None and special_audio_mask.sum() > 0:
            audio_embeddings = self.get_audio_embeddings(audio_ids[-special_audio_mask.sum():]) # -> [seq, dim]
            if multimodal_generation_status.last_step_mode is None: # prefill
                inputs_embeds[:1, special_audio_mask] += audio_embeddings.to(inputs_embeds.device)
            else:
                inputs_embeds[:, special_audio_mask] += audio_embeddings.to(inputs_embeds.device)

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(config=self.config)

        # Update N-gram context
        if use_cache and isinstance(past_key_values, NgramCache):
            past_key_values.update_ngram_context(input_ids)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs
        )

    def get_visual_ids(self, pixel_values, visual_grid_thw, offset=True):
        visual_ids = self.visual_tokenizer.encode(pixel_values, visual_grid_thw)
        if offset:
            visual_ids += self.visual_offset_vals.to(visual_ids.device)
        return visual_ids

    def get_audio_ids(self, audio, encoder_length, bridge_length, offset=True):
        audio_ids = self.audio_tokenizer.encode(audio, encoder_length, bridge_length)
        if offset:
            audio_ids += self.audio_offset_vals.to(audio_ids.device)
        return audio_ids

    @torch.no_grad()
    def decode_visual_ids_and_save(
        self,
        visual_ids,
        save_prefix,
        token_h,
        token_w,
        **kwargs,
    ):
        visual_ids -= self.visual_offset_vals.to(visual_ids.device)

        if not (save_prefix.startswith("./") or save_prefix.startswith("/")):
            save_prefix = f"./{save_prefix}"
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        return self.visual_tokenizer.lazy_decode_and_save(visual_ids, token_h, token_w, f"{save_prefix}_{0}.png")

    @torch.no_grad()
    def decode_audio_ids_and_save(
        self,
        audio_ids,
        save_prefix,
        sampling_rate,
        wave_concat_overlap,
        **kwargs,
    ):
        audio_ids -= self.audio_offset_vals.to(audio_ids.device)

        if not (save_prefix.startswith("./") or save_prefix.startswith("/")):
            save_prefix = f"./{save_prefix}"
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
        save_path = f"{save_prefix}_{0}.wav"
        self.audio_tokenizer.lazy_decode_and_save(audio_ids, sampling_rate, wave_concat_overlap, save_path)
        return [save_path]

    def get_visual_embeddings(self, visual_ids):
        visual_embeddings = self.embed_tokens(visual_ids).sum(dim=1) # [seq, lev] -> [seq, lev, dim] -> [seq, dim]
        visual_embeddings = self.visual_tokenizer.visual_embedding_layer(visual_embeddings)
        return visual_embeddings

    def get_audio_embeddings(self, audio_ids):
        audio_embeddings = self.embed_tokens(audio_ids).sum(dim=1)
        return audio_embeddings

    def get_placeholder_mask(self, input_ids: torch.LongTensor):
        special_image_mask = (input_ids == self.config.visual_config.image_pad_token_id).squeeze(0)
        special_audio_mask = (input_ids == self.config.audio_config.audio_pad_token_id).squeeze(0)
        special_audio_text_start_mask = (input_ids == self.config.audio_config.audiotext_start_token_id).squeeze(0)
        special_audio_text_pad_mask = (input_ids == self.config.audio_config.audiotext_pad_token_id).squeeze(0)
        return special_image_mask, special_audio_mask, special_audio_text_start_mask, special_audio_text_pad_mask


class LongcatNextForCausalLM(LongcatFlashForCausalLM):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    _no_split_modules = [
        "LongcatFlashDecoderLayer",
        "CasualDepthTransformerHead",
    ]
    config_class = LongcatNextConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LongcatNextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.text_vocab_plus_multimodal_special_token_size, bias=False)

        self.visual_head = CasualDepthTransformerHead(
            hidden_size=config.hidden_size,
            codebook_sizes=config.visual_config.vq_config.codebook_sizes,
            transformer_layer_num=config.visual_config.image_head_transformer_layers,
            transformer_dim=config.visual_config.image_head_transformer_dims,
            transformer_ffn_scale=config.visual_config.image_head_transformer_ffn_scale,
        )
        self.audio_head = CasualDepthTransformerHead(
            hidden_size=config.hidden_size,
            codebook_sizes=config.audio_config.vq_config.codebook_sizes,
            transformer_layer_num=config.audio_config.audio_head_transformer_layers,
            transformer_dim=config.audio_config.audio_head_transformer_dims,
            transformer_ffn_scale=config.audio_config.audio_head_transformer_ffn_scale,
        )

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        visual_inputs=None,
        visual_ids=None,
        audio_inputs=None,
        audio_ids=None,
        audio_text_ids=None,
        multimodal_generation_status: LongcatNextForCausalLMGenerationStatus = None,
        visual_generation_config: GenerationConfig = None,
        audio_generation_config: GenerationConfig = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        visual_inputs (`BatchFeature`, *optional*):
            Visual inputs returned by the processor, containing pixel values and grid metadata for image encoding.
        visual_ids (`torch.LongTensor` of shape `(num_visual_tokens, num_codebooks)`, *optional*):
            Quantized visual token ids from the visual tokenizer, used to build visual embeddings during generation.
        audio_inputs (`BatchFeature`, *optional*):
            Audio inputs returned by the processor, containing mel-spectrogram features and length metadata.
        audio_ids (`torch.LongTensor` of shape `(num_audio_tokens, num_codebooks)`, *optional*):
            Quantized audio token ids from the audio tokenizer, used to build audio embeddings during generation.
        audio_text_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Token ids for the audio text transcript generated alongside audio tokens.
        multimodal_generation_status (`LongcatNextForCausalLMGenerationStatus`, *optional*):
            Stateful object tracking the current multimodal generation mode (text / visual / audio) and
            associated counters used to route logits to the correct head during auto-regressive decoding.
        visual_generation_config (`GenerationConfig`, *optional*):
            Generation configuration for the visual head, controlling sampling parameters such as
            `temperature`, `top_k`, `top_p`, and custom parameters like `cfg_scale` and `anyres_config`.
        audio_generation_config (`GenerationConfig`, *optional*):
            Generation configuration for the audio head, controlling sampling parameters such as
            `temperature`, `top_k`, `top_p`, `repetition_penalty`, and `audio_parallel_decoding`.
        """

        if multimodal_generation_status.mode == "visual" and visual_generation_config.custom_params["cfg_scale"] != 1.0 and input_ids.size(0) == 1:
            input_ids = input_ids.repeat((2, 1))

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_inputs=visual_inputs,
            visual_ids=visual_ids,
            audio_inputs=audio_inputs,
            audio_ids=audio_ids,
            audio_text_ids=audio_text_ids,
            multimodal_generation_status=multimodal_generation_status,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        slice_hidden_states = hidden_states[:, slice_indices, :]

        loss, logits = None, None
        if multimodal_generation_status.mode == "visual" and \
            (not multimodal_generation_status.is_img_newline) and (not multimodal_generation_status.is_img_end):
            visual_ids = self.get_multimodal_logits_and_ids(
                self.visual_head,
                visual_ids,
                slice_hidden_states,
                self.model.embed_tokens,
                self.config.visual_config.vq_config.codebook_sizes,
                self.model.visual_offset_vals,
                visual_generation_config,
            )
        else:
            logits = self.lm_head(slice_hidden_states)

        if multimodal_generation_status.mode == "audio" and multimodal_generation_status.is_audio_start:
            audio_ids = self.get_multimodal_logits_and_ids(
                self.audio_head,
                audio_ids,
                slice_hidden_states,
                self.model.embed_tokens,
                self.config.audio_config.vq_config.codebook_sizes,
                self.model.audio_offset_vals,
                audio_generation_config,
            )

        return LongcatNextForCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            visual_ids=visual_ids,
            audio_ids=audio_ids,
        )

    def get_multimodal_logits_and_ids(
        self,
        head_model,
        multimodal_ids,
        hidden_states,
        multimodal_embedding_layer,
        codebook_sizes,
        offset_vals,
        multimodal_generation_config,
    ):
        next_token_ids = torch.zeros(hidden_states.size(0), len(codebook_sizes), dtype=torch.long, device=hidden_states.device)
        multimodal_embedding_layer = multimodal_embedding_layer.to(hidden_states.device)

        for level, _ in enumerate(codebook_sizes):
            logits = head_model(hidden_states, next_token_ids, multimodal_embedding_layer, level) # -> (bs, 1, dim)
            next_token_id = self.inner_sample(logits, multimodal_ids[None, :, level] - offset_vals[level], multimodal_generation_config) # (bs, 1)
            next_token_id += offset_vals[level]
            next_token_ids[:, level] = next_token_id

        return next_token_ids[:1]

    def inner_sample(
        self,
        next_token_logits: torch.Tensor,
        multimodal_ids: torch.LongTensor,
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        logits_processor = self._get_logits_processor(generation_config)

        if "cfg_scale" in generation_config.custom_params and generation_config.custom_params["cfg_scale"] != 1.0:
            cond_logits, uncond_logits = next_token_logits.chunk(2, dim=0)
            next_token_logits = generation_config.custom_params["cfg_scale"] * (cond_logits - uncond_logits) + uncond_logits

        next_token_scores = logits_processor(multimodal_ids, next_token_logits.to(multimodal_ids.device))
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
        return next_tokens

    @torch.no_grad()
    def generate(self, inputs=None, **kwargs):
        """Override to ensure NgramCache is used."""

        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = NgramCache(config=self.config)

        return super().generate(
            inputs=inputs,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        visual_ids,
        audio_ids,
        audio_text_ids,
        multimodal_generation_status,
        generation_config,
        attention_mask,
        cache_position,
        **kwargs,
    ):
        extra_new_tokens = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)
        if visual_ids is None:
            visual_ids = torch.empty(0, len(self.config.visual_config.vq_config.codebook_sizes), dtype=torch.long, device=input_ids.device)
        if audio_ids is None:
            audio_ids = torch.empty(0, len(self.config.audio_config.vq_config.codebook_sizes), dtype=torch.long, device=input_ids.device)
        if audio_text_ids is None:
            audio_text_ids = torch.empty(input_ids.size(0), 0, dtype=torch.long, device=input_ids.device)

        def insert_ids(new_ids, _input_ids, _attention_mask, _cache_position, position=0):
            if position < 0:
                parts = [_input_ids[:, :position], new_ids, _input_ids[:, position:]]
            else:
                parts = [_input_ids, new_ids]
            _input_ids = torch.cat(parts, dim=1)
            insert_len = new_ids.size(1)
            _attention_mask = F.pad(_attention_mask, (0, insert_len), value=1)
            insert_position = _cache_position[-1] + 1 + torch.arange(insert_len, device=_cache_position.device)
            _cache_position = torch.cat([_cache_position, insert_position])
            return _input_ids, _attention_mask, _cache_position

        # multimodal generation status change
        if cache_position[0] != 0:
            multimodal_generation_status.last_step_mode = multimodal_generation_status.mode

        if multimodal_generation_status.mode == "visual":
            multimodal_generation_status.current_image_token_num += 1

        if (input_ids[:, -1] == self.config.visual_config.image_start_token_id).all():
            multimodal_generation_status.switch_to("visual")
            anyres_prefix_ids = self.text_tokenizer.encode(multimodal_generation_status.anyres_prefix, return_tensors="pt")
            anyres_prefix_ids = anyres_prefix_ids.to(input_ids.device)
            extra_new_tokens = torch.cat([extra_new_tokens, anyres_prefix_ids], dim=1)
            input_ids, attention_mask, cache_position = insert_ids(anyres_prefix_ids, input_ids, attention_mask, cache_position, position=-1)
            if input_ids.size(0) == 1: # cfg, change bs=1 -> 2
                input_ids = input_ids.repeat((2, input_ids.size(1)))
                input_ids[1, :-(anyres_prefix_ids.size(-1) + 1)] = 0
                print(f"change to cfg, input_ids: {input_ids}")
                attention_mask = attention_mask.repeat((2, attention_mask.size(1)))

        elif (input_ids[:, -1] == self.config.audio_config.audiogen_start_token_id).all():
            multimodal_generation_status.switch_to("audio")

        elif (input_ids[:, -1] == self.config.audio_config.audiotext_start_token_id).all():
            multimodal_generation_status.is_audio_start = True

        elif ((input_ids[:, -1] == self.config.visual_config.image_end_token_id) | (input_ids[:, -1] == self.config.audio_config.audiogen_end_token_id)).all():
            multimodal_generation_status.switch_to("text")

        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            visual_ids=visual_ids,
            audio_ids=audio_ids,
            audio_text_ids=audio_text_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        if model_inputs["cache_position"][0] != 0:
            model_inputs["visual_inputs"] = None
            model_inputs["audio_inputs"] = None

        return model_inputs, multimodal_generation_status, extra_new_tokens

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        visual_ids=None,
        audio_ids=None,
        audio_text_ids=None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        visual_generation_config = GenerationConfig(**generation_config.visual_generation_config)
        audio_generation_config = GenerationConfig(**generation_config.audio_generation_config)
        multimodal_generation_status = LongcatNextForCausalLMGenerationStatus(visual_generation_config, audio_generation_config)

        pbar = tqdm(iter(int, 1), desc="Generating", unit="tok")
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs, multimodal_generation_status, extra_new_tokens = self.prepare_inputs_for_generation(
                input_ids,
                visual_ids,
                audio_ids,
                audio_text_ids,
                multimodal_generation_status,
                generation_config,
                **model_kwargs,
            )
            if extra_new_tokens.size(1) > 0:
                input_ids = torch.cat([input_ids[:, :-1], extra_new_tokens, input_ids[:, -1:]], dim=1)
                model_kwargs["attention_mask"] = model_inputs["attention_mask"]
                model_kwargs["cache_position"] = model_inputs["cache_position"]

            if multimodal_generation_status.mode == "text" and multimodal_generation_status.last_step_mode == "visual":
                next_tokens = generation_config._eos_token_tensor
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                if streamer is not None:
                    streamer.put(next_tokens.cpu())
                break

            visual_ids = model_inputs["visual_ids"]
            audio_ids = model_inputs["audio_ids"]
            audio_text_ids = model_inputs["audio_text_ids"]

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True, multimodal_generation_status=multimodal_generation_status, visual_generation_config=visual_generation_config, audio_generation_config=audio_generation_config)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True, multimodal_generation_status=multimodal_generation_status, visual_generation_config=visual_generation_config, audio_generation_config=audio_generation_config)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=1,
            )
            if synced_gpus and this_peer_finished:
                continue


            # multimodal generation
            if multimodal_generation_status.mode == "text" or \
                (multimodal_generation_status.mode == "audio" and not multimodal_generation_status.is_audio_text_end):
                # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
                # (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # audio_text_ids done
                if multimodal_generation_status.mode == "audio" and (next_tokens == self.config.audio_config.audiotext_pad_token_id).all():
                    multimodal_generation_status.is_audio_text_end = True

            elif multimodal_generation_status.mode == "visual":
                if multimodal_generation_status.is_img_end:
                    next_tokens = self.model.image_end_token_id.to(input_ids.device)

                elif multimodal_generation_status.is_img_newline:
                    next_tokens = self.model.image_newline_token_id.to(input_ids.device)

                else:
                    visual_ids = torch.cat([visual_ids, outputs.visual_ids], dim=0) # [seq, lev]
                    next_tokens = self.model.image_pad_token_id.to(input_ids.device)

            else: # mode == "audio" and multimodal_generation_status.is_audio_text_end
                next_tokens = self.model.audio_pad_token_id.to(input_ids.device)


            if multimodal_generation_status.mode == "audio":
                # audio_text_ids update
                audio_text_next_tokens = self.model.audiotext_pad_token_id.to(input_ids.device)
                if not multimodal_generation_status.is_audio_text_end:
                    audio_text_next_tokens, next_tokens = next_tokens, audio_text_next_tokens
                audio_text_ids = torch.cat((audio_text_ids, audio_text_next_tokens[:, None]), dim=1)

                # audio_ids update
                if multimodal_generation_status.is_audio_start:
                    if outputs.audio_ids[-1, 0] == (self.model.audio_offset_vals[1]): # offset + (level_1_len)
                        next_tokens = self.model.audiogen_end_token_id.to(input_ids.device)
                    else:
                        next_tokens = self.model.audio_pad_token_id.to(input_ids.device)
                    audio_ids = torch.cat([audio_ids, outputs.audio_ids], dim=0)

                elif (multimodal_generation_status.audio_parallel_decoding) or \
                        (not multimodal_generation_status.audio_parallel_decoding and multimodal_generation_status.is_audio_text_end):
                    next_tokens = self.model.audiotext_start_token_id.to(input_ids.device)


            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

            pbar.update(1)
            pbar.set_postfix({
                "recent_5toks": f"{input_ids[:, -5:].tolist()}",
            })

        pbar.close()

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return LongcatNextForCausalLMGenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    visual_ids=visual_ids,
                    audio_ids=audio_ids,
                    audio_text_ids=audio_text_ids,
                )
            else:
                return LongcatNextForCausalLMGenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    visual_ids=visual_ids,
                    audio_ids=audio_ids,
                    audio_text_ids=audio_text_ids,
                )
        else:
            return input_ids, visual_ids, audio_ids, audio_text_ids


__all__ = ["LongcatNextModel", "LongcatNextForCausalLM"]
