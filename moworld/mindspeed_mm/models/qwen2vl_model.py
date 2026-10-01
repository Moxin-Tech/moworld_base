# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Tuple, Union
import torch
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module_spec.qwen2vl_layer_spec import get_qwen2vl_layer_spec, get_mlp_module_spec, get_qwen2vl_llm_layer_spec
from mindspeed_mm.models.vision.vision_model import Qwen2vlVisionModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel


class Qwen2VLModel(MultiModalModule):
    """
    Vision-Language multi-modal model.
    Qwen2VLModel is an assembled model, which include image_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel, model.json中的配置
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder. not used now.
            "add_image_encoder": (bool),  # Whether to construct the image encoder.
            "add_video_encoder": (bool),  # Whether to construct the video encoder. not used now.
            "add_text_decoder": (bool),  # Whether to construct the text decoder.
            "img_context_token_id": (int),  # Index in the language_embeddings tensor where image_embeddings should be inserted.
            "text_encoder": {...},  # Config for the text encoder. not used now.
            "image_encoder": {...},  # Config for the image encoder.
            "video_encoder": {...},  # Config for the video encoder. not used now.
            "text_decoder": {...},  # Config for the text decoder.
        }
    """

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self.config = core_transformer_config_from_args(get_args())
        self.pre_process: bool = config.pre_process
        self.post_process: bool = config.post_process
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.share_embeddings_and_output_weights = not config.text_decoder.untie_embeddings_and_output_weights
        self.position_embedding_type = config.text_decoder.position_embedding_type
        self.img_context_token_id = config.img_context_token_id
        self.vocab_size = config.text_decoder.vocab_size
        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder = None

        # initialize pipeline parallel configs
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        if mpu.get_virtual_pipeline_model_parallel_world_size() is not None:
            raise NotImplementedError("Not support virtual_pipeline_model_parallel now")
        else:
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_image_encoder:
            self.image_encoder = self._build_image_encoder_model(config.image_encoder)
        if self.add_video_encoder:
            raise NotImplementedError("Not support video_encoder now")
        if self.add_text_decoder:
            self.text_decoder = self._build_text_decoder_model(config.text_decoder)

    def _build_image_encoder_model(self, config):
        vit_layer_spec = get_qwen2vl_layer_spec(is_vit=True)
        proj_layer_spec = get_mlp_module_spec(use_te=False).submodules

        if self.pp_size <= 1:
            return Qwen2vlVisionModel(
                config=config,
                encoder_transformer_layer_spec=vit_layer_spec,
                projector_layer_spec=proj_layer_spec
            )

        if self.pp_size != len(config.vision_encoder.pipeline_num_layers):
            raise ValueError(f"length of vision_encoder.pipeline_num_layers must equal to pipeline-model-parallel-size, "
                             f"but got vision_encoder.pipeline_num_layers length:{len(config.vision_encoder.pipeline_num_layers)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        local_num_layers = config.vision_encoder.pipeline_num_layers[self.pp_rank]
        if local_num_layers == 0:
            self.add_image_encoder = False
            return None

        pipeline_start_index = sum(config.vision_encoder.pipeline_num_layers[:self.pp_rank])
        pipeline_end_index = sum(config.vision_encoder.pipeline_num_layers[:self.pp_rank + 1])

        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.vision_encoder.num_layers

        print(
            f"image encoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.vision_encoder.num_layers = self.pp_size * local_num_layers
        return Qwen2vlVisionModel(
            config=config,
            encoder_transformer_layer_spec=vit_layer_spec,
            projector_layer_spec=proj_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_text_decoder_model(self, config):
        if self.pp_size <= 1:
            return MMGPTModel(
                config=config,
                transformer_layer_spec=get_qwen2vl_llm_layer_spec(),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta,
                pre_process=self.pre_process,
                post_process=self.post_process
            )

        if self.pp_size != len(config.pipeline_num_layers):
            raise ValueError(f"length of pipeline_num_layers must equal to pipeline-model-parallel-size, "
                             f"but got pipeline_num_layers length:{len(config.pipeline_num_layers)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        local_num_layers = config.pipeline_num_layers[self.pp_rank]
        if local_num_layers == 0:
            self.add_text_decoder = False
            return None

        pipeline_start_index = sum(config.pipeline_num_layers[:self.pp_rank])
        pipeline_end_index = sum(config.pipeline_num_layers[:self.pp_rank + 1])

        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.num_layers

        print(
            f"text decoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.num_layers = self.pp_size * local_num_layers

        return MMGPTModel(
                config=config,
                transformer_layer_spec=get_qwen2vl_llm_layer_spec(),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta,
                pre_process=pre_process,
                post_process=post_process
            )


    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError("input_tensor should only be length 1 for vlmodel")
        self.input_tensor_dpo = input_tensor[0]
        if self.add_image_encoder:
            self.image_encoder.set_input_tensor(input_tensor[0])
        elif self.add_text_decoder:
            if self.text_decoder.pre_process:
                self.input_tensor = input_tensor[0]
            else:
                self.text_decoder.set_input_tensor(input_tensor[0])

    def freeze(
            self,
            freeze_text_decoder: bool = False,
            freeze_image_encoder: bool = False,
            freeze_image_projection: bool = False,
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_text_decoder (bool): Freeze the text decoder module.
            freeze_image_encoder (bool): Freeze the image encoder module.
            freeze_image_projection (bool): Freeze the image projector module.
        """
        if self.add_image_encoder:
            self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)
        if self.add_text_decoder and freeze_text_decoder:
            for param in self.text_decoder.parameters():
                param.requires_grad = False

    def compute_loss(self, logits, labels, ignore_flag=False):
        # shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size//get_args().tensor_model_parallel_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if ignore_flag:
            loss = loss * 0.0

        return loss

    def compute_megatron_loss(self, logits, labels):
        # shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # To align with torch.nn.CrossEntropyLoss, disable max normalization in vocab_parallel_cross_entropy (comment out)
        loss = tensor_parallel.vocab_parallel_cross_entropy(shift_logits.float(), shift_labels)
        loss = loss * (shift_labels > -1)
        loss = torch.sum(loss) / torch.sum(shift_labels > -1)

        return loss

    def _build_causal_mask(self, input_ids, attention_mask):
        seq_len = input_ids.shape[1]
        past_seen_token = 0
        cache_position = torch.arange(
            past_seen_token, past_seen_token + seq_len, device=input_ids.device)
        dtype, device = torch.bfloat16, input_ids.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_ids.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_token + sequence_length + 1
        )
        batch_size = input_ids.shape[0]

        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
            return causal_mask < 0


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
            rope_deltas: Optional[torch.LongTensor] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:

        if self.add_image_encoder and pixel_values is not None:
            vit_embeds = self.image_encoder(pixel_values, image_grid_thw)
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

            seq_len = input_ids.shape[1]
            if inference_params is not None:
                past_seen_tokens = attention_mask.shape[1] - 1 if inference_params.key_value_memory_dict else 0
            else:
                past_seen_tokens = 0
            if position_ids is None:
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
                )
                position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1)

            attention_mask = self._build_causal_mask(input_ids=input_ids, attention_mask=attention_mask)

            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None,
                inference_params=inference_params,
            )

            if self.text_decoder.post_process:
                logits = output
                logits = logits.contiguous().float()
                loss = None
                if labels is not None:
                    loss = self.compute_megatron_loss(logits, labels)
                return {
                    "loss": loss,
                    "logits": logits
                }
        return output
