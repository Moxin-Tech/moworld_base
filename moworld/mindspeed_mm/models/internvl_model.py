from typing import List, Optional
import os
import sys
import numpy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core.models.gpt import GPTModel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec, get_vit_layer_spec
from mindspeed_mm.models.common.communications import cal_split_sizes
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward
from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel
from mindspeed_mm.utils.utils import EncoderBalanceComm


class InternVLModel(MultiModalModule):
    """
    Vision-Language multi-modal model.
    VLModel is an assembled model, which may include text_encoder, image_encoder, video_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder.
            "add_image_encoder": (bool),  # Whether to construct the image encoder.
            "add_video_encoder": (bool),  # Whether to construct the video encoder.
            "add_text_decoder": (bool),  # Whether to construct the text decoder.
            "img_embedding_idx": (int),  # Index in the language_embeddings tensor where image_embeddings should be inserted.
            "text_encoder": {...},  # Config for the text encoder.
            "image_encoder": {...},  # Config for the image encoder.
            "video_encoder": {...},  # Config for the video encoder.
            "text_decoder": {...},  # Config for the text decoder.
        }
    """
    def __init__(self, config) -> None:
        super().__init__(config)

        self.config = core_transformer_config_from_args(get_args())
        self.pre_process = config.pre_process
        self.post_process = config.post_process
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.img_embedding_idx = config.img_embedding_idx
        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder_model_id = None
        self.text_decoder = None

        #  This attribute is needed to check if an all-reduce is required
        #  on the word embeddings inside 'finalize_model_grads._allreduce_word_embedding_grads'.
        self.share_embeddings_and_output_weights = False

        self.vocab_size = config.text_decoder.vocab_size
        self.img_context_token_id = config.img_context_token_id

        # initialize pipeline prarallel configs
        self.pp_size = mpu.get_pipeline_model_parallel_world_size()
        self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        if self.enable_vp:
            self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
            self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        self.pp_rank = mpu.get_pipeline_model_parallel_rank()

        if self.add_text_encoder:
            self.text_encoder = TextEncoder(config.text_encoder).get_model()
        if self.add_image_encoder:
            self.image_encoder = self._build_image_encoder_model(config.image_encoder)
        if self.add_video_encoder:
            raise NotImplementedError("Not support video_encoder now")
        if self.add_text_decoder:
            self.text_decoder_model_id = getattr(config.text_decoder, "model_id", None)
            self.text_decoder = self._build_text_decoder_model(config.text_decoder)

    def _build_image_encoder_model(self, config):
        transformer_layer_spec = get_vit_layer_spec(config.vision_encoder)
        if self.pp_size <= 1:
            return VisionModel(
                config=config,
                encoder_transformer_layer_spec=transformer_layer_spec
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.vision_encoder.pipeline_num_layers) * len(
                    config.vision_encoder.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in vision_encoder.pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in vision_encoder.pipeline_num_layers: {len(config.vision_encoder.pipeline_num_layers) * len(config.vision_encoder.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.vision_encoder.pipeline_num_layers):
            raise ValueError(
                f"length of vision_encoder.pipeline_num_layers must equal to pipeline-model-parallel-size, "
                f"but got vision_encoder.pipeline_num_layers length:{len(config.vision_encoder.pipeline_num_layers)} "
                f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.vision_encoder.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.vision_encoder.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_image_encoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.vision_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.vision_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(
                sum(vp_layer) for vp_layer in config.vision_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.vision_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
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
        # num_layers will be divided by pp_size and vp_size in TransformerBlock from megatron.core
        config.vision_encoder.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.vision_encoder.num_layers *= self.vp_size
        return VisionModel(
            config=config,
            encoder_transformer_layer_spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_text_decoder_model(self, config):
        transformer_layer_spec = get_language_layer_spec()
        if self.pp_size <= 1:
            return GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                rotary_percent=config.rotary_percent,
                rotary_base=config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process,
                fp16_lm_cross_entropy=config.fp16_lm_cross_entropy
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in pipeline_num_layers: {len(config.pipeline_num_layers) * len(config.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.pipeline_num_layers):
            raise ValueError(f"length of pipeline_num_layers must equal to pipeline-model-parallel-size, "
                             f"but got pipeline_num_layers length:{len(config.pipeline_num_layers)} "
                             f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_text_decoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                config.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(sum(vp_layer) for vp_layer in config.pipeline_num_layers[:self.vp_rank]) + sum(
                config.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
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
        # num_layers will be divided by pp_size and vp_size in TransformerBlock from megatron.core
        config.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.num_layers *= self.vp_size
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            parallel_output=config.parallel_output,
            position_embedding_type=config.position_embedding_type,
            rotary_percent=config.rotary_percent,
            rotary_base=config.rotary_base,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=config.fp16_lm_cross_entropy
        )

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if not len(input_tensor) == 1:
            raise AssertionError("input_tensor should only be length 1 for vlmodel")
        if self.add_image_encoder:
            self.image_encoder.set_input_tensor(input_tensor[0])
        elif self.add_text_decoder:
            if self.text_decoder.pre_process:
                self.input_tensor = input_tensor[0]
            else:
                self.text_decoder.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_image_encoder: bool = False,
        freeze_image_projection: bool = False,
        freeze_video_encoder: bool = False,
        freeze_text_decoder: bool = False,
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_text_decoder (bool): Freeze the text decoder module.
            freeze_image_encoder (bool): Freeze the image encoder module.
            freeze_image_projection (bool): Freeze the image projector module.
            freeze_video_encoder (bool): Freeze the video encoder module.
        """
        if self.add_image_encoder:
            self.image_encoder.freeze(freeze_image_encoder, freeze_image_projection)
        if self.add_text_decoder and freeze_text_decoder:
            for param in self.text_decoder.parameters():
                param.requires_grad = False

    def _prepare_decoder_attention_mask(self, attention_mask, dtype=torch.float32, device=torch.device("npu"), past_key_values_length=0):
        # create causal mask

        # Copied from transformers.models.bart.modeling_bart._make_causal_mask
        def _make_causal_mask(
            input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
        ):
            """
            Make causal mask used for bi-directional self-attention.
            """
            bsz, tgt_len = input_ids_shape
            mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
            mask_cond = torch.arange(mask.size(-1), device=device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(dtype)

            if past_key_values_length > 0:
                mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
            return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

        # Copied from transformers.models.bart.modeling_bart._expand_mask
        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
            """
            Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
            """
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        input_shape = attention_mask.shape
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask.bool()

    def compute_loss(self, logits, labels, ignore_flag=False):
        # Shift tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if ignore_flag:
            loss = loss * 0.0

        return loss

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        image_flags: Optional[torch.LongTensor] = None,
        transfer: Optional[numpy.ndarray] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        output = None
        args = get_args()
        if self.add_image_encoder:
            vit_embeds = self.image_encoder(image)
            if self.image_encoder.post_process:
                if get_args().encoder_dp_balance:
                    vit_embeds = EncoderBalanceComm.apply(
                            vit_embeds,
                            mpu.get_data_parallel_group(),
                            transfer
                        )
                image_flags = image_flags.squeeze(-1)
                vit_embeds = vit_embeds[image_flags == 1].reshape(1, -1, vit_embeds.shape[-1]).clone()
            output = vit_embeds
            output = output.contiguous()
        else:
            vit_embeds = self.input_tensor

        if self.add_text_decoder:
            input_embeds = None
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                input_embeds = input_embeds.transpose(0, 1)
                B, S, H = input_embeds.shape
                input_embeds = input_embeds.reshape(B * S, H)
                input_ids = input_ids.reshape(B * S)
                selected = (input_ids == self.img_context_token_id)
                if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE' \
                    and (vit_embeds is None or len(vit_embeds) == 0):
                    vit_embeds = torch.zeros((1, input_embeds.shape[1]), dtype=torch.bfloat16).to('npu:0')
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.squeeze(0)
                input_embeds = input_embeds.reshape(B, S, H).transpose(0, 1)

            attention_mask = self._prepare_decoder_attention_mask(attention_mask)

            split_gather_sizes = None
            args = get_args()
            if args.context_parallel_size is not None and args.context_parallel_size > 1:
                split_gather_sizes = cal_split_sizes(input_embeds.shape[0], args.context_parallel_size)
                input_embeds = split_forward_gather_backward(input_embeds, mpu.get_context_parallel_group(),
                                                            dim=0, grad_scale="down", split_sizes=split_gather_sizes)

            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None,
            )

            if self.text_decoder.post_process:
                if args.context_parallel_size is not None and args.context_parallel_size > 1:
                    output = gather_forward_split_backward(output, mpu.get_context_parallel_group(),
                                                            dim=1, grad_scale="up", gather_sizes=split_gather_sizes)
                logits = output
                logits = logits.float()

                loss = None
                if labels is not None:
                    loss = self.compute_loss(logits, labels)

                return {
                    "loss": loss,
                    "logits": logits
                }

        #auto_tuning
        if os.getenv('OOTB_OPTIMIZER_PROFILING', 'FALSE') == 'TRUE' \
            and self.image_encoder is not None and self.image_encoder.post_process:
            required_size = args.micro_batch_size * input_ids.shape[1] * self.vocab_size
            current_size = vit_embeds.numel()
            diff = required_size - current_size

            # Pad with zeros at the end of the data.
            padded = torch.cat([
                vit_embeds.flatten(),
                torch.zeros(diff, device=vit_embeds.device)
            ])

            output = padded.reshape(args.micro_batch_size, input_ids.shape[1], self.vocab_size)

            logits = output
            logits = logits.float()

            loss = None
            if labels is not None:
                loss = self.compute_loss(logits, labels)

            return {
                "loss": loss,
                "logits": logits
            }
        return output
