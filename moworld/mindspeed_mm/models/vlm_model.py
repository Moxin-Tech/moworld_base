# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Tuple, Union

import torch
import torch.distributed as dist
import numpy
from torch.nn import CrossEntropyLoss

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import gather_forward_split_backward, \
    cal_split_sizes, split_forward_gather_backward
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ulysses_world_size
)

from mindspeed_mm.utils.utils import split_forward_gather_backward_with_megatron_cp
from mindspeed_mm.models.common.module_spec.get_layer_spec import get_vit_layer_spec, get_llm_layer_spec, \
    get_projector_layer_spec, get_audio_layer_spec
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.models.audio.audio_model import AudioModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.common.mm_gpt_model import MMGPTModel
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import prepare_positionsids_mask_for_llm
from mindspeed_mm.utils.hetero_parallel import change_parallel_state
from mindspeed_mm.utils.utils import EncoderBalanceComm
try:
    from mindspeed_mm.models.transformers.base_model import FSDP2Mixin, WeightInitMixin
except Exception as e:
    print(
        "⚠️ FSDP2Mixin and WeightInitMixin are not available\n"
        "If you want to use them, please ensure torch version >= 2.7.1"
    )
    class FSDP2Mixin: pass
    class WeightInitMixin: pass


class VLMModel(MultiModalModule, FSDP2Mixin, WeightInitMixin):
    """
    Vision-Language multi-modal model.
    VLMModel is an assembled model, which include image_encoder, text_decoder model.

    Args:
        config (dict): the general config for VLModel, model.json中的配置
        {
            "pre_process": (bool),  # Include the embedding leayer in the gpt decoder (used with pipeline parallelism).
            "post_process": (bool),  # Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism).
            "add_text_encoder": (bool),  # Whether to construct the text encoder. not used now.
            "reward_process: (bool, optional), # Without an output layer in the gpt decoder (only used with videoalign). Defaults to False.
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
        args = get_args()

        self.config = core_transformer_config_from_args(args)
        self.pre_process: bool = config.pre_process
        self.post_process: bool = config.post_process
        self.reward_process: bool = getattr(config, 'reward_process', False)
        self.add_text_encoder = config.text_encoder is not None
        self.add_image_encoder = config.image_encoder is not None
        self.add_video_encoder = config.video_encoder is not None
        self.add_text_decoder = config.text_decoder is not None
        self.add_audio_encoder = hasattr(config, "audio_encoder") and config.audio_encoder is not None

        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder = None

        self.share_embeddings_and_output_weights = not getattr(config.text_decoder,
                                                               'untie_embeddings_and_output_weights', True)
        self.img_context_token_id = config.img_context_token_id
        self.vision_start_token_id = getattr(config, "vision_start_token_id", None)

        # initialize pipeline parallel configs
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
            self.position_embedding_type = config.text_decoder.position_embedding_type
            self.vocab_size = config.text_decoder.vocab_size
            self.text_decoder = self._build_text_decoder_model(config.text_decoder)
        if self.add_audio_encoder:
            self.audio_encoder = self._build_audio_encoder_model(config.audio_encoder)

        if args.hetero_parallel:
            change_parallel_state('text_decoder')

    def shared_embedding_or_output_weight(self):
        """
        This is a convenience method to surface the language model's word embeddings, which is
        necessary for 'finalize_model_grads._allreduce_word_embedding_grads'.
        """
        if self.add_text_decoder:
            return self.text_decoder.shared_embedding_or_output_weight()
        return None

    def _build_image_encoder_model(self, config):
        self.encoder_dp_enable = config.vision_encoder.model_id == "InternViT"

        if get_args().hetero_parallel:
            change_parallel_state('image_encoder')

            self.pp_size = mpu.get_pipeline_model_parallel_world_size()
            self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
            if self.enable_vp:
                self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
                self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            print_rank_0(f'initial: image_encoder pp size is {self.pp_size}')
            print_rank_0(f'initial: image_encoder tp size is {mpu.get_tensor_model_parallel_world_size()}')
            print_rank_0(f'initial: image_encoder cp size is {mpu.get_context_parallel_world_size()}')
            print_rank_0(f'initial: image_encoder dp size is {mpu.get_data_parallel_world_size()}')

        vit_layer_spec = get_vit_layer_spec(config.vision_encoder)
        proj_layer_spec = get_projector_layer_spec(config.vision_projector)

        if self.pp_size <= 1:
            return VisionModel(
                config=config,
                encoder_transformer_layer_spec=vit_layer_spec,
                projector_layer_spec=proj_layer_spec
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
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.vision_encoder.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.vision_encoder.num_layers *= self.vp_size
        return VisionModel(
            config=config,
            encoder_transformer_layer_spec=vit_layer_spec,
            projector_layer_spec=proj_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_audio_encoder_model(self, config):
        if get_args().hetero_parallel:
            change_parallel_state('audio_encoder')
            self.pp_size = mpu.get_pipeline_model_parallel_world_size()
            self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
            if self.enable_vp:
                self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
                self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            print_rank_0(f'initial: audio_encoder pp size is {self.pp_size}')
            print_rank_0(f'initial: audio_encoder tp size is {mpu.get_tensor_model_parallel_world_size()}')
            print_rank_0(f'initial: audio_encoder cp size is {mpu.get_context_parallel_world_size()}')
            print_rank_0(f'initial: audio_encoder dp size is {mpu.get_data_parallel_world_size()}')

        audio_layer_spec = get_audio_layer_spec(config.audio_encoder)

        if self.pp_size <= 1:
            return AudioModel(
                config=config,
                encoder_transformer_layer_spec=audio_layer_spec
            )
        if self.enable_vp:
            if self.pp_size * self.vp_size != len(config.audio_encoder.pipeline_num_layers) * len(
                    config.audio_encoder.pipeline_num_layers[0]):
                raise ValueError(
                    f"The product of pipeline-model-parallel-size and vpp-size must equal to the total number of stage in audio_encoder.pipeline_num_layers, "
                    f"but got pipeline-model-parallel-size: {self.pp_size}, vpp-size: {self.vp_size}, "
                    f"and total number of stage in audio_encoder.pipeline_num_layers: {len(config.audio_encoder.pipeline_num_layers) * len(config.audio_encoder.pipeline_num_layers[0])}.")
        elif self.pp_size != len(config.audio_encoder.pipeline_num_layers):
            raise ValueError(
                f"length of audio_encoder.pipeline_num_layers must equal to pipeline-model-parallel-size, "
                f"but got audio_encoder.pipeline_num_layers length:{len(config.audio_encoder.pipeline_num_layers)} "
                f"and pipeline-model-parallel-size:{self.pp_size}.")

        if self.enable_vp:
            local_num_layers = config.audio_encoder.pipeline_num_layers[self.vp_rank][self.pp_rank]
        else:
            local_num_layers = config.audio_encoder.pipeline_num_layers[self.pp_rank]

        if local_num_layers == 0:
            self.add_audio_encoder = False
            return None

        if self.enable_vp:
            pipeline_start_index = sum(
                sum(vp_layer) for vp_layer in config.audio_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.audio_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank])
            pipeline_end_index = sum(
                sum(vp_layer) for vp_layer in config.audio_encoder.pipeline_num_layers[:self.vp_rank]) + sum(
                config.audio_encoder.pipeline_num_layers[self.vp_rank][:self.pp_rank + 1])
        else:
            pipeline_start_index = sum(config.audio_encoder.pipeline_num_layers[:self.pp_rank])
            pipeline_end_index = sum(config.audio_encoder.pipeline_num_layers[:self.pp_rank + 1])

        pre_process = pipeline_start_index == 0
        post_process = pipeline_end_index == config.audio_encoder.num_layers

        print(
            f"image encoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.audio_encoder.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.audio_encoder.num_layers *= self.vp_size
        return AudioModel(
            config=config,
            encoder_transformer_layer_spec=audio_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_text_decoder_model(self, config):
        if get_args().hetero_parallel:
            change_parallel_state('text_decoder')
            self.pre_process = mpu.is_pipeline_first_stage()
            self.post_process = mpu.is_pipeline_last_stage()
            self.pp_size = mpu.get_pipeline_model_parallel_world_size()
            self.enable_vp = mpu.get_virtual_pipeline_model_parallel_world_size() is not None
            if self.enable_vp:
                self.vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
                self.vp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            self.pp_rank = mpu.get_pipeline_model_parallel_rank()
            print_rank_0(f'initial: text_decoder pp size is {self.pp_size}')
            print_rank_0(f'initial: text_decoder tp size is {mpu.get_tensor_model_parallel_world_size()}')
            print_rank_0(f'initial: text_decoder cp size is {mpu.get_context_parallel_world_size()}')
            print_rank_0(f'initial: text_decoder dp size is {mpu.get_data_parallel_world_size()}')

        if self.pp_size <= 1:
            return MMGPTModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                pre_process=self.pre_process,
                post_process=self.post_process,
                reward_process=self.reward_process
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
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.num_layers = self.pp_size * local_num_layers
        if self.enable_vp:
            config.num_layers *= self.vp_size
        return MMGPTModel(
            config=config,
            transformer_layer_spec=get_llm_layer_spec(config),
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_position_embeddings,
            parallel_output=config.parallel_output,
            position_embedding_type=config.position_embedding_type,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
            pre_process=pre_process,
            post_process=post_process,
            reward_process=self.reward_process
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
            freeze_text_decoder: bool = False,
            freeze_image_encoder: bool = False,
            freeze_audio_encoder: bool = False,
            freeze_audio_projection: bool = False,
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
        if self.add_audio_encoder:
            self.audio_encoder.freeze(freeze_audio_encoder, freeze_audio_projection)
        if self.add_text_decoder and freeze_text_decoder:
            for param in self.text_decoder.parameters():
                param.requires_grad = False

    def compute_loss_with_tensor_parallel(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        args = get_args()
        # To align with torch.nn.CrossEntropyLoss, disable max normalization in vocab_parallel_cross_entropy (comment out)
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)

        # The three loss calculation modes are mutually exclusive:
        # 1. Default behavior (calculate_per_sample_loss=False and calculate_per_token_loss=False):
        #    Calculate the average loss for the micro batch and dividing by micro batch num
        # 2. Token level (calculate_per_token_loss=True):
        #    Keep per-token losses without any aggregation, used for scenarios requiring token-level loss
        # 3. Sample level (calculate_per_sample_loss=True):
        #    Calculate per-sample average loss by first computing the average loss of valid tokens within each sample, then averaging across all samples
        if args.calculate_per_sample_loss:
            loss = loss * (labels > -1)
            batch_mean_loss = loss.sum(dim=1) / (labels > -1).sum(dim=1)
            loss = batch_mean_loss.mean()
        elif args.calculate_per_token_loss:
            pass
        else:
            loss = loss * (labels > -1)
            loss = torch.sum(loss) / torch.sum(labels > -1)

        return loss

    def compute_loss_with_context_parallel(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        args = get_args()
        token_nums = None

        if args.context_parallel_algo == "megatron_cp_algo":
            shift_labels = torch.cat((labels[..., 1:], labels[..., :1]), dim=-1)
            # split and shift labels
            # The default value of the ignore index is -100
            shift_labels[..., -1] = -100
            # shape [batch size, s / cp size] --> [batch size]
            token_nums = (shift_labels > -1).sum(dim=1)
            labels = split_forward_gather_backward_with_megatron_cp(shift_labels, mpu.get_context_parallel_group(), 1)
        elif args.context_parallel_algo == "ulysses_cp_algo":
            # split and shift labels
            shift_labels = labels[..., 1:].contiguous()
            # shape [batch size, s / cp size] --> [batch size]
            token_nums = (shift_labels > -1).sum(dim=1)
            # Calculate the split sizes for each device in context parallelism
            split_gather_sizes = cal_split_sizes(labels.shape[-1], mpu.get_context_parallel_world_size())
            # Reduce the last device's split size by 1 to handle the shifted labels
            split_gather_sizes[-1] = split_gather_sizes[-1] - 1
            labels = split_forward_gather_backward(shift_labels, mpu.get_context_parallel_group(), -1,
                                                   split_gather_sizes, "down")
            if mpu.get_context_parallel_rank() == mpu.get_context_parallel_world_size() - 1:
                logits = logits[..., :-1, :].contiguous()
        elif args.context_parallel_algo == "hybrid_cp_algo":
            # shift labels,
            shift_labels = torch.cat((labels[..., 1:], labels[..., :1]), dim=-1)
            shift_labels[..., -1] = -100  # use padding flag
            token_nums = (shift_labels > -1).sum(dim=1)

            # split shift_labels
            split_gather_sizes = cal_split_sizes(shift_labels.shape[-1],
                                                 get_context_parallel_for_hybrid_ulysses_world_size())

            shift_labels = split_forward_gather_backward(shift_labels, get_context_parallel_group_for_hybrid_ulysses(),
                                                         1, split_gather_sizes, "down")
            labels = split_forward_gather_backward_with_megatron_cp(shift_labels,
                                                                    get_context_parallel_group_for_hybrid_ring(), dim=1)

        loss = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)
        loss = loss * (labels > -1)

        # total_loss shape : [batch size, s]
        total_loss = gather_forward_split_backward(loss, mpu.get_context_parallel_group(), dim=-1)

        # The three loss calculation modes are mutually exclusive:
        # 1. Default behavior (calculate_per_sample_loss=False and calculate_per_token_loss=False):
        #    Calculate the average loss for the micro batch and dividing by micro batch num
        # 2. Token level (calculate_per_token_loss=True):
        #    Keep per-token losses without any aggregation, used for scenarios requiring token-level loss
        # 3. Sample level (calculate_per_sample_loss=True):
        #    Calculate per-sample average loss by first computing the average loss of valid tokens within each sample, then averaging across all samples
        if args.calculate_per_sample_loss:
            # Avoid division by zero
            token_nums_safe = torch.clamp(token_nums, min=1.0)
            batch_mean_loss = total_loss.sum(dim=1) / token_nums_safe
            total_loss = batch_mean_loss.mean()
            token_nums = token_nums.mean()
        elif args.calculate_per_token_loss:
            pass
        else:
            token_nums_sum = torch.sum(token_nums)
            # Avoid division by zero
            if token_nums_sum == 0:
                total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)
            else:
                total_loss = total_loss.sum() / token_nums_sum
            token_nums = token_nums_sum

        return total_loss, token_nums

    def compute_language_model_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        args = get_args()
        loss = None

        # The three loss calculation modes are mutually exclusive:
        # 1. Default behavior (calculate_per_sample_loss=False and calculate_per_token_loss=False):
        #    Calculate the average loss for the micro batch and dividing by micro batch num
        # 2. Token level (calculate_per_token_loss=True):
        #    Keep per-token losses without any aggregation, used for scenarios requiring token-level loss
        # 3. Sample level (calculate_per_sample_loss=True):
        #    Calculate per-sample average loss by first computing the average loss of valid tokens within each sample, then averaging across all samples
        if args.calculate_per_sample_loss:
            batch_size, _, _ = logits.shape
            # To align with huggingface transformers
            if batch_size == 1:
                loss_fct = CrossEntropyLoss()
                logits = logits.view(-1, self.vocab_size)
                labels = labels.view(-1)
                loss = loss_fct(logits.float(), labels)
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
                logits = logits.permute(0, 2, 1).contiguous()
                loss = loss_fct(logits.float(), labels)
                batch_mean_loss = loss.sum(dim=1) / (labels > -1).sum(dim=1)
                loss = batch_mean_loss.mean()
        elif args.calculate_per_token_loss:
            loss_fct = CrossEntropyLoss(reduction='none')
            # Flatten the tokens
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)
            loss = loss_fct(logits.float(), labels)
        else:
            loss_fct = CrossEntropyLoss()
            # Flatten the tokens
            logits = logits.view(-1, self.vocab_size)
            labels = labels.view(-1)
            loss = loss_fct(logits.float(), labels)

        return loss

    def process_multimodal_embeddings(self, input_embeds, input_ids, vit_embeds, audio_embeds, **kwargs):
        deepstack_visual_embeds = []
        if vit_embeds is not None:
            if self.config.sequence_parallel:
                input_embeds = gather_from_sequence_parallel_region(input_embeds)
            input_embeds = input_embeds.transpose(0, 1)  # bsh -> sbh

            image_mask = torch.eq(input_ids, self.img_context_token_id)
            vit_embeds = vit_embeds[:, 0, :]
            indices_tuple = torch.nonzero(image_mask, as_tuple=True)
            input_embeds[indices_tuple] = vit_embeds

            deepstack_image_embeds = kwargs.pop("deepstack_image_embeds", None)
            if deepstack_image_embeds is not None:
                for deepstack_image in deepstack_image_embeds:
                    if self.config.sequence_parallel:
                        deepstack_image = gather_from_sequence_parallel_region(deepstack_image,
                                                                               tensor_parallel_output_grad=False)
                        deepstack_image = deepstack_image[: vit_embeds.shape[0], :]

                    deepstack_emb = deepstack_image.new_zeros(input_embeds.shape)
                    deepstack_emb[indices_tuple] = deepstack_image

                    deepstack_emb = deepstack_emb.transpose(0, 1)
                    if self.config.sequence_parallel:
                        deepstack_emb = tensor_parallel.scatter_to_sequence_parallel_region(deepstack_emb)
                    deepstack_visual_embeds.append(deepstack_emb)

            if 'input_features' in kwargs:
                audio_mask = torch.eq(input_ids, 151646).unsqueeze(-1).expand_as(input_embeds)
                audio_embeds = audio_embeds.to(input_embeds.device, input_embeds.dtype)
                input_embeds = input_embeds.masked_scatter(audio_mask, audio_embeds)

            input_embeds = input_embeds.transpose(0, 1)  # sbh -> bsh
        return input_embeds, deepstack_visual_embeds

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
            transfer: Optional[numpy.ndarray] = None,
            *args, **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:

        # hetero pipeline use
        hetero_pp = hasattr(mpu, "_IS_HETERO_PP_MOUDLE") and mpu._IS_HETERO_PP_MOUDLE

        # MM_GRPO use, if llm_only is True, directly get vit_embeds
        deepstack_visual_embeds = None
        if self.add_image_encoder and self.image_encoder.pre_process and kwargs.get('llm_only', False):
            vit_embeds = kwargs.get('vit_embeds').unsqueeze(1)
        elif self.add_image_encoder and pixel_values is not None and not hetero_pp:
            text_img_num = (input_ids == self.vision_start_token_id).sum(dim=1) if get_args().hetero_parallel else None
            encoder_out = self.image_encoder(pixel_values, image_grid_thw, text_img_num)
            if isinstance(encoder_out, tuple) and len(encoder_out) == 2:
                vit_embeds, deepstack_image_embeds = encoder_out
                kwargs["deepstack_image_embeds"] = deepstack_image_embeds
            else:
                vit_embeds = encoder_out
            if get_args().encoder_dp_balance and self.encoder_dp_enable:
                vit_embeds = EncoderBalanceComm.apply(
                    vit_embeds,
                    mpu.get_data_parallel_group(),
                    transfer
                )

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

        # MM_GRPO use, if vit_only is True, only calculate vit_embeds and return
        if kwargs.get('vit_only', False) and self.image_encoder.post_process:
            return {"vit_embeds": vit_embeds}

        audio_embeds = None
        if self.add_audio_encoder and 'input_features' in kwargs and not hetero_pp:
            audio_embeds = self.audio_encoder(kwargs['input_features'], kwargs['feature_attention_mask'])

        # hetero pipeline use
        if hasattr(mpu, "_IS_HETERO_PP_MOUDLE") and not mpu._IS_HETERO_PP_MOUDLE:
            change_parallel_state('image_encoder')
            return [vit_embeds, audio_embeds]

        if self.add_text_decoder:
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                if kwargs.get('vit_embedings') is not None or kwargs.get('audio_embedings') is not None:
                    vit_embeds = kwargs.get('vit_embedings')
                    audio_embeds = kwargs.get('audio_embedings')
                input_embeds, deepstack_visual_embeds = self.process_multimodal_embeddings(input_embeds, input_ids,
                                                                                           vit_embeds, audio_embeds,
                                                                                           **kwargs)
            else:
                input_embeds = None

            attention_mask, position_ids = prepare_positionsids_mask_for_llm(config=self.config, input_ids=input_ids,
                                                                             inference_params=inference_params,
                                                                             attention_mask=attention_mask,
                                                                             position_ids=position_ids,
                                                                             image_grid_thw=image_grid_thw,
                                                                             rope_deltas=rope_deltas,
                                                                             inputs_embeds=input_embeds,
                                                                             cache_position=cache_position,
                                                                             **kwargs)
            extra_block_kwargs = {}
            if deepstack_visual_embeds is not None and len(deepstack_visual_embeds) > 0:
                extra_block_kwargs['deepstack_visual_embeds'] = deepstack_visual_embeds

            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=None,
                inference_params=inference_params,
                extra_block_kwargs=extra_block_kwargs,
            )

            if self.text_decoder.post_process:
                output = output.contiguous().float()
                loss_dict = {}
                if labels is not None:
                    if mpu.get_context_parallel_world_size() > 1:
                        loss, token_nums = self.compute_loss_with_context_parallel(output, labels)

                        loss_dict["loss"] = loss
                        loss_dict["token_nums"] = token_nums

                        return {
                            "loss_dict": loss_dict,
                            "logits": output
                        }
                    else:
                        # output shape [b, s, vocab_size]
                        shift_logits = output[..., :-1, :].contiguous()
                        # labels shape [b, s]
                        shift_labels = labels[..., 1:].contiguous()

                        if mpu.get_tensor_model_parallel_world_size() > 1:
                            loss = self.compute_loss_with_tensor_parallel(shift_logits, shift_labels)
                        else:
                            loss = self.compute_language_model_loss(shift_logits, shift_labels)

                        loss_dict["loss"] = loss
                        loss_dict["loss_mask"] = shift_labels > -1

                        return {
                            "loss_dict": loss_dict,
                            "logits": output
                        }

                return {
                    "loss": None,
                    "logits": output
                }

        return output
