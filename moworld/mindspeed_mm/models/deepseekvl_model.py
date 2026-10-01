# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Dict, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from einops import rearrange, repeat

from megatron.core import InferenceParams, mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module_spec.get_layer_spec import get_vit_layer_spec, get_llm_layer_spec, \
    get_projector_layer_spec
from mindspeed_mm.models.vision.vision_model import VisionModel
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.text_encoder.text_encoder import TextEncoder
from mindspeed_mm.models.text_decoder.moe_model import MOEModel
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import prepare_positionsids_mask_for_llm


class VLMModel(MultiModalModule):
    """
    Vision-Language multi-modal model.
    VLMModel is an assembled model, which include image_encoder, text_decoder model.

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

        self.text_encoder = None
        self.image_encoder = None
        self.video_encoder = None
        self.text_decoder = None

        self.share_embeddings_and_output_weights = not getattr(config.text_decoder, 'untie_embeddings_and_output_weights', True)
        self.position_embedding_type = config.text_decoder.position_embedding_type
        self.vocab_size = config.text_decoder.vocab_size


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


    def shared_embedding_or_output_weight(self):
        """
        This is a convenience method to surface the language model's word embeddings, which is
        necessary for 'finalize_model_grads._allreduce_word_embedding_grads'.
        """
        if self.add_text_decoder:
            return self.text_decoder.shared_embedding_or_output_weight()
        return None

    def _build_image_encoder_model(self, config):
        vit_layer_spec = get_vit_layer_spec(config.vision_encoder)
        proj_layer_spec = get_projector_layer_spec(config.vision_projector)


        # image token format style
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # Special token for format image token sequence
        embed_std = 1 / torch.sqrt(torch.tensor(config.vision_projector.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(config.vision_projector.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(config.vision_projector.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}")
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")

        if self.pp_size <= 1:
            return VisionModel(
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
        return VisionModel(
            config=config,
            encoder_transformer_layer_spec=vit_layer_spec,
            projector_layer_spec=proj_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
        )

    def _build_text_decoder_model(self, config):
        if self.pp_size <= 1:
            return MOEModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
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
        first_k_dense_replace = config.first_k_dense_replace - pipeline_start_index

        print(
            f"text decoder pipeline config:\
            pp_rank:{self.pp_rank},\
            pre_process:{pre_process},\
            post_process:{post_process},\
            local_num_layers:{local_num_layers}"
        )
        # num_layers will be divided by pp_size in TransformerBlock from megatron.core
        config.num_layers = self.pp_size * local_num_layers
        config.first_k_dense_replace = first_k_dense_replace

        return MOEModel(
                config=config,
                transformer_layer_spec=get_llm_layer_spec(config),
                vocab_size=config.vocab_size,
                max_sequence_length=config.max_position_embeddings,
                parallel_output=config.parallel_output,
                position_embedding_type=config.position_embedding_type,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                rotary_base=config.rope_theta if getattr(config, 'rope_theta', None) else config.rotary_base,
                pre_process=pre_process,
                post_process=post_process
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
        shift_logits = shift_logits.view(-1, self.vocab_size)
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

    def prepare_images_input(
            self,
            images: torch.FloatTensor,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **kwargs
    ):

        if images is None or images_spatial_crop.sum() == 0:
            return None
        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(images[idx, :batch_num_tiles[idx]])

        # the shape is: [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        if total_tiles.shape[0] != sum(batch_num_tiles):
            raise AssertionError
        if total_tiles.shape[0] == 0:
            return None
        return total_tiles

    def combine_images_embeds(
            self,
            input_embeds: torch.FloatTensor,
            images_embeds: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **kwargs
    ):
        """

        Args:
            input_embeds (torch.FloatTensor): [T, b, D]
            images_embeds (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        input_embeds = input_embeds.transpose(0, 1) # [T, b, D] -> [b, T, D]

        _, hw, n_dim = images_embeds.shape
        h = w = int(hw ** 0.5)

        # Pad the image token sequence according to self.tile_tag & self.global_view_pos.
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # the shape is: [hw, D]
                global_features = images_embeds[tile_index]

                # the shape is: [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == "2D":

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # the shape transformation is: cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w
                    )

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(
                        self.image_newline,
                        "d -> (th h) 1 d",
                        th=num_height_tiles,
                        h=h
                    )

                    # the shape is: [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # the shape is: [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0)

                else:
                    # abandoned, this logic will not be executed in practice.
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )
                    local_features = torch.cat(
                        [self.tile_indicators[1:num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1
                    )
                    local_features = rearrange(local_features, 'crop_num hw d -> (crop_num hw) d')

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                indices_tuple = torch.nonzero(images_seq_mask[idx].unsqueeze(-1), as_tuple=True)
                input_embeds[indices_tuple] = images_in_this_batch
        # [b, s, h] -> [s, b, h]
        input_embeds = input_embeds.transpose(0, 1)
        return input_embeds


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,

            images: Optional[torch.Tensor] = None,
            images_seq_mask: Optional[torch.Tensor] = None,
            images_spatial_crop: Optional[torch.Tensor] = None,

            labels: Optional[torch.Tensor] = None,
            inference_params: Optional[InferenceParams] = None,
            **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:

        images = self.prepare_images_input(images, images_spatial_crop)
        if self.add_image_encoder and images is not None:
            vit_embeds = self.image_encoder(images)
            if len(vit_embeds.shape) == 2:
                vit_embeds = vit_embeds.reshape(-1, 1, vit_embeds.shape[-1]).clone()
            output = vit_embeds
        else:
            vit_embeds = self.input_tensor

        if self.add_text_decoder:
            input_embeds = None
            if self.text_decoder.pre_process:
                input_embeds = self.text_decoder.embedding(input_ids=input_ids, position_ids=position_ids).clone()
                if vit_embeds is not None:
                    if self.config.sequence_parallel:
                        input_embeds = gather_from_sequence_parallel_region(input_embeds)
                    input_embeds = self.combine_images_embeds(input_embeds, vit_embeds, images_seq_mask, images_spatial_crop)
                    if self.config.sequence_parallel:
                        input_embeds = scatter_to_sequence_parallel_region(input_embeds)
            attention_mask, position_ids = \
                prepare_positionsids_mask_for_llm(
                    config=self.config,
                    input_ids=input_ids,
                    inference_params=inference_params,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs
                )

            output = self.text_decoder(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=input_embeds,
                labels=labels,
                inference_params=inference_params,
                **kwargs
            )

            if self.text_decoder.post_process:
                logits = output[0]
                logits = logits.contiguous().float()
                loss = output[1]
                if labels is not None:
                    # if use TP then must use compute_megatron_loss, if do not use TP, then two loss are ok, but they are not equal
                    global_args = get_args()
                    if global_args.tensor_model_parallel_size > 1:
                        loss = self.compute_megatron_loss(logits, labels)
                    else:
                        loss = self.compute_loss(logits, labels)
                return {
                    "loss": loss,
                    "logits": logits
                }
        return output
