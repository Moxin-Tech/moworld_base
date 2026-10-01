# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Literal
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.transformer.enums import ModelType
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.utils import deprecate_inference_params, WrappedTensor

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.mcore.process_group_configs import ProcessGroupCollection
from mindspeed_mm.mcore.models.qwen3_5.modules import (
    Qwen3_5VisionTransformerBlock,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    Qwen3_5VisionRotaryEmbedding,
    Qwen3_5TextTransformerBlock,
    Qwen3_5MultimodalRotaryEmbedding
)
from mindspeed_mm.mcore.models.qwen3_5.utils import (
    AllGatherVisionEmbeddings,
    collapse_thw,
    get_vision_cp_data,
    preprocess_packed_seqs,
    cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    CudaGraphScope,
    is_using_quantization_scales,
    process_mtp_loss
)
from mindspeed_mm.mcore.models.qwen3_5.spec import get_qwe3_5_vit_layer_local_spec, get_vision_patch_merger_spec


class Qwen3_5Model(MegatronModule):
    """Qwen3.5 multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
        vision_transformer_config (Qwen3_5ConfigHF): HF config for the vision model.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This
            is typically True for training and False for inference.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings
            in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline
            parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
    """

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        vision_transformer_config: ModuleSpec,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id

        self.square_merge_size = language_transformer_config.spatial_merge_size**2

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        # process groups
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp
        if not hasattr(self.pg_collection, "embd"):
            raise AssertionError(
                "pg_collection must have a embd. In previous version, it used default "
                "`parallel_state.default_embedding_ranks` to create the process group."
                "If you are using the default process group, please use"
                "`parallel_state.get_embedding_group()` "
                "If you don't need embd_group, you need to explicitly set it to None."
            )
        self.embd_group = pg_collection.embd
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        if self.pre_process:
            vision_transformer_layer_spec = get_qwe3_5_vit_layer_local_spec(vision_transformer_config)
            vision_patch_merger_spec = get_vision_patch_merger_spec(vision_transformer_config)

            self.vision_model = Qwen3_5VisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                vision_patch_merger_spec,
                pre_process=True,
                post_process=True,
                pg_collection=pg_collection,
            )

        self.language_model = Qwen3_5TextModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    @property
    def decoder(self):
        """Expose language model decoder for mcore inference compatibility.

        mcore's MambaInferenceStateConfig.from_model() calls get_attr_wrapped_model(model, "decoder"),
        which only traverses .module wrappers. VLM models store the decoder under language_model.decoder,
        so we expose it here to allow the Mamba check to run and correctly return None.
        """
        return getattr(self.language_model, "decoder", None)

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        if len(input_tensor) != 1:
            raise ValueError("input_tensor should only be length 1 for Qwen3.5")

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        if freeze_vision_model and not freeze_vision_projection:
            if self.vision_model is not None:
                for param in self.vision_model.merger.parameters():
                    param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,  # can set at dataset
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        # can set at dataset
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        mm_token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the Qwen3.5 model.
        # there is a workaround for supporting sequence packing with context parallelism
        # cp split with sequence packing will make model lose vision token information, so we need to keep
        # the original input_ids and pack them after vision embedding is calculated,
        # cooporate with verl's models/mcore/model_forward.py
        # pack the combined_embeddings to thd here, we check if packed_seq_params is None to determine if we need to pack the combined_embeddings to thd
        # this function needs the position_ids and attention_mask in BSHD format, no matter use packed_seq or not

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            mm_token_type_ids (torch.Tensor): Token type IDs from transformers >= 5.3.0 processors.
                Not used by Qwen3.5 (which computes its own rope positions).

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """

        del inference_context, runtime_gather_output, mm_token_type_ids  # Unused, kept for API compatibility
        if inference_params is None:
            raise ValueError("not support inference")

        vision_grid_thw = None
        vision_data = None
        vision_mask = None

        # position ids is computed within the model
        position_ids = None

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        # input_ids to pass to the language model for MTP (Multi-Token Prediction).
        # MTP's _get_embeddings rolls input_ids to generate future-token embeddings,
        # so it must be a real tensor. For packed sequences we use the THD-format
        # input_ids_thd (updated below); for regular sequences we use input_ids as-is.
        lm_input_ids = input_ids

        if self.pre_process:
            # can reorganize_inputs at dataset
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=input_ids,  # [1, 840]
                pixel_values=pixel_values,  # [1064, 1536]
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                square_merge_size=self.square_merge_size,
            )

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    if cp_img_num is None:
                        vision_data, vision_grid_thw, cp_img_num, images_padded = cp_split(
                            cp_size,
                            vision_data,
                            vision_grid_thw,
                        )
                    vision_data, vision_grid_thw, seqlen_on_cp_ranks = get_vision_cp_data(
                        vision_data,
                        vision_grid_thw,
                        self.square_merge_size,
                        cp_img_num,
                        images_padded,
                        cp_rank,
                        cp_size,
                    )
                    vision_grid_thw = collapse_thw(vision_grid_thw)
                if vision_data.shape[0] > 0:
                    vision_embeds = self.vision_model(
                        hidden_states=vision_data,  # [1064, 1536]
                        grid_thw=vision_grid_thw,
                    )
                else:
                    vision_embeds = torch.zeros(
                        (0, self.language_model.config.hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                        cp_group=self.pg_collection.cp,
                    )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[vision_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)
            if packed_seq_params is not None:
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
                input_ids_thd, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )
                lm_input_ids = input_ids_thd
                _, _, vision_mask_thd = reorganize_inputs(
                    input_ids=input_ids_thd,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_input_mask=image_input_mask,
                    video_input_mask=video_input_mask,
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                    square_merge_size=self.square_merge_size,
                )

                vision_mask = vision_mask_thd
                combined_embeddings_thd = (
                    preprocess_packed_seqs(
                        combined_embeddings.transpose(0, 1).contiguous(),
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    .transpose(0, 1)
                    .contiguous()
                )
                combined_embeddings = combined_embeddings_thd

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        else:
            combined_embeddings = None
            # On non-pre_process PP stages (e.g. the last stage where MTP runs),
            # convert lm_input_ids to THD format so it matches position_ids.
            if packed_seq_params is not None:
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
                lm_input_ids, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )

        if position_ids is None:
            # BSHD
            # Megatron uses 4D bool masks ([B|1,1,S,S], True=masked); HF uses 2D keep masks ([B,S], 1=keep)
            # For simplicity, we set hf_attention_mask to None.
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=None,
            )  #  [3*b*s]
            if packed_seq_params is not None:
                # convert position_ids to THD format
                position_ids = (
                    preprocess_packed_seqs(
                        position_ids.permute(1, 2, 0),
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    .permute(2, 0, 1)
                    .contiguous()
                )
                attention_mask = None
                self.language_model.rotary_pos_emb.is_thd_format = True

        output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,  # None in encoder
            attention_mask=None,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,  # Added for THD training compatibility
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        return output


class Qwen3_5VisionModel(MultiModalModule):
    """Qwen3 ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        patch_merger_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
        self,
        transformer_config,
        transformer_layer_spec: ModuleSpec,
        patch_merger_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=transformer_config)
        self.spatial_merge_size = transformer_config.spatial_merge_size
        self.patch_size = transformer_config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.pg_collection = pg_collection
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.tp_group = self.pg_collection.tp

        self.patch_embed = Qwen3_5VisionPatchEmbed(transformer_config)
        self.pos_embed = nn.Embedding(transformer_config.num_position_embeddings, transformer_config.hidden_size)
        self.num_grid_per_side = int(transformer_config.num_position_embeddings**0.5)

        head_dim = transformer_config.hidden_size // transformer_config.num_attention_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        self.decoder = Qwen3_5VisionTransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
            pg_collection=self.pg_collection,
        )

        self.merger = None
        if self.post_process:
            self.merger = Qwen3_5VisionPatchMerger(
                transformer_config,
                patch_merger_spec,
                use_postshuffle_norm=False,
            )

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=self.pos_embed.weight.dtype,
            device=self.pos_embed.weight.device,
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        seq_len, _ = hidden_states.size()

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, 1, 1, -1).repeat(1, 1, 1, 2)
        hidden_states = hidden_states[:, None]

        hidden_states = self.decoder(
            hidden_states=hidden_states,  # [1064, 1, 1152]
            attention_mask=None,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=self.build_packed_seq_params(grid_thw),
            **(extra_block_kwargs or {}),
        )

        hidden_states = self.merger(hidden_states)

        # Encodes images into continuous embeddings that can be forwarded to the language model.
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        hidden_states = torch.split(hidden_states, split_sizes)
        hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states

    def build_packed_seq_params(
        self,
        grid_thw: Optional[torch.Tensor],
    ) -> PackedSeqParams:
        seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = seqlens.cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()

        max_seqlen_q = seqlens.max()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format="thd",
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
        )


class Qwen3_5TextModel(GPTModel):
    def __init__(
        self,
        config,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope", "mrope", "none"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
        )

        self.pg_collection = pg_collection

        # rebuild rope
        self.rotary_pos_emb = Qwen3_5MultimodalRotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            cp_group=self.pg_collection.cp,
        )
        self.mrope_section = self.config.mrope_section
        if self.mrope_section is None:
            raise ValueError(
                "mrope require mrope_section setting, but we got None from TransformerConfig"
            )

        # rebuild the transformer block
        self.decoder = Qwen3_5TextTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        # `_preprocess` can optionally return an extra fused cos/sin buffer (for
        # flash decode). Match the upstream GPTModel handling to avoid unpack
        # errors when six values are returned.
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = preproc_output[:5]

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            # Qwen3 VL blocks do not currently consume fused cos/sin; pass along
            # the standard components only.
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        # MTP calls self.embedding directly (bypassing the manual SP scatter that
        # model.py does for the combined VL embeddings). Temporarily wrap the embedding
        # to apply the SP scatter so its output shape matches hidden_states.
        # We write to self.__dict__ directly to bypass nn.Module.__setattr__'s type
        # check, which rejects non-Module values for registered child modules.
        _shadow_embedding = False
        if self.mtp_process and self.config.sequence_parallel:
            _original_embedding = self.embedding

            def _sp_scatter_embedding(input_ids, position_ids):
                out = _original_embedding(input_ids=input_ids, position_ids=position_ids)
                return tensor_parallel.scatter_to_sequence_parallel_region(out)

            _sp_scatter_embedding.word_embeddings = _original_embedding.word_embeddings
            self.__dict__["embedding"] = _sp_scatter_embedding
            _shadow_embedding = True

        result = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

        if _shadow_embedding:
            del self.__dict__["embedding"]

        return result

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor, ignore_index=-100) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """

        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        labels = F.pad(labels, (0, 0, 0, 1), value=ignore_index)
        shift_labels = labels[1:, ...].contiguous()
        loss_mask = shift_labels > -1
        losses = tensor_parallel.vocab_parallel_cross_entropy(logits, shift_labels)
        loss = (losses * loss_mask).view(-1).sum() / loss_mask.sum()
        return loss

    def _preprocess(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        padding_mask: Optional[Tensor] = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            if padding_mask is not None:
                if padding_mask.shape != input_ids.shape:
                    raise ValueError(
                        f"padding_mask shape {padding_mask.shape} does not match "
                        f"input_ids shape {input_ids.shape}"
                    )
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if padding_mask is not None and self.config.sequence_parallel:
                padding_mask = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        padding_mask.transpose(0, 1).contiguous()
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        # this is used to store combined cos/sin embeddings, exclusively for flash infer rope
        rotary_pos_cos_sin = None

        if self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(
                    position_ids,
                    self.mrope_section,
                    cp_group=packed_seq_params.cp_group if packed_seq_params is not None else None,
                )
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implemented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            in_inference_mode
            and (
                (
                    self.config.cuda_graph_impl == "local"
                    and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
                )
                or self.config.flash_decode
            )
            and inference_context.is_static_batching()
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
        else:
            sequence_len_offset = None

        if in_inference_mode:
            # Clear the outputs for padding tokens when using dynamic batching with
            # quantization scales to avoid corrupting amax calculations
            if inference_context.is_dynamic_batching() and is_using_quantization_scales(
                self.config
            ):
                decoder_input[inference_context.padding_slice] = 0.0

            # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
            # reference held by this caller function, enabling early garbage collection for
            # inference. Skip wrapping if decoder_input is logged after decoder completion.
            if not has_config_logger_enabled(self.config):
                decoder_input = WrappedTensor(decoder_input)

        preproc_output = (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        )
        if rotary_pos_cos_sin is not None:
            # only in the case of flashinfer fused rope will we
            # return this extra tensor
            # this is for backwards compatibility with
            # legacy unit tests, which break if you
            # return a 7 tuple instead of 6.
            preproc_output += (rotary_pos_cos_sin,)

        return preproc_output

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    ):
        """Postprocesses decoder hidden states to generate logits or compute loss.

        Applies Multi-Token Prediction if enabled, generates output logits through
        the output layer, and computes language model loss when labels are provided.
        """
        in_inference_mode = inference_context is not None and not self.training

        # Check if speculative decoding is active. When it is, MTP must be
        # computed *after* verification so that it is conditioned on verified
        # tokens rather than stale speculative tokens from the previous step.
        if is_spec_decode is None:
            is_spec_decode = (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.num_speculative_tokens > 0
            )

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if mtp_in_postprocess and not (in_inference_mode or is_spec_decode):
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=None,  # MTP layers don't use KV cache
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                embedding=self.embedding,
                **(extra_block_kwargs or {}),
            )

        if not self.post_process:
            return hidden_states

        if self.config.mtp_num_layers:
            if in_inference_mode or is_spec_decode:
                # Cache decoder hidden states for serial MTP computation
                # after speculative token verification.
                self._decoder_hidden_states_cache = hidden_states
            else:
                # In training/eval, use the utility function for processing MTP loss/scaling.
                hidden_states = process_mtp_loss(
                    hidden_states=hidden_states,
                    labels=labels,
                    loss_mask=loss_mask,
                    output_layer=self.output_layer,
                    output_weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                    is_training=self.training,
                    compute_language_model_loss=self.compute_language_model_loss,
                    config=self.config,
                    cp_group=self.pg_collection.cp,
                    packed_seq_params=packed_seq_params,
                    scale_logits_fn=self._scale_logits if self.config.use_mup else None,
                )
        sequence_parallel_override = False

        if in_inference_mode and inference_context.config.materialize_only_last_token_logits:
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [S, B, H] (with B=1) to [1, S, H] for logit extraction,
                # then back to [S’, B, H] for the output layer.
                reshaped = hidden_states.squeeze(1).unsqueeze(0)
                hidden_states = inference_context.last_token_logits(reshaped).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        # Apply MuP output scaling to logits
        logits = self._scale_logits(logits)

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            if not (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.config.materialize_only_last_token_logits
            ):
                raise AssertionError()
            self.output_layer.sequence_parallel = True

        if has_config_logger_enabled(self.config):
            payload = OrderedDict(
                {
                    'input_ids': input_ids,
                    'position_ids': position_ids,
                    'attention_mask': attention_mask,
                    'decoder_input': decoder_input,
                    'logits': logits,
                }
            )
            log_config_to_disk(self.config, payload, prefix='input_and_logits')

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def _scale_logits(self, logits: Tensor) -> Tensor:
        """Apply MuP output scaling to logits.

        When MuP is enabled, scales logits by mup_output_mult (auto-set to 1/width_mult
        if left at default) to keep output variance stable across widths.

        Args:
            logits (Tensor): Raw logits from the output layer.

        Returns:
            Tensor: Scaled logits if MuP is enabled and mup_output_mult != 1.0,
                    otherwise unchanged logits.
        """
        if not self.config.use_mup:
            return logits
        if self.config.mup_output_mult != 1.0:
            return logits * self.config.mup_output_mult
        return logits
