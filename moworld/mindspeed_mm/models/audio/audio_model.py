# Copyright 2025 The Qwen team; Alibaba Group and the HuggingFace Inc. team. All rights reserved.

import math

import torch
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig, ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training import get_args
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (
    cal_split_sizes,
    split_forward_gather_backward,
    gather_forward_split_backward
)
from torch import nn as nn
from torch.nn import functional as F

from mindspeed_mm.models.audio.omni_audio_encoder import SinusoidsPositionEmbedding, AudioLinear
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.vision.vision_encoders.vision_transformer_block import Qwen2VLVisionTransformerBlock


class AudioModel(MultiModalModule):
    """
    Instantiate a audio encoder model from config.

    Args:
        config (dict): the general config for audio Model
        {
            "audio_encoder": {...},  # Config for the image encoder.
            "audio_projector": {...},  # Config for the image projector.
            "drop_audio_class_token": (bool),  # Drop audio class token(s) before input to the text decoder.
        }
    """

    def __init__(
            self,
            config: TransformerConfig,
            encoder_transformer_layer_spec: ModuleSpec = None,
            projector_layer_spec: ModuleSpec = None,
            pre_process: bool = True,
            post_process: bool = True,
            *args,
            **kwargs
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = config.audio_encoder is not None
        # When enabling PP, projector exists only on the last rank, set to None by default (otherwise freeze is affected).
        self.projector = None
        self.encoder = None
        if self.add_encoder:
            self.encoder = AUDIO_ENCODER_MAPPINGS[config.audio_encoder.model_id](
                config=config.audio_encoder,
                transformer_layer_spec=encoder_transformer_layer_spec,
                pre_process=self.pre_process,
                post_process=self.post_process,
            )

    def set_input_tensor(self, input_tensor):
        self.encoder.set_input_tensor(input_tensor)

    def freeze(
            self,
            freeze_encoder: bool = False,
            freeze_projector: bool = False
    ):
        """
        Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_encoder (bool): Freeze the image encoder module.
            freeze_projection (bool): Freeze the image projector module.
        """

        modules = []
        if freeze_encoder and self.encoder is not None:
            modules.append(self.encoder)
        if freeze_projector and self.projector is not None:
            modules.append(self.projector)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor = None, *args,
                **kwargs) -> torch.Tensor:
        if self.add_encoder:
            encoder_out = self.encoder(input_features=input_features, feature_attention_mask=feature_attention_mask)
        else:
            raise ValueError("add_encoder error!")
        return encoder_out


class OmniAudioEncoder(MultiModalModule):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Qwen2_5OmniAudioEncoderLayer`].

    Args:
        config: Qwen2_5OmniAudioEncoderConfig
    """

    def __init__(self,
                 config: TransformerBlock,
                 transformer_layer_spec: ModuleSpec,
                 pre_process: bool = True,
                 post_process: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__(config)
        self.pre_process = pre_process
        self.post_process = post_process
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.blocks = Qwen2VLVisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = AudioLinear(config.d_model, config.output_dim)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
            self,
            input_features,
            feature_attention_mask,
    ):
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        aftercnn_lens, _ = self._get_feat_extract_output_lengths(audio_feature_lengths)
        feature_lens = (
            audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        )
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
                                      : padded_embed.shape[1], :
                                      ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)
        hidden_states = hidden_states.unsqueeze(0).transpose(0, 1)
        seq_len, _, _ = hidden_states.shape
        if mpu.get_context_parallel_world_size() > 1:
            split_gather_sizes = cal_split_sizes(hidden_states.shape[0], mpu.get_context_parallel_world_size())
            hidden_states = split_forward_gather_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "down"
            )

        if get_args().use_flash_attn:
            attention_mask = None
        else:
            attention_mask = torch.full(
                [1, seq_len, seq_len], torch.finfo(hidden_states.dtype).min, device=hidden_states.device,
                dtype=torch.bool
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        hidden_states = self.blocks(hidden_states, attention_mask=attention_mask, cu_seqlens=cu_seqlens)
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "up"
            )

        hidden_states = hidden_states.squeeze(1)
        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        return torch.cat(token_audio_list, dim=0)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=tensor_list[0].dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


AUDIO_ENCODER_MAPPINGS = {
    "qwen_omni": OmniAudioEncoder,
}
