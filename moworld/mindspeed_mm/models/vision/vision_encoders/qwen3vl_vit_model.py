# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

from typing import Tuple, Optional
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
    HAS_QWEN3VL_TF = True
except ImportError:
    HAS_QWEN3VL_TF = False
from megatron.core.transformer.transformer_config import TransformerConfig


from megatron.core import mpu
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region, gather_from_sequence_parallel_region
from megatron.training import get_args

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import cal_split_sizes, gather_forward_split_backward
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.communications import split_forward_gather_backward
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import PatchEmbed, VisionRotaryEmbedding
from mindspeed_mm.models.vision.vision_encoders.vision_transformer_block import Qwen3VLVisionTransformerBlock


if HAS_QWEN3VL_TF:
    class Qwen3VLTextRotaryEmbedding_llm(Qwen3VLTextRotaryEmbedding):
        def __init__(self, config: Optional[TransformerConfig] = None):
            super().__init__(config=config)
            # head_dim is set to hidden_size // num_attention_heads by default，but they are not equal in qwen3vl.
            # Should be overwritten by "kv_channels"
            self.config.head_dim = self.config.kv_channels
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

        @torch.no_grad()
        def forward(self, x_device, x_dtype, position_ids, unsqueeze_dim=1):
            if "dynamic" in self.rope_type:
                self._dynamic_frequency_update(position_ids, device=x_device)
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
            position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

            device_type = x_device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                freqs = super().apply_interleaved_mrope(freqs, self.mrope_section)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = (emb.cos() * self.attention_scaling).unsqueeze(unsqueeze_dim).permute(2, 0, 1, 3).contiguous()
                sin = (emb.sin() * self.attention_scaling).unsqueeze(unsqueeze_dim).permute(2, 0, 1, 3).contiguous()
            return torch.concat((cos, sin), dim=-1).to(dtype=x_dtype)
else:
    class Qwen3VLTextRotaryEmbedding_llm:
        def __init__(self, config: Optional[TransformerConfig] = None):
            raise NotImplementedError("transformers should be >=4.57.0.dev0 for using Qwen3VL")


class Qwen3VLViT(MultiModalModule):
    """
    Qwen2VLViT vision model.
    Instantiate a Qwen2VLViT model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            pre_process: bool = True,
            post_process: bool = True,
            *args,
            **kwargs,
    ) -> None:
        # projector_config is used for building deepstack layer in Qwen3VLVisionTransformerLayer
        setattr(config, "projector_config", kwargs.get("projector_config", None))
        super().__init__(config=config)

        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.pre_process = pre_process
        self.post_process = post_process

        if self.pre_process:
            self.patch_embed = PatchEmbed(
                patch_size=config.patch_size,
                temporal_patch_size=config.temporal_patch_size,
                in_channels=config.in_channels,
                embed_dim=config.hidden_size,
                bias=config.add_bias_conv
            )

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = Qwen3VLVisionTransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            post_layer_norm=False,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.pre_process = pre_process
        self.post_process = post_process

        if self.pre_process:
            self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.num_grid_per_side = int(config.max_position_embeddings**0.5)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self.unfreeze_param_names = ['pos_embed', 'deepstack_layer']

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
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h.item())
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w.item())

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
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.reshape((t, h // merge_size, merge_size, w // merge_size, merge_size, -1))
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def pad_to_sequence_parallel(self, images, image_grid_thw):
        """
        Adjust image patches for tensor parallelism (TP) by padding to match effective TP size.

        Args:
            images (torch.Tensor): [num_patches, patch_dim] input patch tensor
            image_grid_thw (torch.Tensor): [num_patches, 3] patch grid info (T,H,W)

        Notes:
            - Effective TP size = tensor_model_parallel_size * (spatial_merge_size²)
            - No op if TP size ≤ 1
        """
        all_patch_num = images.shape[0]
        res_dim = 0
        if get_args().tensor_model_parallel_size <= 1:
            return images, image_grid_thw, all_patch_num, res_dim

        tp_size = get_args().tensor_model_parallel_size
        effective_tp_size = tp_size * (self.spatial_merge_size ** 2)

        res_dim = all_patch_num % effective_tp_size
        pad_size = 0
        if res_dim != 0:
            pad_size = effective_tp_size - res_dim  # patch to lcm of tp size and all_patch_num
            zero_tensor = torch.zeros(pad_size, images.shape[1], dtype=images.dtype, device='npu')
            images = torch.cat((images, zero_tensor), dim=0)
            pad_thw = torch.tensor([[1, 2, pad_size // 2]], dtype=image_grid_thw.dtype, device='npu')  # s, t
            image_grid_thw = torch.cat((image_grid_thw, pad_thw), dim=0)

        return images, image_grid_thw, all_patch_num, res_dim

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward function of the Qwen2VL ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        """
        all_patch_num, res_dim = None, None
        if self.pre_process:
            if pixel_values is None or grid_thw is None:
                raise ValueError('You have to specify pixel_values and grid_thw')
            else:
                pixel_values, grid_thw, all_patch_num, res_dim = self.pad_to_sequence_parallel(pixel_values,
                                                                                               grid_thw)
                hidden_states = self.patch_embed(pixel_values)

        else:
            hidden_states = None

        rotary_pos_emb = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + rotary_pos_emb
        hidden_states = hidden_states.unsqueeze(1)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len = hidden_states.shape[0] if hidden_states is not None else pixel_values.shape[-2]
        window_index = None
        window_mask = None
        cu_window_seqlens = None

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        if get_args().use_flash_attn:
            attention_mask = None
            window_mask = None
        else:
            attention_mask = torch.full(
                [1, seq_len, seq_len], torch.finfo(pixel_values.dtype).min, device=pixel_values.device,
                dtype=torch.bool
            )
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        if get_args().sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)

        if mpu.get_context_parallel_world_size() > 1:
            split_gather_sizes = cal_split_sizes(hidden_states.shape[0], mpu.get_context_parallel_world_size())
            rotary_pos_emb = split_forward_gather_backward(
                rotary_pos_emb,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "down"
            )
            hidden_states = split_forward_gather_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "down"
            )

        cos_cache = rotary_pos_emb.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        sin_cache = rotary_pos_emb.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
        rotary_pos_emb = torch.concat((cos_cache, sin_cache), dim=0)
        hidden_states, deepstack_feature_lists = self.blocks(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            window_mask=window_mask,
            cu_seqlens=cu_seqlens,
            cu_window_seqlens=cu_window_seqlens
        )

        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states,
                mpu.get_context_parallel_group(),
                0,
                split_gather_sizes,
                "up"
            )
        if get_args().sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
            if res_dim != 0:
                hidden_states = hidden_states[: all_patch_num]  # s*t

        # should not gather here when sequence parallel otherwise grad norm will be doubled
        return hidden_states, window_index, deepstack_feature_lists
