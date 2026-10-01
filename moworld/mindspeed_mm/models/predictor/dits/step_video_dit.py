# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Optional, Dict, Tuple
from contextlib import nullcontext

from einops import rearrange, repeat
import torch
import torch.nn as nn
from einops import rearrange, repeat
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps
)
from megatron.legacy.model.enums import AttnType
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings.pos_embeddings import RoPE3DStepVideo
from mindspeed_mm.models.common.attention import ParallelAttention
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward


class StepVideoDiT(MultiModalModule):
    def __init__(
        self,
        num_heads: int = 48,
        head_dim: int = 128,
        channel_split: list = None,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_norm_type: str = "rmsnorm",
        attention_norm_elementwise_affine: bool = False,
        attention_norm_eps: float = 1e-6,
        fa_layout: str = "bsnd",
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[list] = None,
        pre_process: bool = True,
        post_process: bool = True,
        **kwargs
    ):
        super().__init__(config=None)

        # Set some common variables used across the board.
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.caption_channels = caption_channels
        self.use_additional_conditions = use_additional_conditions
        self.patch_size = patch_size
        self.sequence_parallel = args.sequence_parallel
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = args.distribute_saved_activations
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        if self.pre_process:
            self.pos_embed = PatchEmbed(
                patch_size=patch_size,
                in_channels=self.in_channels if not use_additional_conditions else in_channels * 2,
                embed_dim=self.inner_dim
            )

            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

            if isinstance(self.caption_channels, int):
                caption_channel = self.caption_channels
            else:
                caption_channel, clip_channel = self.caption_channels
                self.clip_projection = nn.Linear(clip_channel, self.inner_dim)

            self.caption_norm = nn.LayerNorm(caption_channel, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channel, hidden_size=self.inner_dim
            )

        # Rotary positional embeddings
        self.rope = RoPE3DStepVideo(
            ch_split=channel_split
        )

        self.global_layer_idx = tuple(range(num_layers))
        self.transformer_blocks = nn.ModuleList(
            [
                StepVideoTransformerBlock(
                    dim=self.inner_dim,
                    attention_head_dim=self.attention_head_dim,
                    attention_norm_type=attention_norm_type,
                    attention_norm_elementwise_affine=attention_norm_elementwise_affine,
                    attention_norm_eps=attention_norm_eps,
                    fa_layout=fa_layout,
                    rope=self.rope,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 3. Output blocks.
        if self.post_process:
            self.norm_out = nn.LayerNorm(self.inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim ** 0.5)
            self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels)

        self.use_dpo = getattr(args.mm.model, "dpo", None)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def patchfy(self, hidden_states, condition_hidden_states):
        if condition_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, condition_hidden_states], dim=2)
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.ones([len(kv_seqlens), q_seqlen, max(kv_seqlens)], dtype=torch.bool,
                          device=encoder_attention_mask.device)  # b s_q s_kv
        encoder_hidden_states = encoder_hidden_states.squeeze(1)  # b 1 s h => b s h
        encoder_hidden_states = encoder_hidden_states[:, : max(kv_seqlens)]  # b s h
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 0

        return encoder_hidden_states, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        prompt: Optional[list] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor = None,
        **kwargs
    ):
        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.pre_process:
            if hidden_states.ndim != 5:
                raise ValueError("hidden_states's shape should be (bsz, f, ch, h ,w)")

            encoder_hidden_states = prompt[0]  # b 1 s h
            encoder_hidden_states_2 = prompt[1]  # b 1 s h
            motion_score = kwargs.get("motion_score", 5.0)
            condition_hidden_states = kwargs.get("image_latents")

            # Only retain stepllm's mask
            if isinstance(prompt_mask, list):
                encoder_attention_mask = prompt_mask[0]
            # Padding 1 on the mask of the stepllm
            len_clip = encoder_hidden_states_2.shape[2]
            encoder_attention_mask = encoder_attention_mask.squeeze(1).to(
                hidden_states.device)  # stepchat_tokenizer_mask: b 1 s => b s
            encoder_attention_mask = torch.nn.functional.pad(encoder_attention_mask, (len_clip, 0),
                                                            value=1)  # pad attention_mask with clip's length

            bsz, frame, _, height, width = hidden_states.shape
            if mpu.get_context_parallel_world_size() > 1:
                frame //= mpu.get_context_parallel_world_size()
                hidden_states = split_forward_gather_backward(hidden_states, mpu.get_context_parallel_group(), dim=1,
                                                            grad_scale='down')

            height, width = height // self.patch_size, width // self.patch_size
            hidden_states = self.patchfy(hidden_states, condition_hidden_states)
            len_frame = hidden_states.shape[1]

            if self.use_additional_conditions:
                if condition_hidden_states is not None:
                    added_cond_kwargs = {
                        "motion_score": torch.tensor([motion_score], device=hidden_states.device,
                                                    dtype=hidden_states.dtype).repeat(bsz)
                    }
                else:
                    added_cond_kwargs = {
                        "resolution": torch.tensor([(height, width)] * bsz, device=hidden_states.device,
                                                dtype=hidden_states.dtype),
                        "nframe": torch.tensor([frame] * bsz, device=hidden_states.device, dtype=hidden_states.dtype),
                        "fps": fps
                    }
            else:
                added_cond_kwargs = {}

            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs=added_cond_kwargs
            )

            encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))
            if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
                clip_embedding = self.clip_projection(encoder_hidden_states_2)
                encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=2)

            hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()

            encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask, encoder_hidden_states,
                                                                    q_seqlen=frame * len_frame)

            # Rotary positional embeddings
            rotary_pos_emb = self.rope(bsz, frame * mpu.get_context_parallel_world_size(), height, width, hidden_states.device)# s b 1 d
            if mpu.get_context_parallel_world_size() > 1:
                rotary_pos_emb = rotary_pos_emb.chunk(mpu.get_context_parallel_world_size(), dim=0)[mpu.get_context_parallel_rank()]
        else:
            encoder_hidden_states = prompt
            attn_mask = prompt_mask.to(torch.bool)
            embedded_timestep = kwargs["embedded_timestep"]
            rotary_pos_emb = kwargs["rotary_pos_emb"]
            bsz, frame, height, width, len_frame = kwargs["batch_size"], kwargs["frames"], kwargs["h"], kwargs["w"], kwargs["len_frame"]

        with rng_context:
            if self.recompute_granularity == "full":
                hidden_states = self._checkpointed_forward(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    attn_mask,
                    rotary_pos_emb
                )
            else:
                for _, block in zip(self.global_layer_idx, self.transformer_blocks):
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        timestep,
                        attn_mask,
                        rotary_pos_emb
                    )

        output = hidden_states

        if self.post_process:
            hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)
            embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()

            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )

            hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

            output = rearrange(output, '(b f) c h w -> b f c h w', f=frame)

            if mpu.get_context_parallel_world_size() > 1:
                output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=1,
                                                            grad_scale='up')

        rtn = (output, encoder_hidden_states, timestep, embedded_timestep, attn_mask.to(torch.bfloat16), rotary_pos_emb)
        return rtn

    def pipeline_set_prev_stage_tensor(self, input_tensor_list, extra_kwargs):
        """
        Implemnented for pipeline parallelism. The input tensor is got from last PP stage.
        Args:
            input_tensor_list: same as the return value of pipeline_set_next_stage_tensor
            extra_kwargs: kwargs for forward func.

        Returns:
            predictor_input_list: values for predictor forward.
            training_loss_input_list: values to calculate loss.
        """
        score, score_lose = None, None
        if self.use_dpo is not None:
            (prev_output, prompt, predictor_timesteps, embedded_timestep, attn_mask, rotary_pos_emb, score, score_lose,
            latents, noised_latents, timesteps, noise) = input_tensor_list
        else:
            (prev_output, prompt, predictor_timesteps, embedded_timestep, attn_mask, rotary_pos_emb,
            latents, noised_latents, timesteps, noise) = input_tensor_list
        predictor_input_list = [prev_output, predictor_timesteps, prompt, None, attn_mask]
        training_loss_input_list = [latents, noised_latents, timesteps, noise, None]
        extra_kwargs["embedded_timestep"] = embedded_timestep
        extra_kwargs["rotary_pos_emb"] = rotary_pos_emb
        batch_size, frames, _, height, width = latents.shape
        len_frame = ((height - self.patch_size) // self.patch_size + 1) * ((width - self.patch_size) // self.patch_size + 1)
        (extra_kwargs["batch_size"], extra_kwargs["frames"], extra_kwargs["h"], extra_kwargs["w"], extra_kwargs["len_frame"]) = batch_size, frames, height, width, len_frame

        if self.use_dpo is not None:
            score_list = [score, score_lose]
            return predictor_input_list, training_loss_input_list, score_list
        return predictor_input_list, training_loss_input_list

    def pipeline_set_next_stage_tensor(self, input_list, output_list, extra_kwargs=None):
        """return as
        [prev_output, prompt, predictor_timesteps, embedded_timestep, prompt_mask, rotary_pos_emb
         latents, noised_latents, timesteps, noise]
         which should be corresponded with initialize_pipeline_tensor_shapes
        """
        latents, noised_latents, timesteps, noise, _ = input_list
        return list(output_list) + [latents, noised_latents, timesteps, noise]

    @staticmethod
    def initialize_pipeline_tensor_shapes():
        args = get_args()
        micro_batch_size = args.micro_batch_size
        dtype = args.params_dtype

        model_cfg = args.mm.model
        data_cfg = args.mm.data.dataset_param.preprocess_parameters
        num_attention_heads = model_cfg.predictor.num_heads
        attention_head_dim = model_cfg.predictor.head_dim
        hidden_size = num_attention_heads * attention_head_dim
        height = getattr(data_cfg, "max_height", 544)
        width = getattr(data_cfg, "max_width", 992)
        frames = data_cfg.num_frames
        latent_size = (frames // model_cfg.ae.frame_len * 3, height // 2 ** 4, width // 2 ** 4)
        seq_len = latent_size[0] * latent_size[1] * latent_size[2]
        tokenizer_configs = args.mm.data.dataset_param.tokenizer_config
        max_prompt_len = sum([tokenizer_config.get("model_max_length", 0) for tokenizer_config in tokenizer_configs])
        channels = model_cfg.predictor.in_channels
        rtn_size = 1
        use_dpo = getattr(model_cfg, "dpo", None)
        if use_dpo is not None:
            micro_batch_size = micro_batch_size * 2
            rtn_size = 2
        pipeline_tensor_shapes = [
            {"shape": (micro_batch_size * rtn_size, seq_len, hidden_size), "dtype": dtype},  # prev_output
            {"shape": (micro_batch_size * rtn_size, max_prompt_len, hidden_size), "dtype": dtype},  # prompt
            {"shape": (micro_batch_size * rtn_size, 6 * hidden_size), "dtype": dtype},  # predictor_timesteps
            {"shape": (micro_batch_size * rtn_size, hidden_size), "dtype": dtype},  # embedded_timestep
            {"shape": (micro_batch_size * rtn_size, seq_len, max_prompt_len), "dtype": dtype}, # origin_prompt_mask
            {"shape": (seq_len * rtn_size, micro_batch_size, num_attention_heads, attention_head_dim), "dtype": torch.float32},  # rotary_pos_emb
            {"shape": (micro_batch_size, latent_size[0], channels, latent_size[1], latent_size[2]), "dtype": dtype},  # latents(x0)
            {"shape": (micro_batch_size, latent_size[0], channels, latent_size[1], latent_size[2]), "dtype": dtype},  # noised_latents
            {"shape": (micro_batch_size,), "dtype": torch.float32},  # timesteps
            {"shape": (micro_batch_size, latent_size[0], channels, latent_size[1], latent_size[2]), "dtype": dtype},  # noise
        ]
        if use_dpo is not None:
            score_shape = [
                {'shape': (1,), 'dtype': torch.float64},  # score_win
                {'shape': (1,), 'dtype': torch.float64},  # score_lose
            ]
            pipeline_tensor_shapes = pipeline_tensor_shapes[:6] + score_shape + pipeline_tensor_shapes[6:]
        return pipeline_tensor_shapes

    def _get_block(self, layer_number):
        return self.transformer_blocks[layer_number]

    def _checkpointed_forward(
            self,
            latents,
            prompt,
            timestep,
            prompt_mask,
            rotary_pos_emb):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_block(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_num = 0
            while layer_num < self.num_layers:
                latents = tensor_parallel.checkpoint(
                    custom(layer_num, layer_num + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    latents,
                    prompt,
                    timestep,
                    prompt_mask,
                    rotary_pos_emb
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            for layer_num in range(self.num_layers):
                if layer_num < self.recompute_num_layers:
                    latents = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        latents,
                        prompt,
                        timestep,
                        prompt_mask,
                        rotary_pos_emb
                    )
                else:
                    block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        prompt,
                        timestep,
                        prompt_mask,
                        rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        return latents


class StepVideoTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            attention_head_dim: int,
            norm_eps: float = 1e-5,
            attention_norm_type: str = "rmsnorm",
            attention_norm_elementwise_affine: bool = False,
            attention_norm_eps: float = 1e-6,
            fa_layout: str = "bsnd",
            rope=None
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = ParallelAttention(
            query_dim=dim,
            key_dim=dim,
            num_attention_heads=dim // attention_head_dim,
            hidden_size=dim,
            use_qk_norm=True,
            norm_type=attention_norm_type,
            norm_elementwise_affine=attention_norm_elementwise_affine,
            norm_eps=attention_norm_eps,
            is_qkv_concat=True,
            attention_type=AttnType.self_attn,
            rope=rope,
            fa_layout=fa_layout
        )

        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = ParallelAttention(
            query_dim=dim,
            key_dim=dim,
            num_attention_heads=dim // attention_head_dim,
            hidden_size=dim,
            use_qk_norm=True,
            norm_type=attention_norm_type,
            norm_elementwise_affine=attention_norm_elementwise_affine,
            norm_eps=attention_norm_eps,
            is_kv_concat=True,
            attention_type=AttnType.cross_attn,
            fa_layout=fa_layout,
            split_kv_in_forward=False
        )

        self.ff = FeedForward(
            dim=dim,
            activation_fn="gelu-approximate",
            bias=False
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)

    def forward(
            self,
            q: torch.Tensor,
            kv: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            attn_mask=None,
            rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale_shift_table_expanded = self.scale_shift_table[None] + timestep.reshape(-1, 6, self.dim)
        chunks = scale_shift_table_expanded.chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (torch.clone(chunk) for chunk in chunks)

        scale_shift_q = self.norm1(q) * (1 + scale_msa) + shift_msa

        # self attention
        attn_q = self.attn1(
            scale_shift_q,
            input_layout="bsh",
            rotary_pos_emb=rotary_pos_emb
        )
        q = attn_q * gate_msa + q

        # cross attention
        attn_q = self.attn2(
            q,
            kv,
            attn_mask,
            input_layout="bsh"
        )
        q = attn_q + q
        scale_shift_q = self.norm2(q) * (1 + scale_mlp) + shift_mlp

        # feed forward
        ff_output = self.ff(scale_shift_q)
        q = ff_output * gate_mlp + q

        return q


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
            self,
            patch_size=64,
            in_channels=3,
            embed_dim=768,
            layer_norm=False,
            flatten=True,
            bias=True,
    ):
        super().__init__()

        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, latent):
        latent = self.proj(latent).to(latent.dtype)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)
        if self.layer_norm:
            latent = self.norm(latent)

        return latent


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.nframe_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
            self.fps_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
            self.motion_score_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, resolution=None, nframe=None, fps=None, motion_score=None):
        hidden_dtype = next(self.timestep_embedder.parameters()).dtype

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            batch_size = timestep.shape[0]
            motion_score_emb = self.additional_condition_proj(motion_score.flatten()).to(hidden_dtype)
            motion_score_emb = self.motion_score_embedder(motion_score_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + motion_score_emb

            if fps is not None:
                fps_emb = self.additional_condition_proj(fps.flatten()).to(hidden_dtype)
                fps_emb = self.fps_embedder(fps_emb).reshape(batch_size, -1)
                conditioning = conditioning + fps_emb
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Module):
    r"""
        Norm layer adaptive layer norm single (adaLN-single).

        As proposed in PixArt-Alpha.

        Parameters:
            embedding_dim (`int`): The size of each embedding vector.
            use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, time_step_rescale=1000):
        super().__init__()

        args = get_args()
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 2, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        config = core_transformer_config_from_args(args)
        self.linear = tensor_parallel.ColumnParallelLinear(
            embedding_dim,
            6 * embedding_dim,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=True
        )

        self.time_step_rescale = time_step_rescale  # timestep usually in [0, 1], we rescale it to [0,1000] for stability

    def forward(
            self,
            timestep: torch.Tensor,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep * self.time_step_rescale, **added_cond_kwargs)

        out = self.linear(self.silu(embedded_timestep))[0]

        return out, embedded_timestep
