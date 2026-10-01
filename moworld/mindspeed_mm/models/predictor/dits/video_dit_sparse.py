from typing import Optional, Tuple
from contextlib import nullcontext

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.normalization import AdaLayerNormSingle
from megatron.core import mpu, tensor_parallel
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank)
from megatron.training import get_args
from megatron.legacy.model.enums import AttnType

from mindspeed_mm.data.data_utils.constants import INPUT_MASK, MASKED_VIDEO
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings import PatchEmbed2D
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.attention import MultiHeadSparseAttentionSBH
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward
from mindspeed_mm.models.common.embeddings.pos_embeddings import RoPE3DSORA


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class VideoDitSparse(MultiModalModule):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in· the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_q_bias: Whether to use bias for Query in VideoDiTBlock's attention.
        attention_k_bias: Whether to use bias for Key in VideoDiTBlock's attention.
        attention_v_bias: Whether to use bias for Value in VideoDiTBlock's attention.
        fa_layout: The inputs's layout in Flash Attention.
        patch_size: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_elementwise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of the normalization.
        interpolation_scale: The scale for interpolation.
    """

    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_q_bias: bool = False,
        attention_k_bias: bool = False,
        attention_v_bias: bool = False,
        fa_layout: str = "sbh",
        patch_size_thw: Tuple[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        interpolation_scale: Tuple[float] = None,
        sparse1d: bool = False,
        sparse_n: int = 2,
        pre_process: bool = True,
        post_process: bool = True,
        global_layer_idx: Optional[Tuple] = None,
        **kwargs
    ):
        super().__init__(config=None)
        args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel = args.sequence_parallel
        self.gradient_checkpointing = True
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.checkpoint_skip_core_attention = args.recompute_skip_core_attention
        self.recompute_num_layers = args.recompute_num_layers
        self.recompute_num_layers_skip_core_attention = args.recompute_num_layers_skip_core_attention
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        inner_dim = num_heads * head_dim
        self.num_layers = num_layers
        self.patch_size_t = patch_size_thw[0]
        self.patch_size = patch_size_thw[1]

        if self.pre_process:
            self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
            self.pos_embed = PatchEmbed2D(
                patch_size=self.patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )
            self.adaln_single = AdaLayerNormSingle(inner_dim)
            # set label "sequence_parallel", for all_reduce the grad
            for param in self.adaln_single.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)

        # Rotary positional embeddings
        self.rope = RoPE3DSORA(
            head_dim=head_dim,
            interpolation_scale=interpolation_scale
        )

        self.global_layer_idx = global_layer_idx if global_layer_idx is not None else tuple(range(num_layers))
        self.videodit_sparse_blocks = nn.ModuleList(
            [
                VideoDiTSparseBlock(
                    inner_dim,
                    num_heads,
                    head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_q_bias=attention_q_bias,
                    attention_k_bias=attention_k_bias,
                    attention_v_bias=attention_v_bias,
                    fa_layout=fa_layout,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    sparse1d=sparse1d if 1 < _ < 30 else False,
                    sparse_n=sparse_n,
                    sparse_group=_ % 2 == 1,
                    rope=self.rope
                )
                for _ in self.global_layer_idx
            ]
        )

        if self.post_process:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
            self.proj_out = nn.Linear(inner_dim, self.patch_size_t * self.patch_size * self.patch_size * out_channels)
            setattr(self.scale_shift_table, "sequence_parallel", self.sequence_parallel)

    def prepare_sparse_mask(self, video_mask, prompt_mask, sparse_n):
        video_mask = video_mask.unsqueeze(1)
        prompt_mask = prompt_mask.unsqueeze(1)
        _len = video_mask.shape[-1]
        if _len % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - _len % (sparse_n * sparse_n)

        video_mask_sparse = F.pad(video_mask, (0, pad_len, 0, 0), value=-9980.0)
        video_mask_sparse_1d = rearrange(
            video_mask_sparse,
            'b 1 1 (g k) -> (k b) 1 1 g',
            k=sparse_n
        )
        video_mask_sparse_1d_group = rearrange(
            video_mask_sparse,
            'b 1 1 (n m k) -> (m b) 1 1 (n k)',
            m=sparse_n,
            k=sparse_n
        )
        prompt_mask_sparse = prompt_mask.repeat(sparse_n, 1, 1, 1)

        def get_attention_mask(mask, repeat_num):
            mask = mask.to(torch.bool)
            mask = mask.repeat(1, 1, repeat_num, 1)
            return mask

        video_mask_sparse_1d = get_attention_mask(video_mask_sparse_1d, video_mask_sparse_1d.shape[-1])
        video_mask_sparse_1d_group = get_attention_mask(
            video_mask_sparse_1d_group, video_mask_sparse_1d_group.shape[-1]
        )
        prompt_mask_sparse_1d = get_attention_mask(
            prompt_mask_sparse, video_mask_sparse_1d.shape[-1]
        )
        prompt_mask_sparse_1d_group = prompt_mask_sparse_1d

        if not video_mask_sparse_1d.any():
            video_mask_sparse_1d = None
        if not video_mask_sparse_1d_group.any():
            video_mask_sparse_1d_group = None

        args = get_args()
        if args.context_parallel_algo == 'megatron_cp_algo' or args.context_parallel_algo == 'hybrid_cp_algo':
            if args.context_parallel_algo == 'megatron_cp_algo':
                r_size = mpu.get_context_parallel_world_size()
                r_rank = mpu.get_context_parallel_rank()
            else:
                r_size = get_context_parallel_for_hybrid_ring_world_size()
                r_rank = get_context_parallel_for_hybrid_ring_rank()

            if prompt_mask_sparse_1d is not None:
                prompt_mask_sparse_1d_row = prompt_mask_sparse_1d.chunk(r_size, dim=2)[r_rank].contiguous()
                prompt_mask_sparse_1d = [m.contiguous() for m in prompt_mask_sparse_1d_row.chunk(r_size, dim=3)]
            if prompt_mask_sparse_1d_group is not None:
                prompt_mask_sparse_1d_group_row = prompt_mask_sparse_1d_group.chunk(r_size, dim=2)[r_rank].contiguous()
                prompt_mask_sparse_1d_group = [m.contiguous() for m in prompt_mask_sparse_1d_group_row.chunk(r_size, dim=3)]
            if video_mask_sparse_1d is not None:
                video_mask_sparse_1d_row = video_mask_sparse_1d.chunk(r_size, dim=2)[r_rank].contiguous()
                video_mask_sparse_1d = [m.contiguous() for m in video_mask_sparse_1d_row.chunk(r_size, dim=3)]
            if video_mask_sparse_1d_group is not None:
                video_mask_sparse_1d_group_row = video_mask_sparse_1d_group.chunk(r_size, dim=2)[r_rank].contiguous()
                video_mask_sparse_1d_group = [m.contiguous() for m in video_mask_sparse_1d_group_row.chunk(r_size, dim=3)]

        return {
            False: (video_mask_sparse_1d, prompt_mask_sparse_1d),
            True: (video_mask_sparse_1d_group, prompt_mask_sparse_1d_group)
        }

    def forward(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # RNG context.
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.pre_process:
            # pre_process latents
            batch_size, c, frames, h, w = latents.shape
            if mpu.get_context_parallel_world_size() > 1:
                frames //= mpu.get_context_parallel_world_size()
                latents = split_forward_gather_backward(latents, mpu.get_context_parallel_group(), dim=2,
                                                        grad_scale='down')
                prompt = split_forward_gather_backward(prompt, mpu.get_context_parallel_group(),
                                                       dim=2, grad_scale='down')

            latents, prompt, timestep, embedded_timestep = self._operate_on_patched_inputs(
                latents, prompt, timestep, batch_size, **kwargs
            )

            latents = rearrange(latents, 'b s h -> s b h', b=batch_size).contiguous()
            prompt = rearrange(prompt, 'b s h -> s b h', b=batch_size).contiguous()
            timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

            if self.sequence_parallel:
                latents = tensor_parallel.scatter_to_sequence_parallel_region(latents)
                prompt = tensor_parallel.scatter_to_sequence_parallel_region(prompt)

            prompt_mask = prompt_mask.view(batch_size, -1, prompt_mask.shape[-1])
            # convert encoder_attention_mask to a bias the same way we do for attention_mask
            if prompt_mask is not None and prompt_mask.ndim == 3:
                # b, 1, l
                prompt_mask = (1 - prompt_mask.to(self.dtype)) * -10000.0
        else:
            embedded_timestep = kwargs['embedded_timestep']
            batch_size, c, frames, h, w = kwargs['batch_size'], kwargs['c'], kwargs['frames'], kwargs['h'], kwargs['w']
            if mpu.get_context_parallel_world_size() > 1:
                frames //= mpu.get_context_parallel_world_size()

        # 1. mask converting
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = h // self.patch_size, w // self.patch_size
        frames, height, width = torch.tensor(frames), torch.tensor(height), torch.tensor(width)

        # Rotary positional embeddings
        rotary_pos_emb = self.rope(batch_size, frames * mpu.get_context_parallel_world_size(), height, width, latents.device)  # s b 1 d
        if mpu.get_context_parallel_world_size() > 1:
            rotary_pos_emb = rotary_pos_emb.chunk(mpu.get_context_parallel_world_size(), dim=0)[mpu.get_context_parallel_rank()]

        origin_video_mask = video_mask.clone().detach().to(self.dtype)
        origin_prompt_mask = prompt_mask.clone().detach().to(self.dtype)
        if video_mask is not None and video_mask.ndim == 4:
            video_mask = video_mask.to(self.dtype)

            video_mask = video_mask.unsqueeze(1)  # b 1 t h w
            video_mask = F.max_pool3d(
                video_mask,
                kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
                stride=(self.patch_size_t, self.patch_size, self.patch_size)
            )
            video_mask = rearrange(video_mask, 'b 1 t h w -> (b 1) 1 (t h w)')
            video_mask = (1 - video_mask.bool().to(self.dtype)) * -10000.0

        sparse_mask = {}
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = self.prepare_sparse_mask(video_mask, prompt_mask, sparse_n)

        if (video_mask == 0).all():
            video_mask = None

        with rng_context:
            if self.recompute_granularity == "full":
                latents = self._checkpointed_forward(
                    sparse_mask,
                    latents,
                    video_mask=video_mask,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    timestep=timestep,
                    frames=frames,
                    height=height,
                    width=width,
                    rotary_pos_emb=rotary_pos_emb,
                )
            else:
                for i, block in zip(self.global_layer_idx, self.videodit_sparse_blocks):
                    if i > 1 and i < 30:
                        try:
                            video_mask, prompt_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
                        except KeyError:
                            video_mask, prompt_mask = None, None
                    else:
                        try:
                            video_mask, prompt_mask = sparse_mask[1][block.self_atten.sparse_group]
                        except KeyError:
                            video_mask, prompt_mask = None, None
                    latents = block(
                                latents,
                                video_mask=video_mask,
                                prompt=prompt,
                                prompt_mask=prompt_mask,
                                timestep=timestep,
                                frames=frames,
                                height=height,
                                width=width,
                                rotary_pos_emb=rotary_pos_emb,
                            )

        output = latents

        if self.post_process:
            # 3. Output
            output = self._get_output_for_patched_inputs(
                latents=latents,
                timestep=timestep,
                embedded_timestep=embedded_timestep,
                num_frames=frames,
                height=height,
                width=width,
            )  # b c t h w

            if mpu.get_context_parallel_world_size() > 1:
                output = gather_forward_split_backward(output, mpu.get_context_parallel_group(), dim=2,
                                                            grad_scale='up')
        rtn = (output, prompt, timestep, embedded_timestep, origin_video_mask, origin_prompt_mask)
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
        (prev_output, prompt, predictor_timesteps, embedded_timestep, video_mask, prompt_mask,
         latents, noised_latents, timesteps, noise) = input_tensor_list
        predictor_input_list = [prev_output, predictor_timesteps, prompt, video_mask, prompt_mask]
        training_loss_input_list = [latents, noised_latents, timesteps, noise, video_mask]

        extra_kwargs['embedded_timestep'] = embedded_timestep
        (extra_kwargs['batch_size'], extra_kwargs['c'], extra_kwargs['frames'], extra_kwargs['h'],
         extra_kwargs['w']) = latents.shape

        return predictor_input_list, training_loss_input_list

    def pipeline_set_next_stage_tensor(self, input_list, output_list, extra_kwargs=None):
        """return as
        [prev_output, prompt, predictor_timesteps, embedded_timestep, video_mask, prompt_mask,
         latents, noised_latents, timesteps, noise]
         which should be corresponded with initialize_pipeline_tensor_shapes
        """
        latents, noised_latents, timesteps, noise, _ = input_list
        if timesteps.dtype != torch.bfloat16:
            timesteps = timesteps.to(torch.bfloat16)
        return list(output_list) + [latents, noised_latents, timesteps, noise]

    @staticmethod
    def initialize_pipeline_tensor_shapes():
        args = get_args()
        micro_batch_size = args.micro_batch_size
        dtype = args.params_dtype

        model_cfg = args.mm.model
        data_cfg = args.mm.data.dataset_param.preprocess_parameters
        hidden_size = model_cfg.predictor.num_heads * model_cfg.predictor.head_dim
        height = data_cfg.max_height if hasattr(data_cfg, "max_height") else 352
        width = data_cfg.max_width if hasattr(data_cfg, "max_width") else 640
        latent_size = ((data_cfg.num_frames + 3) // 4, height // 8, width // 8)
        divisor = model_cfg.predictor.patch_size_thw[0] * (model_cfg.predictor.patch_size_thw[1] ** 2)
        seq_len = latent_size[0] * latent_size[1] * latent_size[2] // divisor
        max_prompt_len = model_cfg.model_max_length if hasattr(model_cfg, "model_max_length") else 512
        channels = model_cfg.predictor.in_channels

        pipeline_tensor_shapes = [
            {'shape': (seq_len, micro_batch_size, hidden_size), 'dtype': dtype},  # prev_output
            {'shape': (max_prompt_len, micro_batch_size, hidden_size), 'dtype': dtype},  # prompt
            {'shape': (6, micro_batch_size, hidden_size), 'dtype': dtype},  # predictor_timesteps
            {'shape': (micro_batch_size, hidden_size), 'dtype': dtype},  # embedded_timestep
            {'shape': (micro_batch_size, *latent_size), 'dtype': dtype},  # origin_video_mask
            {'shape': (micro_batch_size, 1, max_prompt_len), 'dtype': dtype},  # origin_prompt_mask
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # latents(x0)
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # noised_latents
            {'shape': (micro_batch_size,), 'dtype': dtype},  # timesteps
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # noise
        ]
        return pipeline_tensor_shapes

    def _get_block(self, layer_number):
        return self.videodit_sparse_blocks[layer_number]

    def _checkpointed_forward(
            self,
            sparse_mask,
            latents,
            video_mask,
            prompt,
            prompt_mask,
            timestep,
            frames,
            height,
            width,
            rotary_pos_emb):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_block(index)
                    layer_idx = self.global_layer_idx[index]
                    if layer_idx > 1 and layer_idx < 30:
                        args[0], args[2] = sparse_mask[layer.self_atten.sparse_n][layer.self_atten.sparse_group]
                    else:
                        args[0], args[2] = sparse_mask[1][layer.self_atten.sparse_group]
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
                    video_mask,
                    prompt,
                    prompt_mask,
                    timestep,
                    frames,
                    height,
                    width,
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
                        video_mask,
                        prompt,
                        prompt_mask,
                        timestep,
                        frames,
                        height,
                        width,
                        rotary_pos_emb
                    )
                elif layer_num < self.recompute_num_layers + self.recompute_num_layers_skip_core_attention:
                    block = self._get_block(layer_num)
                    layer_idx = self.global_layer_idx[layer_num]
                    if layer_idx > 1 and layer_idx < 30:
                        video_mask, prompt_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
                    else:
                        video_mask, prompt_mask = sparse_mask[1][block.self_atten.sparse_group]
                    latents = block(
                        latents,
                        video_mask,
                        prompt,
                        prompt_mask,
                        timestep,
                        frames,
                        height,
                        width,
                        rotary_pos_emb,
                        checkpoint_skip_core_attention=self.checkpoint_skip_core_attention
                    )
                else:
                    block = self._get_block(layer_num)
                    layer_idx = self.global_layer_idx[layer_num]
                    if layer_idx > 1 and layer_idx < 30:
                        video_mask, prompt_mask = sparse_mask[block.self_atten.sparse_n][block.self_atten.sparse_group]
                    else:
                        video_mask, prompt_mask = sparse_mask[1][block.self_atten.sparse_group]
                    latents = block(
                        latents,
                        video_mask,
                        prompt,
                        prompt_mask,
                        timestep,
                        frames,
                        height,
                        width,
                        rotary_pos_emb
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        return latents

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def _operate_on_patched_inputs(self, latents, prompt, timestep, batch_size, **kwargs):

        latents = self.pos_embed(latents.to(self.dtype))

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        prompt = self.caption_projection(prompt)  # b, 1, l, d or b, 1, l, d
        if prompt.shape[1] != 1:
            raise ValueError("prompt's shape mismatched")
        prompt = rearrange(prompt, 'b 1 l d -> (b 1) l d')

        return latents, prompt, timestep, embedded_timestep

    def _get_output_for_patched_inputs(
            self, latents, timestep, embedded_timestep, num_frames, height, width
    ):
        batch_size = latents.shape[1]
        shift, scale = (self.scale_shift_table[:, None] + embedded_timestep[None]).chunk(2, dim=0)
        latents = self.norm_out(latents)
        # Modulation
        latents = latents * (1 + scale) + shift
        # From (t//sp*h*w, b, h) to (t*h*w, b, h)
        if self.sequence_parallel:
            latents = tensor_parallel.gather_from_sequence_parallel_region(latents,
                                                                           tensor_parallel_output_grad=False)

        latents = rearrange(latents, 's b h -> b s h', b=batch_size).contiguous()
        latents = self.proj_out(latents)
        latents = latents.squeeze(1)

        # unpatchify
        latents = latents.reshape(
            shape=(
            -1, num_frames, height, width, self.patch_size_t, self.patch_size, self.patch_size,
            self.out_channels)
        )
        latents = torch.einsum("nthwopqc->nctohpwq", latents)
        output = latents.reshape(
            shape=(-1, self.out_channels,
                   num_frames * self.patch_size_t, height * self.patch_size,
                   width * self.patch_size)
        )
        return output


class VideoDiTSparseBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            attention_q_bias: bool = False,
            attention_k_bias: bool = False,
            attention_v_bias: bool = False,
            attention_out_bias: bool = True,
            fa_layout: str = "sbh",
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            sparse1d: bool = False,
            sparse_n: int = 2,
            sparse_group: bool = False,
            rope=None,
    ):
        super().__init__()

        args = get_args()
        self.sequence_parallel = args.sequence_parallel

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.self_atten = MultiHeadSparseAttentionSBH(
            query_dim=dim,
            key_dim=cross_attention_dim if only_cross_attention else None,
            num_attention_heads=num_heads,
            hidden_size=head_dim * num_heads,
            proj_q_bias=attention_q_bias,
            proj_k_bias=attention_k_bias,
            proj_v_bias=attention_v_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            attention_type=AttnType.self_attn,
            fa_layout=fa_layout,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            rope=rope
        )

        # 2. Cross-Attn
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.cross_atten = MultiHeadSparseAttentionSBH(
            query_dim=dim,
            key_dim=cross_attention_dim if not double_self_attention else None,
            num_attention_heads=num_heads,
            hidden_size=head_dim * num_heads,
            proj_q_bias=attention_q_bias,
            proj_k_bias=attention_k_bias,
            proj_v_bias=attention_v_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            attention_type=AttnType.cross_attn,
            fa_layout=fa_layout,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
        )  # is self-attn if encoder_hidden_states is none

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Scale-shift.
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim ** 0.5)
        # set label "sequence_parallel", for all_reduce the grad
        setattr(self.scale_shift_table, "sequence_parallel", self.sequence_parallel)

    def _function_before_self_core_attention(
            self,
            latents,
            timestep,
            frames=None,
            height=None,
            width=None,
            rotary_pos_emb: Optional[torch.FloatTensor] = None
    ):
        batch_size = latents.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
        ).chunk(6, dim=0)
        norm_latents = self.norm1(latents)
        norm_latents = norm_latents * (1 + scale_msa) + shift_msa

        query, key, value = self.self_atten.function_before_core_attention(
            norm_latents,
            frames=frames,
            height=height,
            width=width,
            rotary_pos_emb=rotary_pos_emb
        )

        return query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _function_after_self_core_attention(
            self,
            latents,
            self_core_attn_output,
            prompt: Optional[torch.FloatTensor] = None,
            prompt_mask: Optional[torch.FloatTensor] = None,
            frames=None,
            height=None,
            width=None,
            gate_msa=None,
            shift_mlp=None,
            scale_mlp=None,
            gate_mlp=None,
    ):
        self_attn_output = self.self_atten.function_after_core_attention(
            self_core_attn_output,
            frames=frames,
            height=height,
            width=width,
            dtype=next(self.parameters()).dtype
        )

        attn_output = gate_msa * self_attn_output
        latents = attn_output + latents

        # Cross-Attention
        norm_latents = latents
        attn_output = self.cross_atten(
            query=norm_latents,
            key=prompt,
            mask=prompt_mask,
            frames=frames,
            height=height,
            width=width,
        )
        latents = attn_output + latents

        # Feed-forward
        norm_latents = self.norm2(latents)
        norm_latents = norm_latents * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_latents)
        ff_output = gate_mlp * ff_output
        latents = ff_output + latents

        return latents

    def forward(
            self,
            latents: torch.FloatTensor,
            video_mask: Optional[torch.FloatTensor] = None,
            prompt: Optional[torch.FloatTensor] = None,
            prompt_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            frames: int = None,
            height: int = None,
            width: int = None,
            rotary_pos_emb=None,
            checkpoint_skip_core_attention: bool = False
    ) -> torch.FloatTensor:
        # recompute if skip core attention of self_atten
        if checkpoint_skip_core_attention:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = tensor_parallel.checkpoint(
                self._function_before_self_core_attention,
                False,
                latents,
                timestep,
                frames,
                height,
                width,
                rotary_pos_emb
            )
        else:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._function_before_self_core_attention(
                latents,
                timestep,
                frames,
                height,
                width,
                rotary_pos_emb
            )

        # self core attention
        self_core_attn_output = self.self_atten.function_core_attention(query, key, value, video_mask)

        if checkpoint_skip_core_attention:
            latents = tensor_parallel.checkpoint(
                self._function_after_self_core_attention,
                False,
                latents,
                self_core_attn_output,
                prompt,
                prompt_mask,
                frames,
                height,
                width,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )
        else:
            latents = self._function_after_self_core_attention(
                latents,
                self_core_attn_output,
                prompt,
                prompt_mask,
                frames,
                height,
                width,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )

        return latents


class VideoDitSparseI2V(VideoDitSparse):
    def __init__(
            self,
            num_heads: int = 16,
            head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            cross_attention_dim: Optional[int] = None,
            attention_q_bias: bool = False,
            attention_k_bias: bool = False,
            attention_v_bias: bool = False,
            fa_layout: str = "sbh",
            patch_size_thw: Tuple[int] = None,
            activation_fn: str = "geglu",
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            caption_channels: int = None,
            interpolation_scale: Tuple[float] = None,
            sparse1d: bool = False,
            sparse_n: int = 2,
            vae_scale_factor_t: int = 4,
            pre_process: bool = True,
            post_process: bool = True,
            global_layer_idx: Optional[Tuple] = None,
            **kwargs
    ):
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            attention_q_bias=attention_q_bias,
            attention_k_bias=attention_k_bias,
            attention_v_bias=attention_v_bias,
            fa_layout=fa_layout,
            patch_size_thw=patch_size_thw,
            activation_fn=activation_fn,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            interpolation_scale=interpolation_scale,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            pre_process=pre_process,
            post_process=post_process,
            global_layer_idx=global_layer_idx,
        )
        self.vae_scale_factor_t = vae_scale_factor_t
        inner_dim = num_heads * head_dim

        if self.pre_process:
            self.pos_embed_masked_hidden_states = nn.ModuleList(
                [
                    PatchEmbed2D(
                        patch_size=patch_size_thw[1],
                        in_channels=in_channels,
                        embed_dim=inner_dim,
                    ),
                    zero_module(nn.Linear(inner_dim, inner_dim, bias=False)),
                ]
            )

            self.pos_embed_mask = nn.ModuleList(
                [
                    PatchEmbed2D(
                        patch_size=patch_size_thw[1],
                        in_channels=self.vae_scale_factor_t,
                        embed_dim=inner_dim,
                    ),
                    zero_module(nn.Linear(inner_dim, inner_dim, bias=False)),
                ]
            )

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, **kwargs):
        # inpaint

        input_hidden_states = hidden_states
        input_masked_hidden_states = kwargs.get(MASKED_VIDEO, None)
        input_mask = kwargs.get(INPUT_MASK, None)
        if mpu.get_context_parallel_world_size() > 1:
            input_masked_hidden_states = split_forward_gather_backward(input_masked_hidden_states,
                                                                       mpu.get_context_parallel_group(),
                                                                       dim=2, grad_scale='down')
            input_mask = split_forward_gather_backward(input_mask, mpu.get_context_parallel_group(),
                                                       dim=2, grad_scale='down')
        input_hidden_states = self.pos_embed(input_hidden_states.to(self.dtype))

        input_masked_hidden_states = self.pos_embed_masked_hidden_states[0](input_masked_hidden_states.to(self.dtype))
        input_masked_hidden_states = self.pos_embed_masked_hidden_states[1](input_masked_hidden_states)

        input_mask = self.pos_embed_mask[0](input_mask.to(self.dtype))
        input_mask = self.pos_embed_mask[1](input_mask)

        hidden_states = input_hidden_states + input_masked_hidden_states + input_mask

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep
