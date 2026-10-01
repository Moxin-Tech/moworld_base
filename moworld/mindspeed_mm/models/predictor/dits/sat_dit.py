from functools import reduce
from operator import mul
from typing import Optional, Tuple, Dict
from contextlib import nullcontext

import torch
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from einops import rearrange
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from mindspeed_mm.models.common.ffn import FeedForward as TensorParallelFeedForward
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward
from mindspeed_mm.models.common.embeddings.pos_embeddings import Rotary3DPositionEmbedding
from mindspeed_mm.models.common.embeddings.time_embeddings import TimeStepEmbedding, timestep_embedding
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings.patch_embeddings import VideoPatchEmbed2D, VideoPatch2D, VideoPatch3D
from mindspeed_mm.models.common.attention import ParallelAttention


class SatDiT(MultiModalModule):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        input_size: The shape of the latents (specify if the input is discrete).
        patch_size: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        use_rope: Whether to use rope in attention block.
        interpolation_scale: The scale for interpolation.
        qk_ln: Whether to use layer norm in qk.
        elementwise_affine: Whether to use learnable elementwise affine parameters for qk normalization.
    """

    def __init__(
        self,
        num_layers: int = 1,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        input_size: Tuple[int] = None,
        patch_size: Tuple[int] = None,
        patch_type: str = "2D",
        activation_fn: str = "geglu",
        norm_type: str = "layer_norm",
        num_embeds_ada_norm: Optional[int] = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        elementwise_affine: bool = True,
        text_length: int = None,
        text_hidden_size: int = None,
        time_embed_dim: int = None,
        concat_text_embed: bool = None,
        learnable_pos_embed: bool = False,
        pre_process: bool = True,
        post_process: bool = True,
        global_layer_idx: Optional[Tuple] = None,
        ofs_embed_dim: int = None,
        **kwargs
    ):
        super().__init__(config=None)
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_size = input_size
        # Validate inputs and init args.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single", "qk_ln"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )
        self.patch_size = patch_size
        self.patch_type = patch_type
        self.patch_size_t, self.patch_size_h, self.patch_size_w = patch_size
        self.ofs_embed_dim = ofs_embed_dim
        self.norm_type = norm_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.concat_text_embed = concat_text_embed
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.ori_text_length = text_length
        self.seq_len = text_length + reduce(mul, input_size) // reduce(mul, patch_size)
        self.text_hidden_size = text_hidden_size
        self.elementwise_affine = elementwise_affine
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else inner_dim

        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.checkpoint_skip_core_attention = args.recompute_skip_core_attention
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        if mpu.get_context_parallel_world_size() > 1:
            self.enable_sequence_parallelism = True
        else:
            self.enable_sequence_parallelism = False

        # Initialize blocks

        self.rope = Rotary3DPositionEmbedding(
            hidden_size_head=head_dim,
            text_length=text_length,
            height=input_size[1] // self.patch_size_h,
            width=input_size[2] // self.patch_size_w,
            compressed_num_frames=(input_size[0] - 1) // interpolation_scale[0] + 1,
            hidden_size=inner_dim,
            learnable_pos_embed=learnable_pos_embed
        )
        if self.pre_process:
            # Init PatchEmbed
            if self.patch_type == "3D":
                self.patch_embed = VideoPatch3D(in_channels, inner_dim, self.patch_size)
            else:
                self.patch_embed = VideoPatch2D(in_channels, inner_dim, self.patch_size_h)
            self.time_embed = TimeStepEmbedding(inner_dim, self.time_embed_dim)

            if self.ofs_embed_dim is not None:
                self.ofs_embed = nn.Sequential(
                    nn.Linear(self.ofs_embed_dim, self.ofs_embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.ofs_embed_dim, self.ofs_embed_dim),
                )

            # Init Projection
            self.caption_projection = None
            if text_hidden_size is not None:
                self.caption_projection = nn.Linear(self.text_hidden_size, inner_dim)

        self.global_layer_idx = global_layer_idx if global_layer_idx is not None else tuple(range(num_layers))

        # Init VideoDiTBlock
        self.videodit_blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    dim=inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    use_rope=use_rope,
                    rope=self.rope,
                    interpolation_scale=interpolation_scale,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                    time_embed_dim=self.time_embed_dim,
                    patch_size=self.patch_size
                )
                for i in range(num_layers)
            ]
        )

        if self.post_process:
            # Init Norm
            self.norm_final = nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine, eps=1e-5)
            config = core_transformer_config_from_args(args)
            config.sequence_parallel = False
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                tensor_parallel.ColumnParallelLinear(
                    self.time_embed_dim,
                    2 * inner_dim,
                    config=config,
                    init_method=config.init_method,
                    gather_output=False
                )
            )
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine, eps=1e-6)
            self.proj_out_linear = nn.Linear(inner_dim, reduce(mul, self.patch_size) * self.out_channels)

            for param in self.norm_final.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)
            for param in self.norm_out.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)

        print(self)

    def _get_text_length(self, input_size, text_length):
        t, h, w = input_size
        cp = mpu.get_context_parallel_world_size()
        tp_sp = mpu.get_tensor_model_parallel_world_size() if self.sequence_parallel else 1
        tp_sp_rank = mpu.get_tensor_model_parallel_rank() if self.sequence_parallel else 0
        seq_len = text_length + t * h * w // self.patch_size_t // self.patch_size_h // self.patch_size_w
        seq_begin = (seq_len // cp // tp_sp) * (mpu.get_context_parallel_rank() * tp_sp + tp_sp_rank)
        seq_end = (seq_len // cp // tp_sp) + seq_begin
        if seq_end < text_length:
            return seq_len // cp // tp_sp
        elif seq_begin > text_length:
            return 0
        else:
            return text_length - seq_begin

    def forward(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            latents: Shape (batch size, num latent pixels) if discrete, shape (batch size, channel, height, width) if continuous.
            timestep: Used to indicate denoising step. Optional timestep to be applied as an embedding in AdaLayerNorm.
            prompt: Conditional embeddings for cross attention layer.
            video_mask: An attention mask of shape (batch, key_tokens) is applied to latents.
            prompt_mask: Cross-attention mask applied to prompt.
            added_cond_kwargs: resolution or aspect_ratio.
            class_labels: Used to indicate class labels conditioning.
            use_image_num: The number of images use for training.
        """

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.pre_process:
            _, _, t, h, w = latents.shape
            frames = t - use_image_num
            vid_mask, img_mask = None, None
            prompt_vid_mask, prompt_img_mask = None, None

            # 1. Input
            frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
            height, width = h // self.patch_size_h, w // self.patch_size_w

            if "masked_video" in kwargs.keys() and kwargs["masked_video"] is not None:
                latents = torch.cat([latents, kwargs["masked_video"]], dim=1)

            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, \
                embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
                latents, prompt, timestep, frames)
            if self.concat_text_embed:
                latents_vid = torch.cat((prompt_vid, latents_vid), dim=1)

            if self.enable_sequence_parallelism or self.sequence_parallel:
                latents_vid = latents_vid.transpose(0, 1).contiguous()
            if self.enable_sequence_parallelism:
                latents_vid = split_forward_gather_backward(latents_vid, mpu.get_context_parallel_group(), dim=0,
                                                            grad_scale='down')
            if self.sequence_parallel:
                latents_vid = tensor_parallel.scatter_to_sequence_parallel_region(latents_vid)
        else:
            t, h, w = self.input_size  # PP currently does not support dynamic resolution.
            frames = t - use_image_num
            vid_mask, img_mask = None, None
            prompt_vid_mask, prompt_img_mask = None, None

            # 1. Input
            frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
            height, width = h // self.patch_size_h, w // self.patch_size_w

            latents_vid = latents
            prompt_vid = prompt
            timestep_vid = timestep

        frames = torch.tensor(frames)
        height = torch.tensor(height)
        width = torch.tensor(width)

        # Rotary positional embeddings
        rotary_pos_emb = self.rope(
            rope_T=torch.tensor(t / self.patch_size[0], dtype=torch.int),
            rope_H=torch.tensor(h / self.patch_size[1], dtype=torch.int),
            rope_W=torch.tensor(w / self.patch_size[2], dtype=torch.int)
        )
        if mpu.get_context_parallel_world_size() > 1:
            rotary_pos_emb = rotary_pos_emb.chunk(mpu.get_context_parallel_world_size(), dim=0)[
                mpu.get_context_parallel_rank()]

        with rng_context:
            if self.recompute_granularity == "full":
                if latents_vid is not None:
                    latents_vid = self._checkpointed_forward(
                        latents_vid,
                        video_mask=vid_mask,
                        prompt=prompt_vid,
                        prompt_mask=prompt_vid_mask,
                        timestep=timestep_vid,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width,
                        text_length=torch.tensor(self._get_text_length([t, h, w], self.rope.text_length)),
                        rotary_pos_emb=rotary_pos_emb
                    )
            else:
                for block in self.videodit_blocks:
                    if latents_vid is not None:
                        latents_vid = block(
                            latents_vid,
                            video_mask=vid_mask,
                            prompt=prompt_vid,
                            prompt_mask=prompt_vid_mask,
                            timestep=timestep_vid,
                            class_labels=class_labels,
                            frames=frames,
                            height=height,
                            width=width,
                            rotary_pos_emb=rotary_pos_emb,
                            text_length=torch.tensor(self._get_text_length([t, h, w], self.rope.text_length)),
                            checkpoint_skip_core_attention=self.checkpoint_skip_core_attention
                        )


        # 3. Output
        if self.post_process:
            output_vid, output_img = None, None
            if latents_vid is not None:
                output_vid = self._get_output_for_patched_inputs(
                    latents=latents_vid,
                    timestep=timestep_vid,
                    height=height,
                    width=width,
                )  # [b, c, t, h, w]

            if output_vid is not None and output_img is not None:
                output = torch.cat([output_vid, output_img], dim=2)
            elif output_vid is not None:
                output = output_vid
            elif output_img is not None:
                output = output_img
            return output, prompt_vid, timestep_vid
        else:
            return latents_vid, prompt_vid, timestep_vid

    def _get_block(self, layer_number):
        return self.videodit_blocks[layer_number]

    def _checkpointed_forward(
        self,
        latents,
        video_mask,
        prompt,
        prompt_mask,
        timestep,
        class_labels,
        frames,
        height,
        width,
        rotary_pos_emb,
        text_length,
        **kwargs
    ):
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
                    video_mask,
                    prompt_mask,
                    timestep,
                    class_labels,
                    frames,
                    height,
                    width,
                    rotary_pos_emb,
                    text_length,
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
                        video_mask,
                        prompt_mask,
                        timestep,
                        class_labels,
                        frames,
                        height,
                        width,
                        rotary_pos_emb,
                        text_length,
                    )
                else:
                    block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        video_mask=video_mask,
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        timestep=timestep,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width,
                        rotary_pos_emb=rotary_pos_emb,
                        text_length=text_length,
                        checkpoint_skip_core_attention=self.checkpoint_skip_core_attention
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

    def _operate_on_patched_inputs(self, latents, prompt, timestep, frames):
        b, _, t, h, w = latents.shape
        if self.rope is not None:
            latents_vid, latents_img = self.patch_embed(latents.to(self.dtype), prompt,
                                                        rope_T=t // self.patch_size[0],
                                                        rope_H=h // self.patch_size[1],
                                                        rope_W=w // self.patch_size[2])
            _, seq_len, _ = latents_vid.shape
            pos_emb = self.rope.position_embedding_forward(latents.to(self.dtype),
                                                           seq_length=seq_len - self.rope.text_length)
            if pos_emb is not None:
                latents_vid = latents_vid + pos_emb
        else:
            latents_vid, latents_img = self.patch_embed(latents.to(self.dtype), frames,
                                                        rope_T=t // self.patch_size[0],
                                                        rope_H=h // self.patch_size[1],
                                                        rope_W=w // self.patch_size[2])
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        prompt_vid, prompt_img = None, None

        if self.time_embed is not None:
            timestep_vid = self.time_embed(timestep)
            if self.ofs_embed_dim is not None:
                ofs_emb = timestep_embedding(latents.new_full((1,), fill_value=2.0), self.ofs_embed_dim, dtype=self.dtype)
                ofs_emb = self.ofs_embed(ofs_emb)
                timestep_vid = timestep_vid + ofs_emb

        if self.caption_projection is not None:
            prompt = self.caption_projection(prompt)
            if latents_vid is None:
                prompt_img = rearrange(prompt, 'b 1 l d -> (b 1) l d')
            else:
                prompt_vid = rearrange(prompt[:, :1], 'b 1 l d -> (b 1) l d')
                if latents_img is not None:
                    prompt_img = rearrange(prompt[:, 1:], 'b i l d -> (b i) l d')

        return latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    def _get_output_for_patched_inputs(self, latents, timestep, height=None, width=None):
        x = self.norm_final(latents)
        _scale_shift_table = self.adaLN_modulation(timestep)[0]
        if self.sequence_parallel:
            _scale_shift_table = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(
                _scale_shift_table
            )
        else:
            _scale_shift_table = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(
                _scale_shift_table
            )
        if self.sequence_parallel or self.enable_sequence_parallelism:
            shift, scale = _scale_shift_table.unsqueeze(0).chunk(2, dim=2)
        else:
            shift, scale = _scale_shift_table.unsqueeze(1).chunk(2, dim=2)
        x = self.norm_out(x) * (1 + scale) + shift
        if self.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x, tensor_parallel_output_grad=False)
        if self.sequence_parallel or self.enable_sequence_parallelism:
            x = x.transpose(0, 1).contiguous()
        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, mpu.get_context_parallel_group(), dim=1, grad_scale="up")
        x = x[:, self.rope.text_length:, :]
        x = self.proj_out_linear(x)
        latents = x

        # unpatchify
        output = rearrange(latents, "b (t h w) (c o p q) -> b (t o) c (h p) (w q)",
                           b=latents.shape[0], h=height, w=width,
                           o=self.patch_size_t, p=self.patch_size_h, q=self.patch_size_w,
                           c=self.out_channels).transpose(1, 2)
        return output

    def initialize_pipeline_tensor_shapes(self):
        args = get_args()
        micro_batch_size = args.mm.data.dataloader_param.batch_size
        dtype = args.params_dtype
        latent_size = (self.out_channels, *self.input_size)
        seq_len = self.seq_len

        if self.enable_sequence_parallelism and not self.sequence_parallel:
            prev_output_shape = (seq_len // mpu.get_context_parallel_world_size(), micro_batch_size, self.inner_dim) # SBH
        elif self.sequence_parallel:
            prev_output_shape = (seq_len // mpu.get_context_parallel_world_size() // mpu.get_tensor_model_parallel_world_size(), micro_batch_size, self.inner_dim) # SBH
        else:
            prev_output_shape = (micro_batch_size, seq_len, self.inner_dim)  # BSH

        pipeline_tensor_shapes = [
            {'shape': prev_output_shape, 'dtype': dtype},                                           # prev_stage_output
            {'shape': (micro_batch_size, *latent_size), 'dtype': dtype},                            # latents
            {'shape': (micro_batch_size, self.ori_text_length, self.inner_dim), 'dtype': dtype},    # prompt
            {'shape': (micro_batch_size, self.time_embed_dim), 'dtype': dtype},                     # embedded_timestep
            {'shape': (micro_batch_size, *latent_size), 'dtype': torch.float32},                    # video_diffusion: self.noised_start
            {'shape': (micro_batch_size, 1, 1, 1, 1), 'dtype': torch.float32},                      # video_diffusion: self.c_out
            {'shape': (micro_batch_size), 'dtype': torch.float32},                                  # video_diffusion: self.alpha_cumprod
        ]
        return pipeline_tensor_shapes

    def pipeline_set_prev_stage_tensor(self, input_tensor_list, extra_kwargs=None):
        """
        Process tensor from prev_pipeline_stage, and adjust to predictor input and training loss input.
        Input:
            input_tensor_list:
                model_output, latents, predictor_prompt, predictor_timesteps,
            extra_kwargs (extra parameter for video_diffusion):
                extra_kwargs["noised_start"], extra_kwargs["c_out"], extra_kwargs["alphas_cumprod"]
        Return:
            predictor_input_list (input for self.predictor in SoraModel.forward):
                predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, prompt_prompt_mask
            training_loss_input_list (input for self.compute_loss in SoraModel.forward):
                latents, noised_latents, timesteps, noise, video_mask
        """
        predictor_input_latent, latents, predictor_prompt, predictor_timesteps, \
            extra_kwargs["noised_start"], extra_kwargs["c_out"], extra_kwargs["alphas_cumprod"] = input_tensor_list
        predictor_video_mask, predictor_prompt_mask = None, None
        predictor_input_list = [predictor_input_latent, predictor_timesteps, predictor_prompt, predictor_video_mask, predictor_prompt_mask]
        training_loss_input_list = [latents, None, predictor_timesteps, None, predictor_video_mask]

        return predictor_input_list, training_loss_input_list

    def pipeline_set_next_stage_tensor(self, input_list, output_list, extra_kwargs=None):
        """
        Process predictor output tensors from curr pipeline stage, and adjust to next pipeline stage
        Input:
            input_list: [latents, noised_latents, timesteps, noise, video_mask]
            output_list: [predictor_output, predictor_prompt, predictor_timesteps]
        Return:
            predictor_output, latents, predictor_prompt, predictor_timesteps, extra_kwargs["noised_start"], extra_kwargs["c_out"], extra_kwargs["alphas_cumprod"]
        """
        return [output_list[0], input_list[0], output_list[1], output_list[2], \
                extra_kwargs["noised_start"], extra_kwargs["c_out"], extra_kwargs["alphas_cumprod"]]


class VideoDiTBlock(nn.Module):
    """
    A basic dit block for video generation.

    Args:
        dim: The number out channels in the input and output.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        interpolation_scale: The scale for interpolation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        enable_sequence_parallelism: bool = False,
        time_embed_dim=None,
        rope=None,
        patch_size=None
    ):
        super().__init__()
        self.patch_size = patch_size
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else dim
        self.cross_attention_dim = cross_attention_dim
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.norm_type = norm_type
        self.positional_embeddings = positional_embeddings


        # Define three blocks. Each block has its own normalization layer.
        # 1. Scale-shift.
        config = core_transformer_config_from_args(args)
        config.sequence_parallel = False
        self.scale_shift_table = nn.Sequential(
            nn.SiLU(),
            tensor_parallel.ColumnParallelLinear(
                self.time_embed_dim,
                12 * dim,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
        )

        # 2. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError("If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined.")
        if positional_embeddings == "sinusoidal":
            self.rope = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.rope = rope

        self.enable_sequence_parallelism = enable_sequence_parallelism
        if self.enable_sequence_parallelism or self.sequence_parallel:
            self.self_atten = ParallelAttention(
                query_dim=dim,
                key_dim=None,
                num_attention_heads=num_heads,
                hidden_size=num_heads * head_dim,
                proj_q_bias=attention_bias,
                proj_k_bias=attention_bias,
                proj_v_bias=attention_bias,
                proj_out_bias=attention_out_bias,
                dropout=dropout,
                use_qk_norm=(norm_type == "qk_ln"),
                norm_type="layernorm",
                norm_eps=1e-6,
                rope=self.rope,  # Here is an object
                is_qkv_concat=True
            )
        else:
            self.self_atten = ParallelAttention(
                query_dim=dim,
                key_dim=None,
                num_attention_heads=num_heads,
                hidden_size=head_dim * num_heads,
                dropout=dropout,
                proj_q_bias=attention_bias,
                proj_k_bias=attention_bias,
                proj_v_bias=attention_bias,
                proj_out_bias=attention_out_bias,
                norm_type="layernorm",
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=1e-6,
                use_qk_norm=(norm_type == "qk_ln"),
                rope=self.rope,
                interpolation_scale=interpolation_scale,
                is_qkv_concat=True,
                fa_layout="bnsd"
            )

        # 3. Feed-forward
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = TensorParallelFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias
        )

        for param in self.norm1.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        for param in self.norm2.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        latents: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        timestep: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        frames: torch.int64 = None,
        height: torch.int64 = None,
        width: torch.int64 = None,
        rotary_pos_emb=None,
        text_length: torch.int64 = None,
        checkpoint_skip_core_attention: bool = False,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # before_self_attention
        if checkpoint_skip_core_attention:
            q, k, v, gate_msa, text_gate_msa, scale_mlp, shift_mlp, text_scale_mlp, text_shift_mlp, gate_mlp, text_gate_mlp = tensor_parallel.checkpoint(
                self.before_attention,
                False,
                latents,
                timestep,
                frames,
                height,
                width,
                text_length,
                rotary_pos_emb
            )
        else:
            q, k, v, gate_msa, text_gate_msa, scale_mlp, shift_mlp, text_scale_mlp, text_shift_mlp, gate_mlp, text_gate_mlp = self.before_attention(
                latents, timestep, frames, height, width, text_length, rotary_pos_emb)

        # self_attention
        attn_output = self.attention(
            q=q,
            k=k,
            v=v,
            mask=None
        )

        # after_attention
        if checkpoint_skip_core_attention:
            latents = tensor_parallel.checkpoint(
                self.after_attetion,
                False,
                latents,
                attn_output,
                gate_msa,
                text_gate_msa,
                scale_mlp,
                shift_mlp,
                text_scale_mlp,
                text_shift_mlp,
                gate_mlp,
                text_gate_mlp,
                text_length,
            )
        else:
            latents = self.after_attetion(
                latents,
                attn_output,
                gate_msa,
                text_gate_msa,
                scale_mlp,
                shift_mlp,
                text_scale_mlp,
                text_shift_mlp,
                gate_mlp,
                text_gate_mlp,
                text_length,
            )

        return latents

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def before_attention(
        self,
        latents: torch.Tensor,
        timestep: Dict[str, torch.Tensor] = None,
        frames: torch.int64 = None,
        height: torch.int64 = None,
        width: torch.int64 = None,
        text_length: torch.int64 = None,
        rotary_pos_emb=None,
        key: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ):
        # 1. Self-Attention
        frames = frames.item()
        height = height.item()
        width = width.item()
        text_length = text_length.item()
        input_layout = "sbh" if self.enable_sequence_parallelism or self.sequence_parallel else "bsh"

        if self.enable_sequence_parallelism or self.sequence_parallel:
            _scale_shift_table = self.scale_shift_table(timestep)[0]
            if self.sequence_parallel:
                _scale_shift_table = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(
                    _scale_shift_table
                )
            else:
                _scale_shift_table = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(
                    _scale_shift_table
                )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                text_shift_msa,
                text_scale_msa,
                text_gate_msa,
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ) = _scale_shift_table.unsqueeze(0).chunk(12, dim=2)
            latents_text = latents[:text_length]
            latents_vid = latents[text_length:]
        else:
            _scale_shift_table = self.scale_shift_table(timestep)[0]
            _scale_shift_table = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(
                _scale_shift_table
            )
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                text_shift_msa,
                text_scale_msa,
                text_gate_msa,
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ) = _scale_shift_table.unsqueeze(1).chunk(12, dim=2)
            latents_text = latents[:, :text_length]
            latents_vid = latents[:, text_length:]
        latents_vid = self.norm1(latents_vid)
        latents_text = self.norm1(latents_text)
        latents_vid = latents_vid * (1 + scale_msa) + shift_msa
        latents_text = latents_text * (1 + text_scale_msa) + text_shift_msa
        if self.enable_sequence_parallelism or self.sequence_parallel:
            norm_latents = torch.cat((latents_text, latents_vid), dim=0)  # (s_t + t * h/2 * w/2, b, n * d)
        else:
            norm_latents = torch.cat((latents_text, latents_vid), dim=1)  # (b, s_t + t * h/2 * w/2, n * d)

        q, k, v = self.self_atten.function_before_core_attention(norm_latents, key, input_layout, rotary_pos_emb)

        return q, k, v, gate_msa, text_gate_msa, scale_mlp, shift_mlp, text_scale_mlp, text_shift_mlp, gate_mlp, text_gate_mlp

    def attention(self, q, k, v, mask=None):
        attn_output = self.self_atten.core_attention_flash(q, k, v, mask)
        return attn_output

    def after_attetion(
        self,
        latents,
        attn_output,
        gate_msa,
        text_gate_msa,
        scale_mlp,
        shift_mlp,
        text_scale_mlp,
        text_shift_mlp,
        gate_mlp,
        text_gate_mlp,
        text_length,
    ):
        text_length = text_length.item()
        output_layout = "sbh" if self.enable_sequence_parallelism or self.sequence_parallel else "bsh"
        attn_output = self.self_atten.function_after_core_attention(attn_output, output_layout)

        if self.enable_sequence_parallelism or self.sequence_parallel:
            attn_vid_output = gate_msa * attn_output[text_length:]
            attn_text_output = text_gate_msa * attn_output[:text_length]
            attn_output = torch.cat((attn_text_output, attn_vid_output), dim=0)
        else:
            attn_vid_output = gate_msa * attn_output[:, text_length:]
            attn_text_output = text_gate_msa * attn_output[:, :text_length]
            attn_output = torch.cat((attn_text_output, attn_vid_output), dim=1)

        latents = attn_output + latents

        # 2. Feed-forward
        if self.enable_sequence_parallelism or self.sequence_parallel:
            latents_text = latents[:text_length]
            latents_vid = latents[text_length:]
            latents_text = self.norm2(latents_text)
            latents_vid = self.norm2(latents_vid)
            latents_vid = latents_vid * (1 + scale_mlp) + shift_mlp
            latents_text = latents_text * (1 + text_scale_mlp) + text_shift_mlp
            norm_latents = torch.cat((latents_text, latents_vid), dim=0)
        else:
            latents_text = latents[:, :text_length]
            latents_vid = latents[:, text_length:]
            latents_text = self.norm2(latents_text)
            latents_vid = self.norm2(latents_vid)
            latents_vid = latents_vid * (1 + scale_mlp) + shift_mlp
            latents_text = latents_text * (1 + text_scale_mlp) + text_shift_mlp
            norm_latents = torch.cat((latents_text, latents_vid), dim=1)

        ff_output = self.ff(norm_latents)

        if self.enable_sequence_parallelism or self.sequence_parallel:
            ff_vid_output = gate_mlp * ff_output[text_length:]
            ff_text_output = text_gate_mlp * ff_output[:text_length]
            ff_output = torch.cat((ff_text_output, ff_vid_output), dim=0)
        else:
            ff_vid_output = gate_mlp * ff_output[:, text_length:]
            ff_text_output = text_gate_mlp * ff_output[:, :text_length]
            ff_output = torch.cat((ff_text_output, ff_vid_output), dim=1)

        latents = ff_output + latents
        return latents

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim
