import math
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from megatron.core import mpu, tensor_parallel
from megatron.legacy.model.enums import AttnType
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (
    all_to_all,
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from mindspeed.core.context_parallel.model_parallel_utils import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.attention import FlashAttention, ParallelAttention
from mindspeed_mm.models.common.embeddings import TextProjection
from mindspeed_mm.models.common.normalize import normalize, FP32LayerNorm
from mindspeed_mm.models.common.fpdt_layer import (
    FPDTFlashAttention,
    split_forward_gather_backward_FPDT_tensors,
    gather_forward_split_backward_FPDT_tensors)
from mindspeed_mm.utils.utils import change_tensor_layout


class WanDiT(MultiModalModule):

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        hidden_size: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        img_dim: int = 1280,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        qk_norm: bool = True,
        qk_norm_type: str = "rmsnorm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        max_seq_len: int = 1024,
        fa_layout: str = "bnsd",
        clip_token_len: int = 257,
        pre_process: bool = True,
        post_process: bool = True,
        global_layer_idx: Optional[Tuple] = None,
        attention_async_offload: bool = False,
        fp32_calculate: bool = False,
        seperated_timestep: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        **kwargs,
    ):
        super().__init__(config=None)

        if model_type not in ["t2v", "i2v", "flf2v", "ti2v", "wan2.2-t2v", "wan2.2-i2v", "wan2.2-control-camera"]:
            raise ValueError("Please only select among 't2v', 'i2v', 'ti2v', 'flf2v', 'wan2.2-t2v', 'wan2.2-i2v' and 'wan2.2-control-camera' tasks")

        if not ((hidden_size % num_heads) == 0 and (hidden_size // num_heads) % 2 == 0):
            raise ValueError(
                "The dimension must be divisible by num_heads, and result of 'dim // num_heads' must be even"
            )

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.max_seq_len = max_seq_len
        self.fa_layout = fa_layout
        self.clip_token_len = clip_token_len
        self.pre_process = pre_process
        self.post_process = post_process
        self.global_layer_idx = global_layer_idx if global_layer_idx is not None else tuple(range(num_layers))
        self.head_dim = hidden_size // num_heads

        args = get_args()
        config = core_transformer_config_from_args(args)

        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_layers = (
            args.recompute_num_layers
            if args.recompute_num_layers is not None
            else num_layers
        )

        self.recompute_skip_core_attention = args.recompute_skip_core_attention
        self.recompute_num_layers_skip_core_attention = args.recompute_num_layers_skip_core_attention
        self.attention_async_offload = attention_async_offload
        self.fp32_calculate = fp32_calculate

        self.h2d_stream = torch_npu.npu.Stream() if attention_async_offload else None
        self.d2h_stream = torch_npu.npu.Stream() if attention_async_offload else None

        if self.recompute_granularity == "selective":
            raise ValueError(
                "recompute_granularity does not support selective mode in wanVideo"
            )
        if self.distribute_saved_activations:
            raise NotImplementedError(
                "distribute_save_activation is currently not supported"
            )

        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel

        # context parallel setting
        self.context_parallel_algo = (
            args.context_parallel_algo
            if mpu.get_context_parallel_world_size() > 1
            else None
        )
        if (
            self.context_parallel_algo is not None
            and self.context_parallel_algo
            not in ["ulysses_cp_algo", "hybrid_cp_algo", "megatron_cp_algo"]
        ):
            raise NotImplementedError(
                f"Context_parallel_algo {self.context_parallel_algo} is not implemented"
            )

        self.FPDT = args.mm.model.to_dict().get('predictor', {}).get('FPDT', False)
        self.FPDT_chunk_number = args.mm.model.to_dict().get('predictor', {}).get('FPDT_chunk_number', None)

        if self.pre_process:
            # time embeddings
            self.time_embedding = nn.Sequential(
                nn.Linear(self.freq_dim, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.fp32_calculate:
                self.time_embedding = self.time_embedding.to(torch.float32)
            # time emb projection
            self.time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 6)
            )
            # embeddings
            self.text_embedding = TextProjection(
                self.text_dim, self.hidden_size, partial(nn.GELU, approximate="tanh")
            )
            if model_type in ["i2v", "flf2v"]:
                self.img_emb = MLPProj(self.img_dim, self.hidden_size, model_type == 'flf2v', clip_token_len, self.fp32_calculate)

            self.patch_embedding = nn.Conv3d(
                self.in_dim,
                self.hidden_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

            # camera control adapter for wan2.2-control-camera
            if add_control_adapter:
                from .wan_camera_controller import SimpleAdapter
                self.control_adapter = SimpleAdapter(
                    in_dim_control_adapter, self.hidden_size,
                    kernel_size=self.patch_size[1:], stride=self.patch_size[1:]
                )
            else:
                self.control_adapter = None

        # rope
        self.rope = RoPE3DWan(head_dim=self.head_dim, max_seq_len=self.max_seq_len)

        # attention blocks
        self.blocks = nn.ModuleList(
            [
                WanDiTBlock(
                    model_type,
                    self.hidden_size,
                    self.ffn_dim,
                    self.num_heads,
                    self.qk_norm,
                    self.qk_norm_type,
                    self.cross_attn_norm,
                    self.eps,
                    rope=self.rope,
                    fa_layout=self.fa_layout,
                    clip_token_len=clip_token_len,
                    attention_async_offload=self.attention_async_offload,
                    h2d_stream=self.h2d_stream,
                    d2h_stream=self.d2h_stream,
                    layer_idx=index,
                    num_layers=self.num_layers,
                    fp32_calculate=self.fp32_calculate,
                )
                for index in range(self.num_layers)
            ]
        )

        if self.post_process:
            # head
            self.head = Head(self.hidden_size, self.out_dim, self.patch_size, self.eps)

        self.use_dpo = getattr(args.mm.model, "dpo", None)
        self.seperated_timestep = seperated_timestep

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    @property
    def device(self) -> torch.device:
        """The device of the module (assuming that all the module parameters are in the same device)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].device
        else:
            buffers = tuple(self.buffers())
            return buffers[0].device

    def sinusoidal_embedding_1d(self, dim, position, theta=10000):
        sinusoid = torch.outer(
            position.type(torch.float64),
            torch.pow(
                theta,
                -torch.arange(
                    dim // 2, dtype=torch.float64, device=position.device
                ).div(dim // 2),
            ),
        )
        embs = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return embs.to(position.dtype)

    def _checkpointed_forward(self, blocks, x, *args):
        "Forward method with activation checkpointing."
        num_layers = len(blocks)
        recompute_layers = self.recompute_layers
        recompute_num_layers_skip_core_attention = (
            self.recompute_num_layers_skip_core_attention
            if self.recompute_skip_core_attention
            else 0
        )

        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = blocks[index]
                    x_ = layer(*args)
                return x_

            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < num_layers:
                x = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.distribute_saved_activations,
                    x,
                    *args,
                )
                _layer_num += recompute_layers

        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(num_layers):
                if _layer_num < recompute_layers:
                    x = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.distribute_saved_activations,
                        x,
                        *args,
                    )
                elif _layer_num < recompute_layers + recompute_num_layers_skip_core_attention:
                    block = blocks[_layer_num]
                    x = block(x, *args, recompute_skip_core_attention=True)
                else:
                    block = blocks[_layer_num]
                    x = block(x, *args)
        else:
            raise ValueError(
                f"Invalid activation recompute method {self.recompute_method}."
            )

        return x

    def patchify(self, embs: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        # apply camera control adapter if available
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            embs = embs + y_camera
        # get f, h, w from b c f h w
        grid_sizes = embs.shape[2:]

        # b c f h w  -> b (f h w) c
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c").contiguous()

        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        # b (f h w) (x y z c) -> b c (f x) (h y) (w z)
        patch_out = rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames,
            h=height,
            w=width,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return patch_out

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor = None,
        i2v_clip_feature: torch.Tensor = None,
        i2v_vae_feature: torch.Tensor = None,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if isinstance(prompt, list):
            # 判断 list 内部元素类型
            if len(prompt) > 0 and isinstance(prompt[0], torch.Tensor):
                # 如果是 Tensor 列表，使用 stack 合并 (需保证 shape 一致)
                prompt = torch.stack(prompt)
            else:
                # 如果是数字列表 (如 token ids)，转为 LongTensor
                prompt = torch.tensor(prompt, dtype=torch.long, device=x.device)
        
        # 2. 同步处理 prompt_mask (非常重要！)
        # 因为 prompt 变成了 tensor，mask 也必须对应变成 tensor，否则后面 .view 会报错
        if prompt_mask is not None and isinstance(prompt_mask, list):
            if len(prompt_mask) > 0 and isinstance(prompt_mask[0], torch.Tensor):
                prompt_mask = torch.stack(prompt_mask)
            else:
                prompt_mask = torch.tensor(prompt_mask, dtype=torch.bool, device=x.device)

        if self.pre_process:
            timestep = timestep.to(x[0].device)
            if self.seperated_timestep:  # wan2.2 5B
                timestep = torch.concat([
                    torch.zeros((1, x.shape[3] * x.shape[4] // 4), dtype=x.dtype, device=x.device),
                    torch.ones((x.shape[2] - 1, x.shape[3] * x.shape[4] // 4), dtype=x.dtype,
                               device=x.device) * timestep
                ])
            if timestep.ndim == 2:
                ts_seq_len = timestep.shape[1]
                timestep = timestep.flatten()
            else:
                ts_seq_len = None

            # time embeddings
            timestep = self.sinusoidal_embedding_1d(self.freq_dim, timestep)
            if ts_seq_len is not None and not self.seperated_timestep:
                timestep = timestep.unflatten(0, (-1, ts_seq_len))
            times = self.time_embedding(timestep)
            if self.seperated_timestep:
                times = times.unsqueeze(0)
            time_emb = self.time_projection(times)
            if ts_seq_len is None:
                time_emb = time_emb.unflatten(1, (6, self.hidden_size))
            else:
                time_emb = time_emb.unflatten(2, (6, self.hidden_size))

            # prompt embeddings
            bs = prompt.size(0)
            prompt = prompt.view(bs, -1, prompt.size(-1))
            if prompt_mask is not None:
                seq_lens = prompt_mask.view(bs, -1).sum(dim=-1)
                seq_lens = seq_lens.to(torch.int64)
                for i, seq_len in enumerate(seq_lens):
                    prompt[i, seq_len:] = 0
            prompt_emb = self.text_embedding(prompt)

            # cat i2v & flf2v
            if self.model_type in ["i2v", "flf2v"]:
                i2v_clip_feature = i2v_clip_feature.to(x)
                i2v_vae_feature = i2v_vae_feature.to(x)
                x = torch.cat([x, i2v_vae_feature], dim=1)  # (b, c[x+y], f, h, w)
                clip_embedding = self.img_emb(i2v_clip_feature.float() if self.fp32_calculate else i2v_clip_feature.to(time_emb.dtype))
                prompt_emb = torch.cat([clip_embedding, prompt_emb], dim=1)
            elif self.model_type in ["wan2.2-i2v", "wan2.2-control-camera"]:
                i2v_vae_feature = i2v_vae_feature.to(x)
                x = torch.cat([x, i2v_vae_feature], dim=1)  # (b, c[x+y], f, h, w)

            # patch embedding
            patch_emb = self.patch_embedding(x.to(time_emb.dtype))

            embs, grid_sizes = self.patchify(patch_emb)

            # rotary positional embeddings
            batch_size, frames, height, width = (
                embs.shape[0],
                grid_sizes[0],
                grid_sizes[1],
                grid_sizes[2],
            )
        else:
            batch_size, _, frames, height, width = kwargs["ori_shape"]
            height, width = height // self.patch_size[1], width // self.patch_size[2]
            prompt_emb = kwargs['prompt_emb']
            time_emb = kwargs['time_emb']
            times = kwargs['times']
            embs = x

        rotary_pos_emb = self.rope(batch_size, frames, height, width)

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # cp split
        if self.context_parallel_algo is not None:
            if self.pre_process:
                if self.FPDT:
                    embs = split_forward_gather_backward_FPDT_tensors(embs, seq_dim=1, chunk_number=self.FPDT_chunk_number,
                                              group=mpu.get_context_parallel_group(), grad_scale="down")

                    if time_emb.ndim == 4:
                        time_emb = split_forward_gather_backward_FPDT_tensors(time_emb, seq_dim=1, chunk_number=self.FPDT_chunk_number,
                                                      group=mpu.get_context_parallel_group(), grad_scale="down")
                else:
                    embs = split_forward_gather_backward(
                        embs, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
                    )  # b s h
                    if time_emb.ndim == 4:
                        # b s 6 h for adaLN in wan2.2 5b
                        time_emb = split_forward_gather_backward(
                            time_emb, mpu.get_context_parallel_group(), dim=1, grad_scale="down"
                        )
            if self.FPDT:
                rotary_pos_emb = split_forward_gather_backward_FPDT_tensors(rotary_pos_emb, seq_dim=0, chunk_number=self.FPDT_chunk_number,
                                                    group=mpu.get_context_parallel_group(), grad_scale="down")
            else:
                rotary_pos_emb = split_forward_gather_backward(
                    rotary_pos_emb,
                    mpu.get_context_parallel_group(),
                    dim=0,
                    grad_scale="down",
                )

        with rng_context:
            if self.recompute_granularity == "full":
                embs = self._checkpointed_forward(
                    self.blocks,
                    embs,
                    prompt_emb,
                    time_emb,
                    rotary_pos_emb,
                )
            else:
                for block in self.blocks:
                    embs = block(embs, prompt_emb, time_emb, rotary_pos_emb)

        out = embs
        if self.post_process:
            if self.context_parallel_algo is not None:
                if self.FPDT:
                    embs = gather_forward_split_backward_FPDT_tensors(embs, seq_dim=1, chunk_number=self.FPDT_chunk_number,
                                                   group=mpu.get_context_parallel_group(), grad_scale="up")
                    if time_emb.ndim == 4:
                        time_emb = gather_forward_split_backward_FPDT_tensors(time_emb, seq_dim=1, chunk_number=self.FPDT_chunk_number,
                                                           group=mpu.get_context_parallel_group(), grad_scale="up")
                else:
                    embs = gather_forward_split_backward(
                        embs, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
                    )
                    if time_emb.ndim == 4:
                        # b s 6 h for adaLN in wan2.2 5b
                        time_emb = gather_forward_split_backward(
                            time_emb, mpu.get_context_parallel_group(), dim=1, grad_scale="up"
                        )
            embs_out = self.head(embs, times)
            out = self.unpatchify(embs_out, frames, height, width)

        rtn = (out, prompt, prompt_emb, time_emb, times, prompt_mask)

        return rtn

    def pipeline_set_prev_stage_tensor(self, input_tensor_list, extra_kwargs):
        """
        Implemented for pipeline parallelism. The input tensor is got from last PP stage.
        Args:
            input_tensor_list: same as the return value of pipeline_set_next_stage_tensor
            extra_kwargs: kwargs for forward func.

        Returns:
            predictor_input_list: values for predictor forward.
            training_loss_input_list: values to calculate loss.
        """
        score, score_lose = None, None
        if self.use_dpo is not None:
            (prev_output, prompt, prompt_emb, time_emb, times, prompt_mask, score, score_lose,
            latents, noised_latents, timesteps, noise) = input_tensor_list
        else:
            (prev_output, prompt, prompt_emb, time_emb, times, prompt_mask,
            latents, noised_latents, timesteps, noise) = input_tensor_list
        predictor_input_list = [prev_output, timesteps, prompt, None, prompt_mask]
        training_loss_input_list = [latents, noised_latents, timesteps, noise, None]
        extra_kwargs['prompt_emb'] = prompt_emb
        extra_kwargs['time_emb'] = time_emb
        extra_kwargs['times'] = times
        extra_kwargs["ori_shape"] = latents.shape
        if self.use_dpo is not None:
            score_list = [score, score_lose]
            return predictor_input_list, training_loss_input_list, score_list
        return predictor_input_list, training_loss_input_list

    def pipeline_set_next_stage_tensor(self, input_list, output_list, extra_kwargs=None):
        """
        input_list: [latents, noised_latents, timesteps, noise, video_mask]
        output_list (predict_output):[out, prompt, prompt_emb, time_emb, times, prompt_mask]

        return as
        prev_output, prompt, prompt_emb, prompt_emb, time_emb, times, prompt_mask,
        latents, timesteps, noise

        which should be corresponded with initialize_pipeline_tensor_shapes
        """
        latents, noised_latents, timesteps, noise, _ = input_list
        if timesteps.dtype != torch.float32:
            timesteps = timesteps.to(torch.float32)

        return list(output_list) + [latents, noised_latents, timesteps, noise]

    @staticmethod
    def initialize_pipeline_tensor_shapes():
        args = get_args()
        micro_batch_size = args.micro_batch_size
        dtype = args.params_dtype

        model_cfg = args.mm.model
        data_cfg = args.mm.data.dataset_param.preprocess_parameters
        hidden_size = model_cfg.predictor.hidden_size
        height = getattr(data_cfg, "max_height", 480)
        width = getattr(data_cfg, "max_width", 832)
        vae_scale_factor = getattr(model_cfg.predictor, "vae_scale_factor", [4, 8, 8])
        latent_size = ((data_cfg.num_frames + 3) // vae_scale_factor[0], height // vae_scale_factor[1], width // vae_scale_factor[2])
        divisor = model_cfg.predictor.patch_size[0] * model_cfg.predictor.patch_size[1] * \
                  model_cfg.predictor.patch_size[2]
        seq_len = latent_size[0] * latent_size[1] * latent_size[2] // divisor // mpu.get_context_parallel_world_size()
        channels = model_cfg.predictor.out_dim
        text_dim = model_cfg.predictor.text_dim
        text_len = model_cfg.predictor.text_len
        img_token_len = model_cfg.predictor.clip_token_len if model_cfg.predictor.model_type == 'i2v' else 0
        rtn_size = 1
        use_dpo = getattr(model_cfg, "dpo", None)
        if use_dpo is not None:
            micro_batch_size = micro_batch_size * 2
            rtn_size = 2
        pipeline_tensor_shapes = [
            {'shape': (micro_batch_size * rtn_size, seq_len, hidden_size), 'dtype': dtype},  # prev_output
            {'shape': (micro_batch_size * rtn_size, text_len, text_dim), 'dtype': dtype},  # prompt
            {'shape': (micro_batch_size * rtn_size, text_len + img_token_len, hidden_size), 'dtype': dtype},  # prompt_emb
            {'shape': (micro_batch_size * rtn_size, 6, hidden_size), 'dtype': dtype},  # time_emb
            {'shape': (micro_batch_size * rtn_size, hidden_size), 'dtype': dtype},  # times
            {'shape': (micro_batch_size * rtn_size, 1, text_len), 'dtype': dtype},  # origin_prompt_mask
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # latents(x0)
            {"shape": (micro_batch_size, channels, *latent_size), "dtype": dtype},  # noised_latents
            {'shape': (micro_batch_size,), 'dtype': torch.float32},  # timesteps
            {'shape': (micro_batch_size, channels, *latent_size), 'dtype': dtype},  # noise
        ]

        if use_dpo is not None:
            score_shape = [
                {'shape': (1,), 'dtype': torch.float64},  # score_win
                {'shape': (1,), 'dtype': torch.float64},  # score_lose
            ]
            pipeline_tensor_shapes = pipeline_tensor_shapes[:6] + score_shape + pipeline_tensor_shapes[6:]

        return pipeline_tensor_shapes


class WanDiTBlock(nn.Module):

    def __init__(
        self,
        model_type: "t2v",
        hidden_size: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rmsnorm",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        dropout: float = 0.0,
        rope=None,
        fa_layout=None,
        clip_token_len: int = 257,
        attention_async_offload: bool = False,
        layer_idx: int = 0,
        num_layers: int = 40,
        fp32_calculate: bool = False,
        h2d_stream: Optional[torch_npu.npu.Stream] = None,
        d2h_stream: Optional[torch_npu.npu.Stream] = None,
        **kwargs
    ):
        super().__init__()

        self.model_type = model_type
        self.rope = rope
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.clip_token_len = clip_token_len
        self.fp32_calculate = fp32_calculate

        args = get_args()

        self.FPDT = args.mm.model.to_dict().get('predictor', {}).get('FPDT', False)
        self.FPDT_chunk_number = args.mm.model.to_dict().get('predictor', {}).get('FPDT_chunk_number', None)
        self.distribute_saved_activations = args.distribute_saved_activations

        self.attention_async_offload_param = {
            "async_offload": attention_async_offload,
            "block_idx": layer_idx,
            "depth": num_layers,
            "h2d_stream": h2d_stream,
            "d2h_stream": d2h_stream,
        }

        # modulation
        self.modulation = nn.Parameter(
            torch.randn(1, 6, self.hidden_size) / self.hidden_size**0.5
        )
        self.norm1 = nn.LayerNorm(self.hidden_size, eps=eps, elementwise_affine=False)
        self.self_attn = WanVideoParallelAttention(
            query_dim=hidden_size,
            key_dim=None,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_bias,
            proj_k_bias=attention_bias,
            proj_v_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            use_qk_norm=qk_norm,
            norm_type=qk_norm_type,
            norm_eps=eps,
            rope=rope,
            attention_type=AttnType.self_attn,
            has_img_input=False,
            fa_layout=fa_layout,
        )
        self.norm3 = FP32LayerNorm(self.hidden_size, eps=eps) if fp32_calculate else nn.LayerNorm(self.hidden_size, eps=eps)

        self.cross_attn = WanVideoParallelAttention(
            query_dim=hidden_size,
            key_dim=None,
            num_attention_heads=num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_bias,
            proj_k_bias=attention_bias,
            proj_v_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            dropout=dropout,
            use_qk_norm=qk_norm,
            norm_type=qk_norm_type,
            norm_eps=eps,
            attention_type=AttnType.cross_attn,
            has_img_input=model_type in ["i2v", "flf2v"],
            fa_layout=fa_layout,
        )

        self.norm2 = nn.LayerNorm(self.hidden_size, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, self.hidden_size),
        )

    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale) + shift

    def forward(
        self,
        latents,
        prompt,
        time_emb,
        rotary_pos_emb,
        recompute_skip_core_attention=False
    ):
        # Fixes the memory reorganization problem triggered when fast host dispatch
        # and tensor multi-stream reuse occur simultaneously.
        torch.npu.synchronize()

        # before self attention process
        if recompute_skip_core_attention:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = tensor_parallel.checkpoint(
                self._before_self_attention,
                self.distribute_saved_activations,
                time_emb,
                latents,
                rotary_pos_emb,
            )
        else:
            query, key, value, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._before_self_attention(
                time_emb,
                latents,
                rotary_pos_emb
            )

        # self attention
        attention_async_offload_param = (
            self.attention_async_offload_param
            if recompute_skip_core_attention
            else {}
        )
        self_attn_out = self.self_attn.core_attention_flash(
            query=query,
            key=key,
            value=value,
            **attention_async_offload_param
        )

        # after self attention
        if recompute_skip_core_attention:
            latents = tensor_parallel.checkpoint(
                self._after_self_attention,
                self.distribute_saved_activations,
                self_attn_out,
                latents,
                prompt,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp
            )
        else:
            latents = self._after_self_attention(
                self_attn_out,
                latents,
                prompt,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp
            )

        return latents

    def _before_self_attention(
        self,
        time_emb,
        latents,
        rotary_pos_emb
    ):
        dtype = time_emb.dtype
        modu_dtype = torch.float32 if self.fp32_calculate else dtype
        device = time_emb.device
        has_seq = time_emb.ndim == 4
        chunk_dim = 2 if has_seq else 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=modu_dtype, device=device) + time_emb.to(modu_dtype)
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )

        self_attn_input = self.modulate(
            self.norm1(latents.to(torch.float32)), shift_msa, scale_msa
        ).to(dtype)

        # before self attention
        query, key, value = self.self_attn.function_before_core_attention(
            query=self_attn_input,
            input_layout="bsh",
            rotary_pos_emb=rotary_pos_emb.to(time_emb.device)
        )

        return (
            query, key, value,
            gate_msa, shift_mlp, scale_mlp, gate_mlp
        )

    def _after_self_attention(
        self,
        self_attn_out,
        latents,
        prompt,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp
    ):
        dtype = torch.float32 if self.fp32_calculate else latents.dtype
        self_attn_out = self.self_attn.function_after_core_attention(self_attn_out, output_layout="bsh")

        latents = (latents + gate_msa * self_attn_out).to(latents.dtype)

        # cross attention
        crs_attn_input = self.norm3(latents.to(dtype)).to(latents.dtype)

        # i2v & flf2v
        if self.model_type in ["i2v", "flf2v"]:
            img_clip_token_len = 2 * self.clip_token_len if self.model_type == "flf2v" else self.clip_token_len
            img = prompt[:, :img_clip_token_len]
            txt = prompt[:, img_clip_token_len:]
            crs_attn_out = self.cross_attn(
                query=crs_attn_input,
                key=(img, txt),
                input_layout="bsh",
            )
        # t2v
        else:
            txt = prompt
            crs_attn_out = self.cross_attn(
                query=crs_attn_input,
                key=txt,
                input_layout="bsh",
            )

        latents = latents + crs_attn_out
        modu_out = self.modulate(self.norm2(latents.to(dtype)), shift_mlp, scale_mlp).to(latents.dtype)

        # ffn
        if self.FPDT:
            latents = ((latents.to(dtype)) + gate_mlp * self.fpdt_ffn(modu_out).to(dtype)).to(latents.dtype)
        else:
            latents = ((latents.to(dtype)) + gate_mlp * self.ffn(modu_out).to(dtype)).to(latents.dtype)

        return latents

    def fpdt_ffn(self, x):
        outs = []
        inputs = torch.chunk(x, dim=1, chunks=self.FPDT_chunk_number)
        for input_chunk in inputs:
            outs.append(self.ffn(input_chunk))
        output = torch.concat(outs, dim=1).contiguous()
        return output


class RoPE3DWan(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.freqs = self.get_freq(head_dim)
        self.freqs = [x.npu() for x in self.freqs]

    def get_freq(self, head_dim):
        if head_dim <= 0:
            raise ValueError("head dimension must be greater than 0")

        dim1 = head_dim - 2 * (head_dim // 3)
        dim2 = head_dim // 3

        # generate frequency matrices
        freqs1 = self.rope_params(self.max_seq_len, dim1)
        freqs2 = self.rope_params(self.max_seq_len, dim2)
        freqs3 = self.rope_params(self.max_seq_len, dim2)
        return freqs1, freqs2, freqs3

    def rope_params(self, max_seq_len, dim, theta=10000):
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")

        # compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
        freqs = torch.outer(torch.arange(max_seq_len, device=freqs.device), freqs)

        # convert to complex numbers
        freqs = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        return freqs

    def apply_rotary_pos_emb(self, tokens, freqs):
        dtype = tokens.dtype
        cos, sin = torch.chunk(torch.view_as_real(freqs.to(torch.complex64)), 2, dim=-1)

        B, S, N, D = tokens.shape

        def rotate_half(x):
            half_1, half_2 = torch.chunk(x.reshape((B, S, N, D // 2, 2)), 2, dim=-1)
            return torch.cat((-half_2, half_1), dim=-1).reshape((B, S, N, D))

        cos = cos.expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = sin.expand(-1, -1, -1, -1, 2).flatten(-2)
        res = tokens * cos + rotate_half(tokens) * sin

        return res.to(dtype)

    def forward(self, b, f, h, w):
        seq_len = f * h * w

        # get freqs
        freqs = (
            torch.cat(
                [
                    self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(seq_len, 1, 1, -1)
            .expand(seq_len, b, 1, -1)
        )
        return freqs


class WanVideoParallelAttention(ParallelAttention):

    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int],
        num_attention_heads: int,
        hidden_size: int,
        proj_q_bias: bool = False,
        proj_k_bias: bool = False,
        proj_v_bias: bool = False,
        proj_out_bias: bool = False,
        dropout: float = 0.0,
        use_qk_norm: bool = False,
        norm_type: str = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: int = AttnType.self_attn,
        has_img_input: bool = False,
        fa_layout: str = "bnsd",
        rope=None,
        **kwargs,
    ):
        super().__init__(
            query_dim=query_dim,
            key_dim=key_dim,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            proj_q_bias=proj_q_bias,
            proj_k_bias=proj_k_bias,
            proj_v_bias=proj_v_bias,
            proj_out_bias=proj_out_bias,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            is_qkv_concat=False,
            attention_type=attention_type,
            is_kv_concat=False,
            fa_layout=fa_layout,
            rope=rope,
            **kwargs,
        )

        args = get_args()

        self.FPDT = args.mm.model.to_dict().get('predictor', {}).get('FPDT', False)
        self.FPDT_chunk_number = args.mm.model.to_dict().get('predictor', {}).get('FPDT_chunk_number', None)
        self.FPDT_with_offload = args.mm.model.to_dict().get('predictor', {}).get('FPDT_with_offload', False)


        if self.cp_size > 1 and attention_type == AttnType.self_attn \
            and args.context_parallel_algo in ["megatron_cp_algo", "hybrid_cp_algo"]:
            # The input layout of `ringattn_context_parallel` must be 'sbh'
            fa_layout = "sbh"

        self.core_attention_flash = FlashAttention(
            attention_dropout=dropout,
            fa_layout=fa_layout,
            softmax_scale=1 / math.sqrt(self.head_dim),
        )

        if self.cp_size > 1 and attention_type == AttnType.self_attn \
            and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]:

            if args.context_parallel_algo == "hybrid_cp_algo":
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            else:
                ulysses_group = mpu.get_context_parallel_group()

            if self.FPDT:
                self.core_attention_flash = FPDTFlashAttention(
                    ulysess_context_parallel_group=ulysses_group,
                    hidden_size=hidden_size,
                    head_dim=hidden_size // num_attention_heads,
                    chunk_number=self.FPDT_chunk_number,
                    with_offload=self.FPDT_with_offload
                )
            else:
                self.core_attention_flash = UlyssesContextAttention(self.core_attention_flash, ulysses_group)

        if self.cp_size > 1 and attention_type == AttnType.cross_attn:
            # In the case of cross attention, it is equivalent to performing the raw npu_fusion_attention for the slicing q
            self.core_attention_flash.context_parallel_algo = "ulysses_cp_algo"

        # Normalize
        if self.use_qk_norm:
            self.q_norm = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )
            self.k_norm = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )
            if isinstance(self.q_norm, nn.LayerNorm):
                for param in self.q_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)
            if isinstance(self.k_norm, nn.LayerNorm):
                for param in self.k_norm.parameters():
                    setattr(param, "sequence_parallel", self.sequence_parallel)

        self.has_img_input = has_img_input
        if self.has_img_input:
            args = get_args()
            config = core_transformer_config_from_args(args)

            self.k_img = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False,
            )
            self.v_img = tensor_parallel.ColumnParallelLinear(
                query_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                bias=proj_q_bias,
                gather_output=False,
            )
            self.k_norm_img = normalize(
                norm_type=norm_type,
                in_channels=hidden_size,
                eps=norm_eps,
                affine=norm_elementwise_affine,
                **kwargs,
            )

    def function_after_core_attention(
        self,
        core_attn_out,
        output_layout: str = "sbh"
    ):
        if self.FPDT:
            chunk_number = self.FPDT_chunk_number
            core_attn_out_chunks = torch.chunk(core_attn_out, chunks=chunk_number, dim=0)
            output = [None for _ in range(chunk_number)]
            for i in range(chunk_number):
                output[i], _ = self.proj_out(core_attn_out_chunks[i])
            output = torch.cat(output, dim=0)
        else:
            output, bias = self.proj_out(core_attn_out)
        # reshape
        output = change_tensor_layout(output, "sbh", output_layout)

        output = self.dropout(output)

        return output

    def get_query_key_value_tensors(self, hidden_states, key_value_states):
        """
        Derives `query` tensor from `hidden_states`, and `key`/`value` tensor
        from `hidden_states` or `key_value_states`.
        """
        if self.has_img_input:
            img_key_value_states, context_key_value_states = key_value_states
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            img_key = self.k_img(img_key_value_states)[0]
            img_value = self.v_img(img_key_value_states)[0]
            key = self.proj_k(context_key_value_states)[0]
            value = self.proj_v(context_key_value_states)[0]
        else:
            # Attention heads [s, b, h] --> [s, b, h]
            query = self.proj_q(hidden_states)[0]
            key = self.proj_k(key_value_states)[0]
            value = self.proj_v(key_value_states)[0]

        if self.use_qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
            if self.has_img_input:
                img_key = self.k_norm_img(img_key)

        # [s, b, h] --> [s, b, n, d]
        batch_size = query.shape[1]
        query = query.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )
        key = key.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )
        value = value.view(
            -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
        )

        if self.has_img_input:
            img_key = img_key.view(
                -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
            )
            img_value = img_value.view(
                -1, batch_size, self.num_attention_heads_per_partition, self.head_dim
            )
            key = [img_key, key]
            value = [img_value, value]
        return query, key, value

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None,
        mask: Optional[torch.Tensor] = None,
        input_layout: str = "sbh",
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ):
        if self.has_img_input:
            query, key, value = self.function_before_core_attention(
                query, key, input_layout, rotary_pos_emb
            )
            img_core_attn_out = self.core_attention_flash(query, key[0], value[0], mask)
            core_attn_out = self.core_attention_flash(query, key[1], value[1], mask)
            core_attn_out = img_core_attn_out + core_attn_out
            out = self.function_after_core_attention(core_attn_out, input_layout)
            return out
        else:
            return super().forward(query, key, mask, input_layout, rotary_pos_emb)


class Head(nn.Module):

    def __init__(
        self, dim: int, out_dim: int, patch_size: List[int], eps: float = 1e-6, fp32_calculate: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.fp32_calculate = fp32_calculate

        self.norm = FP32LayerNorm(dim, eps=eps, elementwise_affine=False) if fp32_calculate else nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, latents, times):
        if times.ndim == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=torch.float32 if self.fp32_calculate else times.dtype, device=times.device) + times.unsqueeze(2)
            ).chunk(2, dim=2)
            out = self.head((self.norm(latents.float() if self.fp32_calculate else latents) * (1 + scale.squeeze(2)) + shift.squeeze(2)).to(latents.dtype))
        else:
            shift, scale = (
                self.modulation.to(dtype=torch.float32 if self.fp32_calculate else times.dtype, device=times.device) + times
            ).chunk(2, dim=1)
            out = self.head((self.norm(latents.float() if self.fp32_calculate else latents) * (1 + scale) + shift).to(latents.dtype))
        return out


class MLPProj(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, flf_pos_emb=False, clip_token_len=257, fp32_calculate=False):
        super().__init__()

        self.proj = nn.Sequential(
            FP32LayerNorm(in_dim) if fp32_calculate else nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            FP32LayerNorm(out_dim) if fp32_calculate else nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: only used in "flf2v"
            self.emb_pos = nn.Parameter(torch.zeros(1, clip_token_len * 2, in_dim))

    def forward(self, image_emb):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_emb.shape
            image_emb = image_emb.view(-1, 2 * n, d)
            image_emb = image_emb + self.emb_pos
        return self.proj(image_emb)
