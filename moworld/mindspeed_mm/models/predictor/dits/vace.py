import copy
from contextlib import nullcontext, contextmanager
from typing import Optional, Tuple
import numpy as np
import torch
import torch_npu
import torch.nn as nn

from megatron.core import tensor_parallel, mpu
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.checkpoint import load_checkpoint
from mindspeed_mm.models.predictor.dits.wan_dit import WanDiTBlock, WanDiT, RoPE3DWan


class VaceWanAttentionBlock(nn.Module):
    def __init__(self, **kwargs):
        super(VaceWanAttentionBlock, self).__init__()
        self.layer_idx = kwargs['layer_idx']
        self.dim = kwargs['hidden_size']

        self.wan_dit_block = WanDiTBlock(**kwargs)

        if self.layer_idx == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(
            self,
            vace_context,
            latents,
            prompt,
            time_emb,
            rotary_pos_emb,
            recompute_skip_core_attention=False
    ):
        if self.layer_idx == 0:
            vace_context = self.before_proj(vace_context) + latents
            all_c = []
        else:
            all_c = list(torch.unbind(vace_context))
            vace_context = all_c.pop(-1)
        vace_context = self.wan_dit_block(vace_context, prompt, time_emb, rotary_pos_emb, recompute_skip_core_attention)
        c_skip = self.after_proj(vace_context)
        all_c += [c_skip, vace_context]
        vace_context = torch.stack(all_c)
        return vace_context


class VaceDit(nn.Module):
    def __init__(
            self,
            vace_layers: Tuple[int] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
            has_image_input: bool = False,
            patch_size: Tuple[int] = (1, 2, 2),
            text_len: int = 512,
            in_dim: int = 96,
            hidden_size: int = 1536,
            ffn_dim: int = 8960,
            freq_dim: int = 256,
            text_dim: int = 4096,
            img_dim: int = 1280,
            out_dim: int = 16,
            num_heads: int = 12,
            num_layers: int = 32,
            qk_norm: bool = True,
            qk_norm_type: str = 'rmsnorm',
            cross_attn_norm: bool = False,
            eps: float = 1e-6,
            max_seq_len: int = 1024,
            fa_layout: str = "bnsd",
            clip_token_len: int = 257,
            pre_process: bool = True,
            post_process: bool = True,
            global_layer_idx: Optional[Tuple] = None,
            atention_async_offload: bool = False,
            fp32_calculate: bool = False,
            **kwargs,
    ):
        super(VaceDit, self).__init__()
        self.vace_layers = vace_layers
        self.vace_to_wan = {vace_layer_num: wan_layer_num for wan_layer_num, vace_layer_num in enumerate(self.vace_layers)}
        self.has_image_input = has_image_input
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
        self.global_layer_idx = global_layer_idx
        self.head_dim = hidden_size // num_heads

        args = get_args()
        config = core_transformer_config_from_args(args)

        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_layers = {
            args.recompute_num_layers
            if args.recompute_num_layers is not None
            else num_layers
        }

        self.recompute_skip_core_attention = args.recompute_skip_core_attention
        self.recompute_num_layers_skip_core_attention = args.recompute_num_layers_skip_core_attention
        self.attention_async_offload = atention_async_offload
        self.fp32_calculate = fp32_calculate

        self.h2d_stream = torch_npu.npu.Stream() if atention_async_offload else None
        self.d2h_stream = torch_npu.npu.Stream() if atention_async_offload else None

        # rope
        self.rope = RoPE3DWan(head_dim=self.head_dim, max_seq_len=self.max_seq_len)

        # vace blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(
                model_type="t2v",
                hidden_size=self.hidden_size,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                qk_norm=self.qk_norm,
                qk_norm_type=self.qk_norm_type,
                cross_attn_norm=self.cross_attn_norm,
                eps=self.eps,
                rope=self.rope,
                fa_layout=self.fa_layout,
                clip_token_len=self.clip_token_len,
                atention_async_offload=self.attention_async_offload,
                layer_idx=index,
                num_layers=self.num_layers,
                fp32_calculate=self.fp32_calculate,
                h2d_stream=self.h2d_stream,
                d2h_stream=self.d2h_stream
            )
            for index in range(len(self.vace_layers))
        ])

        # vace patch embeddings
        self.vace_patch_embedding = torch.nn.Conv3d(
            self.in_dim,
            self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

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

    def forward(
            self, embs, vace_context, prompt_emb, time_emb, rotary_pos_emb
    ):
        # Embed each context sequence and add batch dimension
        vace_context_embed = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        # Flatten and transpose dimensions for proper tensor shape
        vace_context_reshape = [u.flatten(2).transpose(1, 2) for u in vace_context_embed]
        # Pad sequences to equal size and batch them together
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, embs.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in vace_context_reshape
        ])

        for block in self.vace_blocks:
            c = block(c, embs, prompt_emb, time_emb, rotary_pos_emb)
        hints = torch.unbind(c)[:-1]
        return hints


class VACEModel(MultiModalModule):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(config=None)
        self.vace_config = kwargs.pop('vace_dit')
        self.wan_config = kwargs
        # Model initialization is performed on a meta device to avoid random initialization.
        with self.meta_init():
            self.wan_dit = WanDiT(**self.wan_config)
            self.vace_dit = VaceDit(**self.vace_config)

        # Freeze the Wan module.
        self.freeze()

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        return self.wan_dit.dtype

    @property
    def device(self) -> torch.device:
        """The device of the module (assuming that all the module parameters are in the same device)."""
        return self.wan_dit.device

    def post_init(self):
        if "vace_pretrained" in self.vace_config and self.vace_config["vace_pretrained"] is not None:
            load_checkpoint(self.vace_dit, self.vace_config['vace_pretrained'], assign=True)
        elif self.vace_dit.device.type == "meta":
            self.vace_dit = VaceDit(**self.vace_config)
            self.vace_patch_embedding_replace(self.wan_dit, self.vace_dit)

    def forward(
            self,
            latents: torch.Tensor = None,
            timestep: torch.Tensor = None,
            prompt: torch.Tensor = None,
            prompt_mask: torch.Tensor = None,
            vace_context=None,
            vace_scale=1.0,
            use_unified_sequence_parallel: bool = False,
            **kwargs
    ):
        timestep = timestep.to(latents[0].device)
        # time embeddings
        times = self.wan_dit.time_embedding(
            self.wan_dit.sinusoidal_embedding_1d(self.wan_dit.freq_dim, timestep)
        )
        time_emb = self.wan_dit.time_projection(times).unflatten(1, (6, self.wan_dit.hidden_size))

        bs = prompt.size(0)
        prompt = prompt.view(bs, -1, prompt.size(-1))
        if prompt_mask is not None:
            seq_lens = prompt_mask.view(bs, -1).sum(dim=-1)
            seq_lens = seq_lens.to(torch.int64)
            for i, seq_lens in enumerate(seq_lens):
                prompt[i, seq_lens:] = 0
        prompt_emb = self.wan_dit.text_embedding(prompt)

        x = latents
        # patch embedding
        patch_emb = self.wan_dit.patch_embedding(x.to(time_emb.dtype))

        embs, grid_sizes = self.wan_dit.patchify(patch_emb)

        # rotary positional embeddings
        batch_size, frames, height, width = (
            embs.shape[0],
            grid_sizes[0],
            grid_sizes[1],
            grid_sizes[2],
        )
        rotary_pos_emb = self.wan_dit.rope(batch_size, frames, height, width)
        vace_hints = self.vace_dit(embs, vace_context, prompt_emb, time_emb, rotary_pos_emb)

        for block_id, block in enumerate(self.wan_dit.blocks):
            embs = block(embs, prompt_emb, time_emb, rotary_pos_emb)
            if vace_context is not None and block_id in self.vace_dit.vace_to_wan:
                current_vace_hint = vace_hints[self.vace_dit.vace_to_wan[block_id]]
                embs = embs + current_vace_hint * vace_scale

        embs_out = self.wan_dit.head(embs, times)
        out = self.wan_dit.unpatchify(embs_out, frames, height, width)
        rtn = (out, prompt, prompt_emb, time_emb, times, prompt_mask)

        return rtn

    def vace_patch_embedding_replace(self, wan_dit: WanDiT, vace_dit: VaceDit):
        vace_dit.vace_patch_embedding.bias = copy.deepcopy(wan_dit.patch_embedding.bias)
        weight_shape = list(vace_dit.vace_patch_embedding.weight.shape)
        weight_shape[-1] -= wan_dit.patch_embedding.weight.shape[1] * 2
        vace_dit.vace_patch_embedding.weight = torch.nn.Parameter(
            torch.cat((copy.deepcopy(wan_dit.patch_embedding.weight),
                       copy.deepcopy(wan_dit.patch_embedding.weight),
                       torch.zeros(weight_shape, device=wan_dit.patch_embedding.weight.device, dtype=self.dtype)),
                      dim=1)
        )

    @contextmanager
    def meta_init(self, device=torch.device("meta"), include_buffers: bool = False):
        old_register_parameter = torch.nn.Module.register_parameter
        if include_buffers:
            old_register_buffer = torch.nn.Module.register_buffer

        def register_empty_parameter(module, name, param):
            old_register_parameter(module, name, param)
            if param is not None:
                param_cls = type(module._parameters[name])
                kwargs = module._parameters[name].__dict__
                kwargs["requires_grad"] = param.requires_grad
                module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

        def register_empty_buffer(module, name, buffer, persistent=True):
            old_register_buffer(module, name, buffer, persistent=persistent)
            if buffer is not None:
                module._buffers[name] = module._buffers[name].to(device)

        def patch_tensor_constructor(fn):
            def wrapper(*args, **kwargs):
                kwargs['device'] = device
                return fn(*args, **kwargs)

            return wrapper

        if include_buffers:
            tensor_constructors_to_patch = {
                torch_function_name: getattr(torch, torch_function_name)
                for torch_function_name in ["empty", "zeros", "ones", "full"]
            }
        else:
            tensor_constructors_to_patch = {}

        try:
            torch.nn.Module.register_parameter = register_empty_parameter
            if include_buffers:
                torch.nn.Module.register_buffer = register_empty_buffer
            for torch_function_name in tensor_constructors_to_patch.keys():
                setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
            yield
        finally:
            torch.nn.Module.register_parameter = old_register_parameter
            if include_buffers:
                torch.nn.Module.register_buffer = old_register_buffer
            for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
                setattr(torch, torch_function_name, old_torch_function)

    def freeze(self):
        self.wan_dit.eval()
        self.wan_dit.requires_grad_(False)
        self.vace_dit.train()
        self.vace_dit.requires_grad_(True)
