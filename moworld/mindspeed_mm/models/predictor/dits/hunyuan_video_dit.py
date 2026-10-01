import math
from typing import List, Optional, Union, Tuple
from contextlib import nullcontext
from einops import rearrange
import torch
from torch import nn
import torch_npu

from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import (
    split_forward_gather_backward,
    gather_forward_split_backward,
    all_to_all,
    cal_split_sizes
)
from mindspeed.core.context_parallel.model_parallel_utils import (
    get_context_parallel_group_for_hybrid_ulysses,
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ulysses_world_size,
    get_context_parallel_for_hybrid_ring_global_ranks,
    get_context_parallel_for_hybrid_ring_rank
)
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings import (
    PatchEmbed3D,
    TimeStepEmbedding,
    TextProjection,
    SingleTokenRefiner,
    get_nd_rotary_pos_embed
)
from mindspeed_mm.models.common.blocks import MLP, ModulateDiT
from mindspeed_mm.models.common.activations import get_activation_layer
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.attention import do_npu_fusion_attention


def _get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="npu")

    for i in range(batch_size):
        seq_len = text_len[i] + img_len
        seq_len1 = i * max_len + seq_len
        seq_len2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = seq_len1
        cu_seqlens[2 * i + 2] = seq_len2

    return cu_seqlens


def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None, shapes=None):
    args = get_args()
    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
        if args.use_cp_send_recv_overlap else None
    cp_para['pse'] = pse
    cp_para['pse_type'] = pse_type

    cp_para['megatron_cp_in_bnsd'] = args.megatron_cp_in_bnsd

    output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=shapes)

    return output


class HunyuanVideoDiT(MultiModalModule):
    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            in_channels: int = 4,  # should be VAE.latent_channels,
            out_channels: Optional[int] = None,
            num_heads: int = 24,
            head_dim: int = 128,
            mlp_width_ratio: int = 4,
            mlp_act_type: str = "gelu_tanh",
            mm_double_blocks_depth: int = 20,
            mm_single_blocks_depth: int = 40,
            double_stream_full_recompute_layers: Optional[int] = None,
            single_stream_full_recompute_layers: Optional[int] = None,
            rope_dim_list: Tuple[int] = (16, 56, 56),
            qkv_bias: bool = True,
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            guidance_embed: bool = False,
            text_projection: str = "single_refiner",
            text_states_dim: Tuple[int] = (4096, 768),
            timestep_hidden_dim: int = 256,
            use_attention_mask: bool = True,
            rope_theta: int = 256,
            embeded_guidance_scale: float = 6.016,
            attention_async_offload: bool = False,
            i2v_condition_type: str = None,
            **kwargs
        ):
        super().__init__(config=None)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.text_states_dim = text_states_dim

        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.hidden_size = num_heads * head_dim
        self.heads_num = num_heads
        self.embeded_guidance_scale = embeded_guidance_scale

        self.attention_async_offload = attention_async_offload
        self.h2d_stream = torch_npu.npu.Stream() if attention_async_offload else None
        self.d2h_stream = torch_npu.npu.Stream() if attention_async_offload else None

        args = get_args()
        config = core_transformer_config_from_args(args)
        self.i2v_condition_type = i2v_condition_type
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.double_stream_full_recompute_layers = double_stream_full_recompute_layers if double_stream_full_recompute_layers is not None else mm_double_blocks_depth
        self.single_stream_full_recompute_layers = single_stream_full_recompute_layers if single_stream_full_recompute_layers is not None else mm_single_blocks_depth
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in HunyuanvideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel

        # context parallel setting
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo", "hybrid_cp_algo", "megatron_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")

        if sum(rope_dim_list) != head_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {head_dim}"
            )

        # time modulation
        self.time_in = TimeStepEmbedding(timestep_hidden_dim, time_embed_dim=self.hidden_size)

        # txt modulation
        self.vector_in = MLP(
            in_channels=self.text_states_dim[1],
            hidden_channels=self.hidden_size,
            out_features=self.hidden_size,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=False
        )

        self.img_in = PatchEmbed3D(
            self.patch_size, self.in_channels, self.hidden_size
        )

        # guidance modulation
        self.guidance_in = (
            TimeStepEmbedding(timestep_hidden_dim, self.hidden_size)
            if guidance_embed
            else None
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim[0],
                self.hidden_size,
                nn.SiLU,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                in_channels=self.text_states_dim[0],
                hidden_size=self.hidden_size,
                time_embed_dim=timestep_hidden_dim,
                heads_num=num_heads,
                depth=2
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                layer_index=layer_index,
                num_layers=mm_double_blocks_depth + mm_single_blocks_depth,
                attention_async_offload=self.attention_async_offload,
                h2d_stream=self.h2d_stream,
                d2h_stream=self.d2h_stream,
                condition_type=self.i2v_condition_type
            )
            for layer_index in range(mm_double_blocks_depth)
        ])

        # single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                layer_index=layer_index + mm_double_blocks_depth,
                num_layers=mm_double_blocks_depth + mm_single_blocks_depth,
                attention_async_offload=self.attention_async_offload,
                h2d_stream=self.h2d_stream,
                d2h_stream=self.d2h_stream,
                condition_type=self.i2v_condition_type
            )
            for layer_index in range(mm_single_blocks_depth)
        ])

        if self.enable_tensor_parallel:
            config.sequence_parallel = False
            self.adaLN_modulation = nn.Sequential(
                get_activation_layer("silu")(),
                tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    2 * self.hidden_size,
                    bias=True,
                    config=config,
                    init_method=config.init_method,
                    gather_output=False
                )
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.adaLN_modulation = nn.Sequential(
            get_activation_layer("silu")(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        self.norm_final = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.norm_final.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        if self.enable_tensor_parallel:
            self.proj_out = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                math.prod(patch_size) * out_channels,
                bias=True,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
        else:
            self.proj_out = nn.Linear(self.hidden_size, math.prod(patch_size) * out_channels, bias=True)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def unpatchify(self, x, t, h, w) -> torch.Tensor:
        """
        input:
            x: (B, t*h*w, d)
        return:
            x: (B, c, T, H, W)
        """
        b = x.shape[0]
        c = self.unpatchify_channels
        patch_t, patch_h, patch_w = self.patch_size
        x = x.reshape(b, t, h, w, c, patch_t, patch_h, patch_w)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        output = x.reshape(b, c, t * patch_t, h * patch_h, w * patch_w)
        return output

    def _get_block(
            self,
            dit_type: str,
            layer_number: int
    ):
        if dit_type == "double_stream":
            return self.double_blocks[layer_number]
        elif dit_type == "single_stream":
            return self.single_blocks[layer_number]
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")

    def _checkpointed_forward(
            self,
            dit_type: str,
            img_and_txt: tuple,
            *args
    ):
        "Forward method with activation checkpointing."
        if dit_type == "double_stream":
            num_layers = len(self.double_blocks)
            recompute_layers = self.double_stream_full_recompute_layers
        elif dit_type == "single_stream":
            num_layers = len(self.single_blocks)
            recompute_layers = self.single_stream_full_recompute_layers
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")

        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = self._get_block(dit_type, index)
                    x_ = layer(*args)
                return x_
            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < num_layers:
                img_and_txt = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.distribute_saved_activations,
                    *img_and_txt,
                    *args
                )
                _layer_num += recompute_layers

        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(num_layers):
                if _layer_num < recompute_layers:
                    img_and_txt = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.distribute_saved_activations,
                        *img_and_txt,
                        *args
                    )
                else:
                    block = self._get_block(dit_type, _layer_num)
                    img_and_txt = block(*img_and_txt, *args, block_full_attention=False)
        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")

        return img_and_txt

    def forward(
            self,
            x: torch.Tensor,
            timestep: torch.Tensor,
            prompt: List[torch.Tensor],
            prompt_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
            guidance: torch.Tensor = None,
            **kwargs
    ):
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        x = x.to(self.dtype)

        if isinstance(prompt_mask, list):
            prompt_mask = prompt_mask[0]
        prompt_mask = prompt_mask.to(x.device)
        prompt[0] = prompt[0].view(-1, prompt[0].shape[-2], prompt[0].shape[-1]).to(self.dtype)  # B*N, seq_len, Dim
        prompt[1] = prompt[1].to(self.dtype)  # B, N, Dim
        prompt_mask = prompt_mask.view(-1, prompt_mask.shape[-1])  # B*N, seqlen

        # Prepare modulation vectors
        vec = self.time_in(timestep)

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(timestep)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
            frist_frame_token_num = torch.tensor(frist_frame_token_num)
        else:
            token_replace_vec = None
            frist_frame_token_num = None

        # text modulation
        vec_2 = self.vector_in(prompt[1])
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                guidance = torch.tensor(
                    [self.embeded_guidance_scale] * x.shape[0],
                    dtype=torch.float32,
                ).to(vec.device).to(vec.dtype) * 1000.0
            vec = vec + self.guidance_in(guidance)

        img = self.img_in(x)

        if self.text_projection == "linear":
            txt = self.txt_in(prompt[0])
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(prompt[0], timestep, prompt_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = _get_cu_seqlens(prompt_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        rope_sizes = list(x.shape)[-3:]
        rope_sizes = [rope_sizes[i] // self.patch_size[i] for i in range(3)]
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            self.rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            theta_rescale_factor=1
        )
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2).to(device=vec.device, dtype=vec.dtype)  # b s n d
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2).to(device=vec.device, dtype=vec.dtype)  # b s n d

        if bs == 1:
            txt = txt[:, :prompt_mask.sum()]
            txt_seq_len = txt.shape[1]

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # cp split
        if self.context_parallel_algo is not None:
            img = split_forward_gather_backward(
                img,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            )  # b s n d
            freqs_cos = split_forward_gather_backward(
                freqs_cos,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            )
            freqs_sin = split_forward_gather_backward(
                freqs_sin,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            )
            img_seq_len = img_seq_len // mpu.get_context_parallel_world_size()

        img_seq_len = torch.tensor(img_seq_len)
        max_seqlen_q = torch.tensor(max_seqlen_q)
        max_seqlen_kv = torch.tensor(max_seqlen_kv)

        if self.sequence_parallel:
            # b s h -> s b h
            img = img.transpose(0, 1).contiguous()
            # split img_seq for tp-sp
            img = tensor_parallel.scatter_to_sequence_parallel_region(img)

        # --------------------------- Pass through DiT blocks --------------------------
        with rng_context:
            if self.recompute_granularity == "full":
                # double_stream
                img, txt = self._checkpointed_forward(
                    "double_stream",
                    (img, txt),
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cos,
                    freqs_sin,
                    token_replace_vec,
                    frist_frame_token_num
                )

                if self.sequence_parallel:
                    txt = txt.transpose(0, 1).contiguous()
                    x = torch.cat([img, txt], dim=0)
                else:
                    x = torch.cat([img, txt], dim=1)

                # single_stream
                x = self._checkpointed_forward(
                    "single_stream",
                    (x, ),
                    vec,
                    img_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cos,
                    freqs_sin,
                    token_replace_vec,
                    frist_frame_token_num
                )[0]

            else:
                for _, block in enumerate(self.double_blocks):
                    img, txt = block(
                        img=img,
                        txt=txt,
                        vec=vec,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        token_replace_vec=token_replace_vec,
                        frist_frame_token_num=frist_frame_token_num
                    )
                if self.sequence_parallel:
                    txt = txt.transpose(0, 1).contiguous()
                    x = torch.cat([img, txt], dim=0)
                else:
                    x = torch.cat([img, txt], dim=1)

                for _, block in enumerate(self.single_blocks):
                    x = block(
                        x=x,
                        vec=vec,
                        img_len=img_seq_len,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        token_replace_vec=token_replace_vec,
                        frist_frame_token_num=frist_frame_token_num
                    )[0]

        if self.sequence_parallel:
            img = x[:img_seq_len // mpu.get_tensor_model_parallel_world_size()]
        else:
            img = x[:, :img_seq_len]

        # --------------------- Final layer ------------
        if self.enable_tensor_parallel:
            shift_scale = self.adaLN_modulation(vec)[0]
            if self.sequence_parallel:
                shift_scale = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(
                    shift_scale
                )
            else:
                shift_scale = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(
                    shift_scale
                )
            shift, scale = shift_scale.chunk(2, dim=-1)
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)

        x = self.norm_final(img) * (1 + scale) + shift

        if self.enable_tensor_parallel:
            x = self.proj_out(x)[0]
            if self.sequence_parallel:
                x = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(x)
            else:
                x = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(x)
        else:
            x = self.proj_out(x)

        if self.sequence_parallel:
            x = x.transpose(0, 1).contiguous() # s b h -> b s h

        if self.context_parallel_algo is not None:
            x = gather_forward_split_backward(
                x,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="up"
            )
        output = self.unpatchify(x, tt, th, tw)
        return output


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with separate modulation for text and image/video
    """
    def __init__(
            self,
            hidden_size: int,
            heads_num: int,
            mlp_width_ratio: float,
            mlp_act_type: str = "gelu_tanh",
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            qkv_bias: bool = False,
            layer_index: int = 0,
            num_layers: int = 60,
            attention_async_offload: bool = False,
            h2d_stream: Optional[torch_npu.npu.Stream] = None,
            d2h_stream: Optional[torch_npu.npu.Stream] = None,
            condition_type: str = None
    ):
        super().__init__()

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.layer_index = layer_index
        self.num_layers = num_layers
        self.attention_async_offload = attention_async_offload
        self.h2d_stream = h2d_stream
        self.d2h_stream = d2h_stream
        self.i2v_condition_type = condition_type

        # context parallel setting
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo", "hybrid_cp_algo", "megatron_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")
        self.distribute_saved_activations = args.distribute_saved_activations
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel
        self.num_attention_heads_per_partition = self.heads_num // mpu.get_tensor_model_parallel_world_size()

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.img_norm1.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        if self.enable_tensor_parallel:
            self.img_attn_qkv = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
        else:
            self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.img_attn_q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.img_attn_k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        if qk_norm:
            for param in self.img_attn_q_norm.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)
            for param in self.img_attn_k_norm.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.txt_norm1.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        if self.enable_tensor_parallel:
            config.sequence_parallel = False  # disable txt_seq sp
            self.txt_attn_qkv = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.txt_attn_q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.txt_attn_k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        # disable txt_seq sp
        for param in self.txt_attn_q_norm.parameters():
            setattr(param, "sequence_parallel", False)
        for param in self.txt_attn_k_norm.parameters():
            setattr(param, "sequence_parallel", False)

        if self.enable_tensor_parallel:
            self.img_attn_proj = tensor_parallel.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False
            )
        else:
            self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.img_norm2.parameters():
            setattr(param, "sequence_parallel", True)

        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=True
        )

        if self.enable_tensor_parallel:
            config.sequence_parallel = False  # disable txt_seq sp
            self.txt_attn_proj = tensor_parallel.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.txt_norm2.parameters():
            setattr(param, "sequence_parallel", False)

        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=False
        )

    def forward(
            self,
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[torch.Tensor] = None,
            max_seqlen_kv: Optional[torch.Tensor] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            token_replace_vec: torch.Tensor = None,
            frist_frame_token_num: int = None,
            block_full_attention: bool = True
    ):
        if block_full_attention:
            (
                img_q, img_k, img_v,
                txt_q, txt_k, txt_v,
                img_mod1_gate, txt_mod1_gate, img_mod2_gate, txt_mod2_gate,
                img_mod2_scale, txt_mod2_scale, img_mod2_shift, txt_mod2_shift,
                tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
                tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift,
            ) = self._before_attention(img, txt, vec, freqs_cos, freqs_sin, token_replace_vec, frist_frame_token_num)
        else:
            (
                img_q, img_k, img_v,
                txt_q, txt_k, txt_v,
                img_mod1_gate, txt_mod1_gate, img_mod2_gate, txt_mod2_gate,
                img_mod2_scale, txt_mod2_scale, img_mod2_shift, txt_mod2_shift,
                tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
                tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift,
            ) = tensor_parallel.checkpoint(
                self._before_attention,
                self.distribute_saved_activations,
                img,
                txt,
                vec,
                freqs_cos,
                freqs_sin,
                token_replace_vec,
                frist_frame_token_num
            )

        # Run actual attention
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            heads_num=self.num_attention_heads_per_partition,
            head_dim=self.head_dim,
            attn_mask=None,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
            context_parallel_algo=self.context_parallel_algo,
            h2d_stream=self.h2d_stream,
            d2h_stream=self.d2h_stream,
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            attention_async_offload=(not block_full_attention) and self.attention_async_offload
        )

        if block_full_attention:
            img, txt = self._after_attention(
                attn, img, txt,
                img_mod1_gate, txt_mod1_gate,
                img_mod2_gate, txt_mod2_gate,
                img_mod2_scale, txt_mod2_scale,
                img_mod2_shift, txt_mod2_shift,
                frist_frame_token_num,
                tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
                tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift,
            )
        else:
            img, txt = tensor_parallel.checkpoint(
                self._after_attention,
                self.distribute_saved_activations,
                attn, img, txt,
                img_mod1_gate, txt_mod1_gate,
                img_mod2_gate, txt_mod2_gate,
                img_mod2_scale, txt_mod2_scale,
                img_mod2_shift, txt_mod2_shift,
                frist_frame_token_num,
                tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
                tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift,
            )

        return img, txt

    def _before_attention(
            self,
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            token_replace_vec: torch.Tensor = None,
            frist_frame_token_num: int = None
    ):
        if self.i2v_condition_type == "token_replace":
            img_mod1 = self.img_mod(vec)
            token_replace_img_mod1 = self.img_mod(token_replace_vec)
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate
            ) = img_mod1.chunk(6, dim=-1)
            (
                tr_img_mod1_shift,
                tr_img_mod1_scale,
                tr_img_mod1_gate,
                tr_img_mod2_shift,
                tr_img_mod2_scale,
                tr_img_mod2_gate
            ) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            img_mod1 = self.img_mod(vec)
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate
            ) = img_mod1.chunk(6, dim=-1)
            tr_img_mod1_shift = None
            tr_img_mod1_scale = None
            tr_img_mod1_gate = None
            tr_img_mod2_shift = None
            tr_img_mod2_scale = None
            tr_img_mod2_gate = None

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention,
        img_modulated = self.img_norm1(img)
        if self.i2v_condition_type == "token_replace":
            img_zero = img_modulated[:, :frist_frame_token_num] * (1 + tr_img_mod1_scale) + tr_img_mod1_shift
            img_orig = img_modulated[:, frist_frame_token_num:] * (1 + img_mod1_scale) + img_mod1_shift
            img_modulated = torch.cat([img_zero, img_orig], dim=1)
        else:
            img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift

        if self.enable_tensor_parallel:
            img_qkv = self.img_attn_qkv(img_modulated)[0]
            if self.sequence_parallel:
                img_qkv = img_qkv.transpose(0, 1).contiguous()  # s b h -> b s h
        else:
            img_qkv = self.img_attn_qkv(img_modulated)

        # b s h
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_attention_heads_per_partition)
        img_seq_len = img_q.shape[1]

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)

        # Apply RoPE if needed
        if freqs_cos is not None and freqs_sin is not None:
            img_q = npu_rotary_position_embedding(img_q, freqs_cos, freqs_sin, mode=1)
            img_k = npu_rotary_position_embedding(img_k, freqs_cos, freqs_sin, mode=1)

        # Prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift

        if self.enable_tensor_parallel:
            txt_qkv = self.txt_attn_qkv(txt_modulated)[0]
        else:
            txt_qkv = self.txt_attn_qkv(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_attention_heads_per_partition)

        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        return (
            img_q, img_k, img_v,
            txt_q, txt_k, txt_v,
            img_mod1_gate, txt_mod1_gate, img_mod2_gate, txt_mod2_gate,
            img_mod2_scale, txt_mod2_scale, img_mod2_shift, txt_mod2_shift,
            tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
            tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift
        )

    def _after_attention(
            self,
            attn, img, txt,
            img_mod1_gate, txt_mod1_gate,
            img_mod2_gate, txt_mod2_gate,
            img_mod2_scale, txt_mod2_scale,
            img_mod2_shift, txt_mod2_shift,
            frist_frame_token_num,
            tr_img_mod1_gate, tr_img_mod1_scale, tr_img_mod1_shift,
            tr_img_mod2_gate, tr_img_mod2_scale, tr_img_mod2_shift
    ):
        if self.sequence_parallel:
            img_seq_len = img.shape[0] * mpu.get_tensor_model_parallel_world_size()
        else:
            img_seq_len = img.shape[1]
        img_attn, txt_attn = attn[:, :img_seq_len], attn[:, img_seq_len:]

        # Calculate the img blocks
        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                img_attn = img_attn.transpose(0, 1).contiguous()  # b s h -> s b h
            if self.i2v_condition_type == "token_replace":
                x = self.img_attn_proj(img_attn)[0]
                img_zero = x[:, :frist_frame_token_num] * tr_img_mod1_gate
                img_orig = x[:, frist_frame_token_num:] * img_mod1_gate
                img = img + torch.concat((img_zero, img_orig), dim=1)
            else:
                img = img + self.img_attn_proj(img_attn)[0] * img_mod1_gate
        else:
            if self.i2v_condition_type == "token_replace":
                x = self.img_attn_proj(img_attn)
                img_zero = x[:, :frist_frame_token_num] * tr_img_mod1_gate
                img_orig = x[:, frist_frame_token_num:] * img_mod1_gate
                img = img + torch.concat((img_zero, img_orig), dim=1)
            else:
                img = img + self.img_attn_proj(img_attn) * img_mod1_gate

        if self.i2v_condition_type == "token_replace":
            x = self.img_norm2(img)
            img_zero = x[:, :frist_frame_token_num] * (1 + tr_img_mod2_scale) + tr_img_mod2_shift
            img_orig = x[:, frist_frame_token_num:] * (1 + img_mod2_scale) + img_mod2_shift
            x = self.img_mlp(torch.concat((img_zero, img_orig), dim=1))
            img_zero = x[:, :frist_frame_token_num] * tr_img_mod2_gate
            img_orig = x[:, frist_frame_token_num:] * img_mod2_gate
            img = img + torch.concat((img_zero, img_orig), dim=1)
        else:
            img = img + \
                self.img_mlp(
                    self.img_norm2(img) * (1 + img_mod2_scale) + img_mod2_shift
                ) * img_mod2_gate

        # Calculate the txt blocks
        if self.enable_tensor_parallel:
            txt = txt + self.txt_attn_proj(txt_attn)[0] * txt_mod1_gate
        else:
            txt = txt + self.txt_attn_proj(txt_attn) * txt_mod1_gate
        txt = txt + \
            self.txt_mlp(
                self.txt_norm2(txt) * (1 + txt_mod2_scale) + txt_mod2_shift
            ) * txt_mod2_gate

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers.
    """
    def __init__(
            self,
            hidden_size: int,
            heads_num: int,
            mlp_width_ratio: int = 4,
            mlp_act_type: str = "gelu_tanh",
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            qk_scale: float = None,
            layer_index: int = 0,
            num_layers: int = 60,
            attention_async_offload: bool = False,
            h2d_stream: Optional[torch_npu.npu.Stream] = None,
            d2h_stream: Optional[torch_npu.npu.Stream] = None,
            condition_type: str = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.i2v_condition_type = condition_type

        self.layer_index = layer_index
        self.num_layers = num_layers
        self.attention_async_offload = attention_async_offload
        self.h2d_stream = h2d_stream
        self.d2h_stream = d2h_stream

        # context parallel setting
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo", "hybrid_cp_algo", "megatron_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")
        self.distribute_saved_activations = args.distribute_saved_activations
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel
        self.num_attention_heads_per_partition = self.heads_num // mpu.get_tensor_model_parallel_world_size()

        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if self.enable_tensor_parallel:
            self.linear1_qkv = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=True
            )
        else:
            self.linear1_qkv = nn.Linear(hidden_size, hidden_size * 3)

        self.q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        if self.enable_tensor_parallel:
            self.linear1_mlp = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                mlp_hidden_dim,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=True
            )
            self.linear2 = tensor_parallel.RowParallelLinear(
                hidden_size + mlp_hidden_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False,
                bias=True
            )
        else:
            self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim)
            self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size)

        for param in self.q_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        for param in self.k_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        for param in self.pre_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
            self,
            x: torch.Tensor,
            vec: torch.Tensor,
            img_len: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[torch.Tensor] = None,
            max_seqlen_kv: Optional[torch.Tensor] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            token_replace_vec: torch.Tensor = None,
            frist_frame_token_num: int = None,
            block_full_attention: bool = True
    ):
        if block_full_attention:
            mod_gate, x_mod, img_q, img_k, img_v, txt_q, txt_k, txt_v, tr_mod_gate, tr_mod_scale, tr_mod_shift = self._before_attention(
                x,
                vec,
                img_len,
                freqs_cos,
                freqs_sin,
                cu_seqlens_q,
                token_replace_vec,
                frist_frame_token_num
            )
        else:
            mod_gate, x_mod, img_q, img_k, img_v, txt_q, txt_k, txt_v, tr_mod_gate, tr_mod_scale, tr_mod_shift = tensor_parallel.checkpoint(
                self._before_attention,
                self.distribute_saved_activations,
                x,
                vec,
                img_len,
                freqs_cos,
                freqs_sin,
                cu_seqlens_q,
                token_replace_vec,
                frist_frame_token_num
            )

        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            heads_num=self.num_attention_heads_per_partition,
            head_dim=self.head_dim,
            attn_mask=None,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
            context_parallel_algo=self.context_parallel_algo,
            h2d_stream=self.h2d_stream,
            d2h_stream=self.d2h_stream,
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            attention_async_offload=(not block_full_attention) and self.attention_async_offload
        )

        if block_full_attention:
            output = self._after_attention(attn, x, x_mod, img_len, mod_gate, frist_frame_token_num, tr_mod_gate, tr_mod_scale, tr_mod_shift)
        else:
            output = tensor_parallel.checkpoint(
                self._after_attention,
                self.distribute_saved_activations,
                attn,
                x,
                x_mod,
                img_len,
                mod_gate,
                frist_frame_token_num,
                tr_mod_gate,
                tr_mod_scale,
                tr_mod_shift
            )

        return output

    def _before_attention(
            self,
            x,
            vec,
            img_len,
            freqs_cos,
            freqs_sin,
            cu_seqlens_q,
            token_replace_vec,
            frist_frame_token_num
    ):
        if self.i2v_condition_type == "token_replace":
            mod = self.modulation(vec)
            tr_mod = self.modulation(token_replace_vec)
            mod_shift, mod_scale, mod_gate = mod.chunk(3, dim=-1)
            tr_mod_shift, tr_mod_scale, tr_mod_gate = tr_mod.chunk(3, dim=-1)
        else:
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
            tr_mod_shift, tr_mod_scale, tr_mod_gate = None, None, None

        if self.i2v_condition_type == "token_replace":
            x_norm = self.pre_norm(x)
            x_zero = x_norm[:, :frist_frame_token_num] * (1 + tr_mod_scale) + tr_mod_shift
            x_orig = x_norm[:, frist_frame_token_num:] * (1 + mod_scale) + mod_shift
            x_mod = torch.cat([x_zero, x_orig], dim=1)
        else:
            x_mod = self.pre_norm(x) * (1 + mod_scale) + mod_shift

        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                img_qkv = self.linear1_qkv(x_mod[:img_len // self.tp_size])[0]
                self.linear1_qkv.sequence_parallel = False
                txt_qkv = self.linear1_qkv(x_mod[img_len // self.tp_size:])[0]
                self.linear1_qkv.sequence_parallel = True
                qkv = torch.cat([img_qkv, txt_qkv], dim=0)
                qkv = qkv.transpose(0, 1)  # s b h -> b s h
            else:
                qkv = self.linear1_qkv(x_mod)[0]
        else:
            qkv = self.linear1_qkv(x_mod)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_attention_heads_per_partition)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cos is not None and freqs_sin is not None:
            img_q, txt_q = q[:, :img_len, :, :], q[:, img_len:, :, :]  # b s n d
            img_k, txt_k = k[:, :img_len, :, :], k[:, img_len:, :, :]

            img_q = npu_rotary_position_embedding(img_q, freqs_cos, freqs_sin, mode=1)
            img_k = npu_rotary_position_embedding(img_k, freqs_cos, freqs_sin, mode=1)

        img_v = v[:, :img_len, :, :]
        txt_v = v[:, img_len:, :, :]

        return mod_gate, x_mod, img_q, img_k, img_v, txt_q, txt_k, txt_v, tr_mod_gate, tr_mod_scale, tr_mod_shift

    def _after_attention(
            self,
            attn: torch.Tensor,
            x: torch.Tensor,
            x_mod: torch.Tensor,
            img_len: torch.Tensor,
            mod_gate: torch.Tensor,
            frist_frame_token_num: int,
            tr_mod_gate: torch.Tensor,
            tr_mod_scale: torch.Tensor,
            tr_mod_shift: torch.Tensor
    ):
        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                attn = attn.transpose(0, 1).contiguous()  # b s h -> s b h
                img_attn = attn[:img_len]
                txt_attn = attn[img_len:]

                img_mod = x_mod[:img_len // mpu.get_tensor_model_parallel_world_size()]  # s b h
                txt_mod = x_mod[img_len // mpu.get_tensor_model_parallel_world_size():]

                img_mlp = self.linear1_mlp(img_mod)[0]
                self.linear1_mlp.sequence_parallel = False
                txt_mlp = self.linear1_mlp(txt_mod)[0]
                self.linear1_mlp.sequence_parallel = True

                img_output = self.linear2(torch.cat((img_attn, self.mlp_act(img_mlp)), 2))[0]

                # disable txt_seq sequence_parallel
                self.linear2.sequence_parallel = False
                txt_output = self.linear2(torch.cat((txt_attn, self.mlp_act(txt_mlp)), 2))[0]
                self.linear2.sequence_parallel = True

                output = torch.cat([img_output, txt_output], dim=0)
            else:
                mlp = self.linear1_mlp(x_mod)[0]
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))[0]
        else:
            mlp = self.linear1_mlp(x_mod)
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))

        if self.i2v_condition_type == "token_replace":
            out_zero = output[:, :frist_frame_token_num] * tr_mod_gate
            out_orig = output[:, frist_frame_token_num:] * mod_gate
            output = torch.cat([out_zero, out_orig], dim=1)
            return (x + output, )
        else:
            return (x + output * mod_gate, )


def parallel_attention(
    q: Tuple[torch.Tensor, torch.Tensor],
    k: Tuple[torch.Tensor, torch.Tensor],
    v: Tuple[torch.Tensor, torch.Tensor],
    heads_num: int,
    head_dim: int,
    attn_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_kvlen: Optional[torch.Tensor] = None,
    context_parallel_algo: str = None,
    h2d_stream: Optional[torch_npu.npu.Stream] = None,
    d2h_stream: Optional[torch_npu.npu.Stream] = None,
    layer_index: int = 0,
    num_layers: int = 0,
    attention_async_offload: bool = False
):
    img_q, txt_q = q
    img_k, txt_k = k
    img_v, txt_v = v
    bs, img_seq_len, _, _ = img_q.shape
    # input: b s n d, output: b s h

    if context_parallel_algo == "ulysses_cp_algo":
        img_q = all_to_all(img_q, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)
        img_k = all_to_all(img_k, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)
        img_v = all_to_all(img_v, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)

        def shrink_head(txt, dim=2):
            txt = split_forward_gather_backward(txt, mpu.get_context_parallel_group(), dim=dim)
            return txt

        if bs == 1:
            attn = do_npu_fusion_attention(
                q=torch.cat([img_q, shrink_head(txt_q, 2)], dim=1).transpose(1, 2).contiguous(),
                k=torch.cat([img_k, shrink_head(txt_k, 2)], dim=1).transpose(1, 2).contiguous(),
                v=torch.cat([img_v, shrink_head(txt_v, 2)], dim=1).transpose(1, 2).contiguous(),
                head_num=heads_num // mpu.get_context_parallel_world_size(),
                attn_mask=attn_mask,
                layout="BNSD",
                softmax_scale=1 / math.sqrt(head_dim),
                async_offload=attention_async_offload,
                block_idx=layer_index,
                depth=num_layers,
                h2d_stream=h2d_stream,
                d2h_stream=d2h_stream
            )
            attn = attn.transpose(1, 2).contiguous()
            img_attn = attn[:, :img_seq_len * mpu.get_context_parallel_world_size()]  # b img_seq sub_n d
            txt_attn = attn[:, img_seq_len * mpu.get_context_parallel_world_size():]  # b txt_seq sub_n d
        else:
            raise NotImplementedError("not implemented ulysses_cp_algo for batch_size > 1")

        img_attn = all_to_all(img_attn, mpu.get_context_parallel_group(), scatter_dim=1, gather_dim=2)  # b sub_img_seq n d
        txt_attn = gather_forward_split_backward(txt_attn, mpu.get_context_parallel_group(), dim=2)  # b txt_seq n d
        attn = torch.cat([img_attn, txt_attn], dim=1)
        attn = attn.view(bs, -1, heads_num * head_dim)
    elif context_parallel_algo == "hybrid_cp_algo":
        img_q = all_to_all(img_q, get_context_parallel_group_for_hybrid_ulysses(), scatter_dim=2, gather_dim=1)
        img_k = all_to_all(img_k, get_context_parallel_group_for_hybrid_ulysses(), scatter_dim=2, gather_dim=1)
        img_v = all_to_all(img_v, get_context_parallel_group_for_hybrid_ulysses(), scatter_dim=2, gather_dim=1)
        split_sizes = cal_split_sizes(txt_q.shape[1], get_context_parallel_for_hybrid_ring_world_size())

        def shrink_seq_head(txt):
            split_sizes = cal_split_sizes(txt.shape[1], get_context_parallel_for_hybrid_ring_world_size())
            txt = split_forward_gather_backward(txt, get_context_parallel_group_for_hybrid_ring(), split_sizes=split_sizes, dim=1)
            txt = split_forward_gather_backward(txt, get_context_parallel_group_for_hybrid_ulysses(), dim=2)
            return txt

        q = torch.cat((img_q, shrink_seq_head(txt_q)), dim=1)
        k = torch.cat((img_k, shrink_seq_head(txt_k)), dim=1)
        v = torch.cat((img_v, shrink_seq_head(txt_v)), dim=1)

        q = q.view(bs, q.shape[1], -1).transpose(0, 1).contiguous()
        k = k.view(bs, k.shape[1], -1).transpose(0, 1).contiguous()
        v = v.view(bs, v.shape[1], -1).transpose(0, 1).contiguous()

        rank_shape = dict(zip(list(range(get_context_parallel_for_hybrid_ring_world_size())), [split_size + img_q.shape[1] for split_size in split_sizes]))

        attn = do_ring_context_parallel(
            q,
            k,
            v,
            head_num=heads_num // get_context_parallel_for_hybrid_ulysses_world_size(),
            softmax_scale=1 / math.sqrt(head_dim),
            attn_mask=None,
            shapes=rank_shape
        )

        attn = attn.transpose(0, 1)
        img_attn = attn[:, :img_seq_len * get_context_parallel_for_hybrid_ulysses_world_size()]
        txt_attn = attn[:, img_seq_len * get_context_parallel_for_hybrid_ulysses_world_size():]

        txt_attn = gather_forward_split_backward(txt_attn, get_context_parallel_group_for_hybrid_ring(), gather_sizes=split_sizes, dim=1)
        txt_attn = gather_forward_split_backward(txt_attn, get_context_parallel_group_for_hybrid_ulysses(), dim=2)

        img_attn = all_to_all(img_attn, get_context_parallel_group_for_hybrid_ulysses(), scatter_dim=1, gather_dim=2)

        attn = torch.cat([img_attn, txt_attn], dim=1).contiguous()
    elif context_parallel_algo == "megatron_cp_algo":
        split_sizes = cal_split_sizes(txt_q.shape[1], mpu.get_context_parallel_world_size())

        def shrink_seq_head(txt):
            split_sizes = cal_split_sizes(txt.shape[1], mpu.get_context_parallel_world_size())
            txt = split_forward_gather_backward(txt, mpu.get_context_parallel_group(), split_sizes=split_sizes, dim=1)
            return txt

        q = torch.cat((img_q, shrink_seq_head(txt_q)), dim=1)
        k = torch.cat((img_k, shrink_seq_head(txt_k)), dim=1)
        v = torch.cat((img_v, shrink_seq_head(txt_v)), dim=1)

        q = q.view(bs, q.shape[1], -1).transpose(0, 1).contiguous()
        k = k.view(bs, k.shape[1], -1).transpose(0, 1).contiguous()
        v = v.view(bs, v.shape[1], -1).transpose(0, 1).contiguous()

        rank_shape = dict(zip(list(range(mpu.get_context_parallel_world_size())), [split_size + img_q.shape[1] for split_size in split_sizes]))

        attn = do_ring_context_parallel(
            q,
            k,
            v,
            head_num=heads_num,
            softmax_scale=1 / math.sqrt(head_dim),
            attn_mask=None,
            shapes=rank_shape
        )

        attn = attn.transpose(0, 1)
        img_attn = attn[:, :img_seq_len]
        txt_attn = attn[:, img_seq_len:]

        txt_attn = gather_forward_split_backward(txt_attn, mpu.get_context_parallel_group(), gather_sizes=split_sizes, dim=1)

        attn = torch.cat([img_attn, txt_attn], dim=1).contiguous()
    else:
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        if bs == 1:
            attn = do_npu_fusion_attention(
                q=q.transpose(1, 2).contiguous(),
                k=k.transpose(1, 2).contiguous(),
                v=v.transpose(1, 2).contiguous(),
                head_num=heads_num,
                attn_mask=attn_mask,
                layout="BNSD",
                softmax_scale=1 / math.sqrt(head_dim),
                async_offload=attention_async_offload,
                block_idx=layer_index,
                depth=num_layers,
                h2d_stream=h2d_stream,
                d2h_stream=d2h_stream
            )
            attn = attn.transpose(1, 2).contiguous()
            attn = attn.view(bs, -1, heads_num * head_dim)
        else:
            q = q.view(-1, heads_num * head_dim)
            k = k.view(-1, heads_num * head_dim)
            v = v.view(-1, heads_num * head_dim)
            attn = do_npu_fusion_attention(
                q=q,
                k=k,
                v=v,
                head_num=heads_num,
                attn_mask=attn_mask,
                layout="TND",
                actual_seq_qlen=actual_seq_qlen.tolist(),
                actual_seq_kvlen=actual_seq_kvlen.tolist(),
                softmax_scale=1 / math.sqrt(head_dim),
                async_offload=attention_async_offload,
                block_idx=layer_index,
                depth=num_layers,
                h2d_stream=h2d_stream,
                d2h_stream=d2h_stream
            )
            attn = attn.view(bs, -1, heads_num * head_dim)

    return attn
