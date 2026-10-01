# Modified from FLUX
# license: Apache-2.0 License
from typing import Tuple, Optional
import torch
import torch_npu
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops import repeat

from megatron.core import mpu
from megatron.training import get_args
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.checkpoint import auto_grad_checkpoint
from mindspeed_mm.models.common.blocks import FinalLayer
from mindspeed_mm.models.common.embeddings import TimestepEmbedder
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.attention import ParallelAttention
from mindspeed_mm.models.common.blocks import MLP, ModulateDiT
from mindspeed_mm.models.common.activations import get_activation_layer


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class LigerRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim)
        sin size: (1, seq_len, head_dim)
        """
        q = q.contiguous()
        k = k.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # rotate position embedding
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # save the information needed for backpropagation
        ctx.save_for_backward(cos, sin)

        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    def backward(self, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim)
        sin size: (1, seq_len, head_dim)
        """
        # Get the forward-saved cos and sin
        cos, sin = self.saved_tensors

        dq = dq.contiguous()
        dk = dk.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

        # rotate position embedding
        dq_rotated = (dq * cos) + (rotate_half(dq) * sin)
        dk_rotated = (dk * cos) + (rotate_half(dk) * sin)

        # Returns 6 values, corresponding to 6 inputs for forward
        # q, k, cos, sin, position_ids, unsqueeze_dim
        return (
            dq_rotated.to(dq.dtype),
            dk_rotated.to(dk.dtype),
            None,  # cos does not require gradients
            None,  # sin does not require gradients
            None,  # position_ids does not require gradients
            None  # unsqueeze_dim does not require gradients
        )


class LigerEmbedND(nn.Module):
    def __init__(self, theta: int = 10000, axes_dim=None):
        axes_dim = axes_dim or [16, 56, 56]
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, v: Tensor, pe):
        if isinstance(pe, torch.Tensor):
            q, k = apply_rope(q, k, pe)
        else:
            cos, sin = pe
            q, k = LigerRopeFunction.apply(q, k, cos, sin)
        q = rearrange(q, "b n s d -> b s n d")
        k = rearrange(k, "b n s d -> b s n d")
        v = rearrange(v, "b n s d -> b s n d")
        return q, k, v

    def liger_rope(self, pos: Tensor, dim: int):
        if dim % 2 == 1:
            raise ValueError(f"dim {dim} must be an even number.")
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (self.theta ** scale)
        out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
        cos = out.cos()
        sin = out.sin()
        return cos, sin

    def forward(self, ids: Tensor):
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            cos, sin = self.liger_rope(ids[..., i], self.axes_dim[i])
            cos_list.append(cos)
            sin_list.append(sin)

        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()
        return cos_emb, sin_emb


class MMDiT(MultiModalModule):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        hidden_size: int = 3072,
        num_heads: int = 24,
        mm_double_blocks_depth: int = 19,
        mm_single_blocks_depth: int = 38,
        double_stream_full_recompute_layers: Optional[int] = None,
        single_stream_full_recompute_layers: Optional[int] = None,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        vec_in_dim: int = 768,
        context_in_dim: int = 768,
        attention_q_bias: bool = False,
        attention_k_bias: bool = False,
        attention_v_bias: bool = False,
        fused_qkv: bool = False,
        guidance_embed: bool = False,
        guidance: float = 4.0,
        cond_embed: bool = False,
        use_liger_rope: bool = False,
        norm_type: str = "rmsnorm",
        time_factor: float = 1000.0,
        **kwargs
    ):
        super().__init__(config=None)
        self.in_channels = in_channels
        self.out_channels = in_channels

        # model size related
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.guidance_embed = guidance_embed
        self.guidance = guidance
        self.cond_embed = cond_embed

        # computation related
        if mpu.get_context_parallel_world_size() > 1:
            self.enable_sequence_parallelism = True
        else:
            self.enable_sequence_parallelism = False

        # input size related
        self.patch_size = patch_size

        # embedding
        self.pe_embedder = LigerEmbedND()
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = TimestepEmbedder(self.hidden_size)
        self.time_factor = time_factor
        self.vector_in = MLP(
            in_channels=vec_in_dim,
            hidden_channels=self.hidden_size,
            out_features=self.hidden_size,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_sequence_parallelism,
            enable_tp_sp=False
        )
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size)
            if guidance_embed
            else nn.Identity()
        )
        self.cond_in = (
            nn.Linear(self.in_channels + self.patch_size ** 2, self.hidden_size, bias=True)
            if cond_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        # blocks
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    mlp_act_type=mlp_act_type,
                    attention_q_bias=attention_q_bias,
                    attention_k_bias=attention_k_bias,
                    attention_v_bias=attention_v_bias,
                    fused_qkv=fused_qkv,
                    norm_type=norm_type,
                    rope=self.pe_embedder
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    mlp_act_type=mlp_act_type,
                    fused_qkv=fused_qkv,
                    norm_type=norm_type,
                    rope=self.pe_embedder
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        # final layer
        self.final_layer = FinalLayer(self.hidden_size, 1, self.out_channels)

        # recompute
        args = get_args()
        self.recompute_granularity = args.recompute_granularity
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in MMDiT")
        self.double_stream_full_recompute_layers = double_stream_full_recompute_layers or mm_double_blocks_depth
        self.single_stream_full_recompute_layers = single_stream_full_recompute_layers or mm_single_blocks_depth

        self._input_requires_grad = False
        self.initialize_weights()

    def initialize_weights(self):
        if self.cond_embed:
            nn.init.zeros_(self.cond_in.weight)
            nn.init.zeros_(self.cond_in.bias)

    def forward(self, latents, timestep, prompt, **kwargs):
        prompt = self.process_prompt(prompt)
        img_ids, txt_ids = self.prepare_ids(prompt, **kwargs)
        cond = kwargs.get("cond", None)
        guidance = torch.full(
            (latents.shape[0],), self.guidance, device=latents.device, dtype=latents.dtype
        )
        img, txt, vec, pe = self.prepare_block_inputs(
            latents, img_ids, prompt[0], txt_ids, timestep, prompt[1], cond, guidance
        )

        ckpt_depth_double = 0
        ckpt_depth_single = 0
        if self.recompute_granularity == "full":
            ckpt_depth_double = self.double_stream_full_recompute_layers
            ckpt_depth_single = self.single_stream_full_recompute_layers

        for block in self.double_blocks[:ckpt_depth_double]:
            img, txt = auto_grad_checkpoint(block, img, txt, vec, pe)

        for block in self.double_blocks[ckpt_depth_double:]:
            img, txt = block(img, txt, vec, pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks[:ckpt_depth_single]:
            img = auto_grad_checkpoint(block, img, vec, pe)
        for block in self.single_blocks[ckpt_depth_single:]:
            img = block(img, vec, pe)

        img = img[:, txt.shape[1]:, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def prepare_ids(self, prompt, shape, **kwargs):
        B, C, T, H, W = shape
        t5_embedding = prompt[0]
        img_ids = torch.zeros(T, H // 2, W // 2, 3)
        img_ids[..., 0] = img_ids[..., 0] + torch.arange(T)[:, None, None]
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(H // 2)[None, :, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(W // 2)[None, None, :]
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=t5_embedding.shape[0])
        txt_ids = torch.zeros(t5_embedding.shape[0], t5_embedding.shape[1], 3)
        return img_ids.to(t5_embedding.device, t5_embedding.dtype), txt_ids.to(t5_embedding.device, t5_embedding.dtype)

    def process_prompt(self, prompt):
        for i, _ in enumerate(prompt):
            prompt[i] = prompt[i].squeeze(1)
        return prompt

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,  # t5 encoded vec
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,  # clip encoded vec
        cond: Tensor = None,
        guidance: Tensor = None,
    ):
        """
        obtain the processed:
            img: projected noisy img latent,
            txt: text context (from t5),
            vec: clip encoded vector,
            pe: the positional embeddings for concatenated img and txt
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                f"Input img and txt tensors must have 3 dimensions. img shape is {img.shape}, txt shape is {txt.shape}.")

        # running on sequences img
        img = self.img_in(img)
        if self.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            img = img + self.cond_in(cond)

        vec = self.time_in(timesteps, time_factor=self.time_factor)
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(guidance, time_factor=self.time_factor)
        vec = vec + self.vector_in(y_vec)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        return img, txt, vec, pe


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        attention_q_bias: bool = False,
        attention_k_bias: bool = False,
        attention_v_bias: bool = False,
        attention_out_bias: bool = True,
        fused_qkv: bool = True,
        norm_type: str = "rmsnorm",
        rope=None,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.rope = rope
        self.fused_qkv = fused_qkv
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1

        # image stream
        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=False,
            zero_initialize=False
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = ParallelAttention(
            query_dim=self.hidden_size,
            key_dim=None,
            num_attention_heads=self.num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_q_bias,
            proj_k_bias=attention_k_bias,
            proj_v_bias=attention_v_bias,
            proj_out_bias=attention_out_bias,
            norm_type=norm_type,
            norm_eps=1e-6,
            use_qk_norm=True,
            is_qkv_concat=fused_qkv,
            fa_layout="bsnd",
            rope=self.rope
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tp_sp=False
        )

        # text stream
        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=False,
            zero_initialize=False
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = ParallelAttention(
            query_dim=self.hidden_size,
            key_dim=None,
            num_attention_heads=self.num_heads,
            hidden_size=hidden_size,
            proj_q_bias=attention_q_bias,
            proj_k_bias=attention_k_bias,
            proj_v_bias=attention_v_bias,
            proj_out_bias=attention_out_bias,
            norm_type=norm_type,
            norm_eps=1e-6,
            use_qk_norm=True,
            is_qkv_concat=fused_qkv,
            fa_layout="bsnd",
            rope=self.rope
        )
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tp_sp=False
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, **kwargs):
        # attn is the DoubleStreamBlock;
        # process img and txt separately while both is influenced by text vec
        # vec will interact with image latent and text context
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate
        ) = self.img_mod(vec)[:, None, :].chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate
        ) = self.txt_mod(vec)[:, None, :].chunk(6, dim=-1)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        img_q, img_k, img_v = self.img_attn.function_before_core_attention(img_modulated, input_layout="bsh")

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        txt_q, txt_k, txt_v = self.txt_attn.function_before_core_attention(txt_modulated, input_layout="bsh")

        if not self.fused_qkv:
            img_q = rearrange(img_q, "s b n d -> b n s d")
            img_k = rearrange(img_k, "s b n d -> b n s d")
            img_v = rearrange(img_v, "s b n d -> b n s d")
            txt_q = rearrange(txt_q, "s b n d -> b n s d")
            txt_k = rearrange(txt_k, "s b n d -> b n s d")
            txt_v = rearrange(txt_v, "s b n d -> b n s d")

        # run actual attention, image and text attention are calculated together by concat different attn heads
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        q, k, v = self.rope.apply_rotary_pos_emb(q, k, v, pe)
        head_num = q.shape[2]
        scale = q.shape[-1] ** (-0.5)
        attn1 = torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=1.0, sparse_mode=2, scale=scale)[0]
        attn1 = rearrange(attn1, "b s n d -> b s (n d)")
        txt_attn, img_attn = attn1[:, : txt_q.shape[2]], attn1[:, txt_q.shape[2]:]

        # calculate the img blocks
        img = img + img_mod1_gate * self.img_attn.function_after_core_attention(img_attn)
        img = img + img_mod2_gate * self.img_mlp((1 + img_mod2_scale) * self.img_norm2(img) + img_mod2_shift)

        # calculate the txt blocks
        txt = txt + txt_mod1_gate * self.txt_attn.function_after_core_attention(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp((1 + txt_mod2_scale) * self.txt_norm2(txt) + txt_mod2_shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_scale: float = None,
        fused_qkv: bool = True,
        norm_type: str = "rmsnorm",
        rope=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.fused_qkv = fused_qkv
        self.rope = rope
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1

        if fused_qkv:
            # qkv and mlp_in
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)

        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.q_norm = normalize(self.head_dim, eps=1e-6, norm_type=norm_type)
        self.k_norm = normalize(self.head_dim, eps=1e-6, norm_type=norm_type)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=False,
            zero_initialize=False
        )

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, **kwargs):
        mod_shift, mod_scale, mod_gate = self.modulation(vec)[:, None, :].chunk(3, dim=-1)
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        if self.fused_qkv:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            q = rearrange(self.q_proj(x_mod), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x_mod), "B L (H D) -> B L H D", H=self.num_heads)
            v, mlp = torch.split(self.v_mlp(x_mod), [self.hidden_size, self.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=self.num_heads)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        if not self.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")

        # compute attention
        q, k, v = self.rope.apply_rotary_pos_emb(q, k, v, pe)

        # fa
        head_num = q.shape[2]
        scale = q.shape[-1] ** (-0.5)
        attn_1 = torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=1.0, sparse_mode=2, scale=scale)[0]
        attn_1 = rearrange(attn_1, "b s n d -> b s (n d)")

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn_1, self.mlp_act(mlp)), 2))
        output = x + mod_gate * output
        return output
