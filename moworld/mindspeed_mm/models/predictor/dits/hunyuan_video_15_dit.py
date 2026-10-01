# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.
import os
from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
from einops import rearrange

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.attention import parallel_attention, get_activation_layer, \
    get_norm_layer
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.communications import all_gather
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, \
    VisionProjection, MLP, LinearWarpforSingle, MLPEmbedder, FinalLayer, ModulateDiT, modulate, \
    apply_gate, apply_rotary_emb, get_nd_rotary_pos_embed
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.token_refiner import SingleTokenRefiner
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.utils import get_parallel_state, sync_tensor_for_sp
from mindspeed_mm.models.predictor.dits.hunyuanvideo15.utils import maybe_fallback_attn_mode
from mindspeed_mm.models.text_encoder.hunyuan15_byt5 import ByT5Mapper


class HunyuanVideo15DiT(MultiModalModule):

    def __init__(
            self,
            model_id: str = "hunyuanvideo15dit",
            from_pretrained: str = None,
            dtype: str = "bf16",
            patch_size: list = None,
            in_channels: int = 4,
            concat_condition: bool = True,
            out_channels: int = None,
            hidden_size: int = 3072,
            num_heads: int = 24,
            mlp_width_ratio: float = 4.0,
            mlp_act_type: str = "gelu_tanh",
            mm_double_blocks_depth: int = 20,
            mm_single_blocks_depth: int = 40,
            rope_dim_list: list = None,
            qkv_bias: bool = True,
            qk_norm: bool = True,
            qk_norm_type: str = "rms",
            guidance_embed: bool = False,
            use_meanflow: bool = False,
            text_projection: str = "single_refiner",
            use_attention_mask: bool = True,
            text_states_dim: int = 4096,
            text_states_dim_2: int = 768,
            text_pool_type: str = None,
            rope_theta: int = 256,
            attn_mode: str = "flash",
            glyph_byT5_v2: bool = False,
            vision_projection: str = "none",
            vision_states_dim: int = 1280,
            is_reshape_temporal_channels: bool = False,
            use_cond_type_embedding: bool = False,
            byt5_in_in_dim: int = 1472,
            byt5_in_out_dim: int = 2048,
            byt5_in_hidden_dim: int = 2048,
            attn_param: dict = None,
            task_type: str = "t2v",
            **kwargs
    ):
        super().__init__(config=None)
        self.model_id = model_id
        self.from_pretrained = from_pretrained
        self.kwargs_dict = kwargs
        factory_kwargs = {}

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner.
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.attn_mode = attn_mode
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.byt5_in_in_dim = byt5_in_in_dim
        self.byt5_in_out_dim = byt5_in_out_dim
        self.byt5_in_hidden_dim = byt5_in_hidden_dim

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=self.byt5_in_in_dim,
                out_dim=self.byt5_in_out_dim,
                hidden_dim=self.byt5_in_hidden_dim,
                out_dim1=hidden_size,
                use_residual=False
            )

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.task_type = task_type

        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels, concat_condition=concat_condition,
            **factory_kwargs
        )

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                num_heads,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )
        self.vector_in = (
            MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            ) if self.text_pool_type is not None else None
        )
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )

        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    attn_mode=attn_mode,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # STA
        if attn_param is None:
            raise AssertionError("model needs attn_param")
        else:
            self.attn_param = attn_param

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)

            if not self.glyph_byT5_v2:
                raise AssertionError("text type embedding is only used when glyph_byT5_v2 is True")
            if vision_projection is None:
                raise AssertionError("text type embedding is only used when vision_projection is not None")
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None

        self.parallel_state = get_parallel_state()
        self.sp_enabled = self.parallel_state.sp_enabled
        self.sp_group = self.parallel_state.sp_group if self.sp_enabled else None

    def load_hunyuan_state_dict(self, model_path):
        load_key = "module"
        bare_model = "unknown"

        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=True
            )

        if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
            bare_model = False
        if bare_model is False:
            if load_key in state_dict:
                state_dict = state_dict[load_key]
            else:
                raise KeyError(
                    f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                    f"are: {list(state_dict.keys())}."
                )

        result = self.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print("[load.py] Missing keys when loading state_dict:")
        if result.unexpected_keys:
            print("[load.py] Unexpected keys when loading state_dict:")
        if result.missing_keys or result.unexpected_keys:
            pass

        return result

    def load_state_dict_with_dtype(self, model, state_dict, target_dtype, strict=False):
        model = model.to(target_dtype)
        converted_state_dict = {}
        for k, v in state_dict.items():
            if v.dtype in [torch.float16, torch.float32, torch.float64]:
                converted_state_dict[k] = v.to(target_dtype)
            else:
                converted_state_dict[k] = v
        result = model.load_state_dict(converted_state_dict, strict=strict)
        return result

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        if not (
                sum(rope_dim_list) == head_dim
        ):
            raise AssertionError("sum(rope_dim_list) should equal to head_dim of attention layer")
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], byt5_txt_i[~byt5_text_mask_i],
                         txt_i[~text_mask_i]], dim=0
                    )
                reorder_mask_i = torch.cat(
                    [byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i], byt5_text_mask_i[~byt5_text_mask_i],
                     text_mask_i[~text_mask_i]], dim=0
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.long)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.long)

        return reorder_txt, reorder_mask

    def forward(
            self,
            noised_latents: torch.Tensor,
            timestep: torch.LongTensor,
            prompt: List[torch.Tensor],
            prompt_mask: List[torch.Tensor],
            timestep_r=None,
            output_features=False,
            output_features_stride=8,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None,
            guidance=None,
            **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            parallel_dims = get_parallel_state()

            if self.training:
                b, c, f, h, w = noised_latents.shape
                cond_latents = torch.zeros([b, c + 1, f, h, w], device=noised_latents.device, dtype=noised_latents.dtype)
                if self.task_type == "t2v":
                    cond_latents = cond_latents
                elif self.task_type == "i2v":
                    cond_latents = kwargs.get("cond_latents", None)
                    if cond_latents is None:
                        raise ValueError(f"cond_latents cannot be None")
                else:
                    raise ValueError(f"Do not support task type:{self.task_type}")
                hidden_states = torch.cat([noised_latents, cond_latents], dim=1)
            else:
                hidden_states = noised_latents

            vision_states = kwargs.get("vision_states", None)

            if guidance is None:
                guidance = torch.tensor(
                    [6016.0], device=hidden_states.device, dtype=torch.bfloat16
                )
            img = x = hidden_states.to(self.dtype)
            if parallel_dims.sp_enabled:
                prompt_mask = sync_tensor_for_sp(prompt_mask, parallel_dims.sp_group)
                prompt = sync_tensor_for_sp(prompt, parallel_dims.sp_group)

            text_mask = prompt_mask[0][0]
            t = timestep
            txt = prompt[0][0]
            bs, _, ot, oh, ow = x.shape
            tt, th, tw = (
                ot // self.patch_size[0],
                oh // self.patch_size[1],
                ow // self.patch_size[2],
            )
            self.attn_param['thw'] = [tt, th, tw]
            if freqs_cos is None and freqs_sin is None:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

            img = self.img_in(img)

            sp_enabled = parallel_dims.sp_enabled
            if sp_enabled:
                sp_size = parallel_dims.sp
                sp_rank = parallel_dims.sp_rank
                if img.shape[1] % sp_size != 0:
                    n_token = img.shape[1]
                    if not n_token > (n_token // sp_size + 1) * (sp_size - 1):
                        raise AssertionError(f'Too short context length for SP {sp_size}')
                img = torch.chunk(img, sp_size, dim=1)[sp_rank]
                freqs_cos = torch.chunk(freqs_cos, sp_size, dim=0)[sp_rank]
                freqs_sin = torch.chunk(freqs_sin, sp_size, dim=0)[sp_rank]

            # Prepare modulation vectors
            vec = self.time_in(t)

            if self.guidance_embed:
                if guidance is None:
                    raise ValueError(
                        "Didn't get guidance strength for guidance distilled model."
                    )
                vec = vec + self.guidance_in(guidance)

            if timestep_r is not None:
                vec = vec + self.time_r_in(timestep_r)

            # Embed text tokens
            if self.text_projection == "linear":
                txt = self.txt_in(txt)
            elif self.text_projection == "single_refiner":
                txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
            else:
                raise NotImplementedError(
                    f"Unsupported text_projection: {self.text_projection}"
                )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
                )
                txt = txt + cond_emb

            if self.glyph_byT5_v2 and len(prompt) == 2 and len(prompt_mask) == 2:
                byt5_text_states = prompt[1][0]
                byt5_text_mask = prompt_mask[1][0]
                byt5_txt = self.byt5_in(byt5_text_states)
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(
                        torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                    )
                    byt5_txt = byt5_txt + cond_emb
                txt, text_mask = self.reorder_txt_token(
                    byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True
                )

            if self.vision_in is not None and vision_states is not None:
                extra_encoder_hidden_states = self.vision_in(vision_states)
                # If t2v, set extra_attention_mask to 0 to avoid attention to semantic tokens
                if self.task_type == "t2v" and torch.all(vision_states == 0):
                    extra_attention_mask = torch.zeros(
                        (bs, extra_encoder_hidden_states.shape[1]),
                        dtype=text_mask.dtype,
                        device=text_mask.device,
                    )
                    # Set vision tokens to zero to mitigate potential block mask error in SSTA
                    extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
                else:
                    extra_attention_mask = torch.ones(
                        (bs, extra_encoder_hidden_states.shape[1]),
                        dtype=text_mask.dtype,
                        device=text_mask.device,
                    )
                # Ensure valid tokens precede padding tokens
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(
                        2 * torch.ones_like(
                            extra_encoder_hidden_states[:, :, 0],
                            dtype=torch.long,
                            device=extra_encoder_hidden_states.device,
                        )
                    )
                    extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb

                txt, text_mask = self.reorder_txt_token(
                    extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
                )

            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

            # Pass through double-stream blocks
            for index, block in enumerate(self.double_blocks):
                force_full_attn = (
                        self.attn_mode in ["flex-block-attn"]
                        and self.attn_param["win_type"] == "hybrid"
                        and self.attn_param["win_ratio"] > 0
                        and (
                                (index + 1) % self.attn_param["win_ratio"] == 0
                                or (index + 1) == len(self.double_blocks)
                        )
                )
                self.attn_param["layer-name"] = f"double_block_{index + 1}"
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    freqs_cis=freqs_cis,
                    text_mask=text_mask,
                    attn_param=self.attn_param,
                    is_flash=force_full_attn,
                    block_idx=index,
                )

            txt_seq_len = txt.shape[1]
            img_seq_len = img.shape[1]

            # Merge image and text for single-stream blocks
            x = torch.cat((img, txt), 1)
            features_list = [] if output_features else None
            if len(self.single_blocks) > 0:
                for index, block in enumerate(self.single_blocks):
                    force_full_attn = (
                            self.attn_mode in ["flex-block-attn"]
                            and self.attn_param["win_type"] == "hybrid"
                            and self.attn_param["win_ratio"] > 0
                            and (
                                    (index + 1) % self.attn_param["win_ratio"] == 0
                                    or (index + 1) == len(self.single_blocks)
                            )
                    )
                    self.attn_param["layer-name"] = f"single_block_{index + 1}"
                    x = block(
                        x=x,
                        vec=vec,
                        txt_len=txt_seq_len,
                        freqs_cis=(freqs_cos, freqs_sin),
                        text_mask=text_mask,
                        attn_param=self.attn_param,
                        is_flash=force_full_attn,
                    )
                    if output_features and index % output_features_stride == 0:
                        features_list.append(x[:, :img_seq_len, ...])
            img = x[:, :img_seq_len, ...]

            # Final Layer
            img = self.final_layer(img, vec)
            if sp_enabled:
                img = all_gather(img, dim=1, group=parallel_dims.sp_group)
            img = self.unpatchify(img, tt, th, tw)
            if output_features:
                features_list = torch.stack(features_list, dim=0)
                if sp_enabled:
                    features_list = all_gather(features_list, dim=2, group=parallel_dims.sp_group)
            else:
                features_list = None
            return (img, features_list)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def unpatchify(self, x, t, h, w):
        """
        Unpatchify a tensorized input back to frame format.

        Args:
            x (Tensor): Input tensor of shape (N, T, patch_size**2 * C)
            t (int): Number of time steps
            h (int): Height in patch units
            w (int): Width in patch units

        Returns:
            Tensor: Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        if not t * h * w == x.shape[1]:
            raise AssertionError(f"model nees t * h * w == x.shape[1]")
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def set_attn_mode(self, attn_mode: str):
        attn_mode = maybe_fallback_attn_mode(attn_mode)
        self.attn_mode = attn_mode
        for block in self.double_blocks:
            block.attn_mode = attn_mode
        for block in self.single_blocks:
            block.attn_mode = attn_mode

    def save_lora_adapter(
            self,
            save_directory,
            adapter_name: str = "default",
            upcast_before_saving: bool = False,
            safe_serialization: bool = True,
            weight_name: Optional[str] = None,
    ):
        """
        Save the LoRA parameters corresponding to the underlying model.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            adapter_name: (`str`, defaults to "default"): The name of the adapter to serialize. Useful when the
                underlying model has multiple adapters loaded.
            upcast_before_saving (`bool`, defaults to `False`):
                Whether to cast the underlying model to `torch.float32` before serialization.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            weight_name: (`str`, *optional*, defaults to `None`): Name of the file to serialize the state dict with.
        """
        from peft.utils import get_peft_model_state_dict

        from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY, LORA_WEIGHT_NAME, LORA_WEIGHT_NAME_SAFE
        from diffusers.utils import get_adapter_name
        import safetensors
        import json
        from pathlib import Path

        if adapter_name is None:
            adapter_name = get_adapter_name(self)

        if adapter_name not in getattr(self, "peft_config", {}):
            raise ValueError(f"Adapter name {adapter_name} not found in the model.")

        lora_adapter_metadata = self.peft_config[adapter_name].to_dict()

        lora_layers_to_save = get_peft_model_state_dict(
            self.to(dtype=torch.float32 if upcast_before_saving else None), adapter_name=adapter_name
        )
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if safe_serialization:

            def save_function(weights, filename):
                # Inject framework format.
                metadata = {"format": "pt"}
                if lora_adapter_metadata is not None:
                    for key, value in lora_adapter_metadata.items():
                        if isinstance(value, set):
                            lora_adapter_metadata[key] = list(value)
                    metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(lora_adapter_metadata, indent=2, sort_keys=True)

                return safetensors.torch.save_file(weights, filename, metadata=metadata)

        else:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_path = Path(save_directory, weight_name).as_posix()
        lora_layers_to_save = {
            k: (v.full_tensor() if hasattr(v, 'full_tensor') else v)
            for k, v in lora_layers_to_save.items()
        }
        if os.environ.get('RANK', '0') == '0':
            try:
                save_function(lora_layers_to_save, save_path)
            except OSError as e:
                print(f"Failed to save model: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
        print(f"Model weights saved in {save_path}")


class MMDoubleStreamBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_width_ratio: float,
            mlp_act_type: str = "gelu_tanh",
            attn_mode: str = None,
            qk_norm: bool = True,
            qk_norm_type: str = "rms",
            qkv_bias: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.num_heads = num_heads
        self.attn_mode = attn_mode

        if hidden_size % num_heads != 0:
            raise AssertionError(f"hidden_size({hidden_size}) must be divisible by num_heads({num_heads})")
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True,
                           **factory_kwargs)

        self.txt_mod = ModulateDiT(
            hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True,
                           **factory_kwargs)

        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
            self,
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            freqs_cis: tuple = None,
            text_mask=None,
            attn_param=None,
            is_flash=False,
            block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.num_heads)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.num_heads)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.num_heads)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            if not (
                    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ):
                raise AssertionError(
                    f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}")
            img_q, img_k = img_qq, img_kk

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.num_heads)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.num_heads)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.num_heads)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        attn_mode = 'flash' if is_flash else self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
            block_idx=block_idx,
        )

        img_attn, txt_attn = attn[:, :img_q.shape[1]].contiguous(), attn[:, img_q.shape[1]:].contiguous()

        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)
            ),
            gate=img_mod2_gate,
        )

        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_width_ratio: float = 4.0,
            mlp_act_type: str = "gelu_tanh",
            attn_mode: str = None,
            qk_norm: bool = True,
            qk_norm_type: str = "rms",
            qk_scale: float = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.attn_mode = attn_mode

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)
        self.mlp_act = get_activation_layer(mlp_act_type)()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
            self,
            x: torch.Tensor,
            vec: torch.Tensor,
            txt_len: int,
            freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
            text_mask=None,
            attn_param=None,
            is_flash=False,
    ) -> torch.Tensor:
        """Forward pass for the single stream block."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.num_heads)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.num_heads)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.num_heads)

        mlp = self.linear1_mlp(x_mod)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        if not (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ):
            raise AssertionError(
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}")
        img_q, img_k = img_qq, img_kk

        if is_flash:
            attn_mode = 'flash'
        else:
            attn_mode = self.attn_mode
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
            attn_mode=attn_mode,
            attn_param=attn_param,
        )
        output = self.linear2(attn, self.mlp_act(mlp))

        return x + apply_gate(output, gate=mod_gate)
