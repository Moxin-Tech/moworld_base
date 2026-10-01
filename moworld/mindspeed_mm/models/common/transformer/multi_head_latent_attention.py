# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.utils import set_actual_seq_len, set_position_ids, get_actual_seq_len, get_position_ids
try:
    from mindspeed.core.pipeline_parallel.fb_overlap.modules.attention import launch_async_all2all_hook, launch_async_all2all
    from .mla_up_proj_overlap_tp_comm import mla_up_projection_overlap_tp_comm
except ImportError:
    pass

from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.training import get_args

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


@dataclass
class MLASelfAttentionSubmodules(SelfAttentionSubmodules):
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    linear_qb: Union[ModuleSpec, type] = None
    linear_kvb: Union[ModuleSpec, type] = None


@dataclass
class MLASelfAttentionWithMMSplitSubmodules(SelfAttentionSubmodules):
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    linear_qk_nope: Union[ModuleSpec, type] = None
    linear_kv_nope: Union[ModuleSpec, type] = None
    linear_qk_rope: Union[ModuleSpec, type] = None
    linear_v: Union[ModuleSpec, type] = None


class MultiHeadLatentAttention(SelfAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )
        args = get_args()

        self.use_flash_attn = args.use_flash_attn
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.v_head_dim = args.v_head_dim

        self.mla_mm_split = self.config.mla_mm_split
        self.mla_fa_without_pad = self.config.mla_fa_without_pad
        self.padded_base_length = self.config.padded_base_length
        self.shape_order = self.config.shape_order

        query_projection_size = self.config.num_attention_heads * self.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        max_dim = max(self.v_head_dim, self.q_head_dim)
        self.fa_padding_length = math.ceil(max_dim / self.padded_base_length) * self.padded_base_length

        if self.q_lora_rank is None:
            self.q_rank = self.config.num_attention_heads * self.q_head_dim
            self.q_layernorm = None
        else:
            self.q_rank = self.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.q_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.q_layernorm = None

            if not self.mla_mm_split:
                self.linear_qb = build_module(
                    submodules.linear_qb,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.q_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qb",
                )
            else:
                self.linear_qk_nope = build_module(
                    submodules.linear_qk_nope,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.qk_nope_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qk_nope",
                )
                self.linear_qk_rope = build_module(
                    submodules.linear_qk_rope,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.qk_rope_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qk_rope",
                )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.q_rank + self.kv_lora_rank + self.qk_rope_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.kv_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        if not self.mla_mm_split:
            self.linear_kvb = build_module(
                submodules.linear_kvb,
                self.kv_lora_rank,
                self.config.num_attention_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kvb",
            )
        else:
            self.linear_kv_nope = build_module(
                submodules.linear_kv_nope,
                self.kv_lora_rank,
                self.config.num_attention_heads * self.qk_nope_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_nope",
            )
            self.linear_v = build_module(
                submodules.linear_v,
                self.kv_lora_rank,
                self.config.num_attention_heads * self.v_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="v",
            )

        self.linear_proj = build_module(
            submodules.linear_proj,
            query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )
        # hook async A2A launcher inside mla forward when TP > 1.
        # a2a should be launched after TP communication finished to avoid bandwidth compete.
        if args.moe_fb_overlap and parallel_state.get_tensor_model_parallel_world_size() > 1:
            self.a2a_hooked_on_attention = True
        else:
            self.a2a_hooked_on_attention = False

        self.mla_up_proj_tp_overlap = self.config.mla_up_proj_tp_overlap
        self.recompute_mla_up_proj = self.config.recompute_mla_up_proj
        self.mla_zero_memory = self.config.mla_zero_memory

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_params=None,
    ):
        """
        Do patch for repeating KV so that GQA+Ulysses is better supported.
        """
        args = get_args()

        def mla_attention(hidden_states):
            args = get_args()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

            # For self attention we just duplicate the rotary_pos_emb if it isn't already
            nonlocal rotary_pos_emb
            if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_len, bsz, _ = hidden_states.shape
            q_len = q_len * tp_size if self.config.sequence_parallel else q_len

            qkv_combo = self.linear_qkv(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, hn]
            q_a, compressed_kv, k_pe = torch.split(
                qkv_combo,
                [
                    self.q_rank,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                ],
                dim=-1,
            )

            if self.mla_up_proj_tp_overlap:
                query, key, value = mla_up_projection_overlap_tp_comm(q_a, compressed_kv, k_pe, rotary_pos_emb,
                                                                      packed_seq_params, self)
            else:
                if self.q_layernorm is not None:
                    q_a = self.q_layernorm(q_a)
                    if not self.mla_mm_split:
                        q, _ = self.linear_qb(q_a)
                        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                        q_nope, q_pe = torch.split(
                            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                        )
                    else:
                        q_nope, _ = self.linear_qk_nope(q_a)
                        q_pe, _ = self.linear_qk_rope(q_a)
                        q_nope = q_nope.view(
                            q_len, bsz, self.num_attention_heads_per_partition, -1
                        )
                        q_pe = q_pe.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                else:
                    if self.config.sequence_parallel:
                        q_a = gather_from_sequence_parallel_region(q_a)
                    q_a = tensor_parallel.scatter_to_tensor_model_parallel_region(q_a)
                    q = q_a.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                    q_nope, q_pe = torch.split(
                        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                    )

                if self.config.sequence_parallel:
                    k_pe = gather_from_sequence_parallel_region(k_pe)
                else:
                    k_pe = tensor_parallel.copy_to_tensor_model_parallel_region(k_pe)

                k_pe = k_pe.view(q_len, bsz, 1, self.qk_rope_head_dim)
                compressed_kv_norm = self.k_layernorm(compressed_kv)

                if not self.mla_mm_split:
                    kv, _ = self.linear_kvb(compressed_kv_norm)
                    kv = kv.view(
                        q_len,
                        bsz,
                        self.num_attention_heads_per_partition,
                        self.qk_nope_head_dim + self.v_head_dim,
                    )
                    k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                else:
                    k_nope, _ = self.linear_kv_nope(compressed_kv_norm)
                    value, _ = self.linear_v(compressed_kv_norm)
                    k_nope = k_nope.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                    value = value.view(q_len, bsz, self.num_attention_heads_per_partition, -1)

                if self.a2a_hooked_on_attention:
                    launch_async_all2all()

                if rotary_pos_emb is not None:
                    q_pos_emb, k_pos_emb = rotary_pos_emb

                    b, h, s, d = q_pe.shape
                    q_pe = q_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
                    b, h, s, d = k_pe.shape
                    k_pe = k_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

                    if packed_seq_params is not None:
                        cu_seqlens_q = packed_seq_params.cu_seqlens_q
                        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                    else:
                        cu_seqlens_q = cu_seqlens_kv = None

                    # Megatron v0.12 judges multi_latent_attention when calculating RoPE, then slices input params row-wise alternately;
                    # Skip temporarily.
                    tmp_multi_latent_attention = self.config.multi_latent_attention
                    self.config.multi_latent_attention = False
                    q_pe = apply_rotary_pos_emb(q_pe, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                    k_pe = apply_rotary_pos_emb(k_pe, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)
                    self.config.multi_latent_attention = tmp_multi_latent_attention
                    del tmp_multi_latent_attention

                query = torch.cat([q_nope, q_pe], dim=-1)

                k_pe = k_pe.expand(k_pe.shape[0], k_pe.shape[1], query.shape[2], k_pe.shape[3])
                key = torch.cat([k_nope, k_pe], dim=-1)

                if (
                    self.use_flash_attn
                    and self.q_head_dim != self.v_head_dim
                    and not self.mla_fa_without_pad
                ):
                    if self.shape_order == "BNSD":
                        value = F.pad(value, [0, self.q_head_dim - self.v_head_dim])
                    else:
                        query = F.pad(query, [0, self.fa_padding_length - self.q_head_dim])
                        key = F.pad(key, [0, self.fa_padding_length - self.q_head_dim])
                        value = F.pad(value, [0, self.fa_padding_length - self.v_head_dim])

                # Do repeat KV to support GQA+Ulysses
                args = get_args()
                should_kv_repeat_before_uly = (
                    args.context_parallel_size > 1
                    and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]
                    and args.kv_head_repeat_before_uly_alltoall
                    )
                heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                if should_kv_repeat_before_uly and heads_per_gqa_group > 1:
                    key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                    value = value.repeat_interleave(heads_per_gqa_group, dim=2)

            # ==================================
            # core attention computation
            # ==================================
            attn_mask_type = AttnMaskType.causal
            if self.checkpoint_core_attention and self.training:
                core_attn_out = self._checkpointed_attention_forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    packed_seq_params=packed_seq_params,
                )
            else:
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    attention_mask,
                    attn_mask_type=attn_mask_type,
                    packed_seq_params=packed_seq_params,
                )

            if self.recompute_mla_up_proj and core_attn_out.requires_grad:
                self.recompute_mla_up_proj_ckpt.discard_output()
                core_attn_out.register_hook(self.recompute_mla_up_proj_ckpt.recompute)

            if packed_seq_params is not None:
                # reshape to same output shape as unpacked case
                # the shape transformation is: (t, np, hn) -> (t, b=1, h=np*hn)
                # t is the pack size = sum (sq_i)
                # note that batch is a dummy dimension in the packed case
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

            if self.use_flash_attn and not self.mla_fa_without_pad:
                core_attn_out = core_attn_out.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
                core_attn_out = core_attn_out.reshape(q_len, bsz, self.num_attention_heads_per_partition * self.v_head_dim)

            return core_attn_out


        if self.mla_zero_memory:
            self.mla_checkpoint_manager = CheckpointWithoutOutput()
            core_attn_out = self.mla_checkpoint_manager.checkpoint(mla_attention,
                                                                        False,
                                                                        hidden_states)
            if args.reset_position_ids:
                self.mla_checkpoint_manager.ctx.actual_len = get_actual_seq_len()
                self.mla_checkpoint_manager.ctx.position_id = get_position_ids()
        else:
            core_attn_out = mla_attention(hidden_states)

        # =================
        # Output. [sq, b, h]
        # =================
        if self.a2a_hooked_on_attention and core_attn_out.requires_grad:
            core_attn_out.register_hook(launch_async_all2all_hook)

        output, bias = self.linear_proj(core_attn_out)

        if self.mla_zero_memory:
            self.mla_checkpoint_manager.discard_output()
            if output.requires_grad:
                if args.reset_position_ids:
                    output.register_hook(recompute_mla(self.mla_checkpoint_manager))
                else:
                    output.register_hook(self.mla_checkpoint_manager.recompute)
        return output, bias


def recompute_mla(mla_checkpoint_manager):
    """
    recompute_mla when reset_position_ids is enabled.
    """
    def hook_fn(grad):
        actual_seq_len = getattr(mla_checkpoint_manager.ctx, "actual_len", None)
        position_ids = getattr(mla_checkpoint_manager.ctx, "position_id", None)
        change_pos_id = False
        if position_ids is not None:
            change_pos_id = True
            old_position_id = get_position_ids()
            set_position_ids(position_ids)
        change_seq_len = False
        if actual_seq_len is not None:
            change_seq_len = True
            old_actual_seq_len = get_actual_seq_len()
            set_actual_seq_len(actual_seq_len)

        mla_checkpoint_manager.recompute(grad)

        if change_pos_id:
            set_position_ids(old_position_id)
        if change_seq_len:
            set_actual_seq_len(old_actual_seq_len)

    return hook_fn


class LinearNoTP(torch.nn.Linear):
    def __init__(
        self,
        input_size,
        output_size,
        config,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            bias=kwargs.get('bias', True),
            dtype=config.params_dtype,
        )
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)
        self.config = config

        # Set fixed random seed for weight initialization
        current_seed = torch.random.initial_seed()
        torch.manual_seed(123)
        torch.nn.init.xavier_uniform_(self.weight)
        torch.random.manual_seed(current_seed)

    def forward(self, input_):
        if hasattr(self.weight, "quant_state"):
            output = bnb.matmul_4bit(input_, self.weight.t(), self.weight.quant_state, bias=self.bias)
        else:
            output = torch.matmul(input_, self.weight.t())
        return output

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        args = get_args()
        if hasattr(args, "qlora_save_dequantize") and args.qlora_save_dequantize and getattr(self.weight, "quant_state", None) is not None:
            self.weight = torch.nn.Parameter(bnb.functional.dequantize_4bit(self.weight.data, self.weight.quant_state))
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if getattr(self.weight, "quant_state", None) is not None:
            for k, v in self.weight.quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if any(['bitsandbytes' in i for i in state_dict.keys()]):  # is quantized linear
            qs_dict = {key: v for k, v in state_dict.items() if (key := k.replace(prefix, "")) != '_extra_state'}
            self.weight = bnb.nn.Params4bit.from_prequantized(
                data=qs_dict.get('weight'),
                quantized_stats={key.replace('weight.', ''): qs_dict[key] for key in qs_dict if key != 'weight'},
                requires_grad=False,
                device='npu')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
