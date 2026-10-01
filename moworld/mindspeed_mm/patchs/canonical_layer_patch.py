# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Union
import warnings

from dataclasses import dataclass
import torch
import torch.nn.functional as F

import megatron
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.attention import SelfAttention, Attention, AttnMaskType, SelfAttentionSubmodules
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlSelfAttention, Qwen2vlVitSelfAttention

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class SplitQKVSelfAttentionSubmodules(SelfAttentionSubmodules):
    q_proj: Union[ModuleSpec, type] = None
    k_proj: Union[ModuleSpec, type] = None
    v_proj: Union[ModuleSpec, type] = None


@dataclass
class SplitUpGateMLPSubmodules:
    gate_proj: Union[ModuleSpec, type] = None
    up_proj: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class PatchSplitQKVSelfAttention(Qwen2vlSelfAttention):
    """Implementation of Splitting QKV Self-Attention Layer, which only rewrites the logic related to QKV projection."""

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SplitQKVSelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        # Invoke parent class initialization
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )

        # Remove the parent class's merged projection layer and replace it with independent Q/K/V projections.
        if hasattr(self, 'linear_qkv'):
            del self.linear_qkv

        # Build independent Q/K/V projection layers
        self.q_proj = build_module(
            submodules.q_proj,
            self.config.hidden_size,
            self.query_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='q',
        )

        self.k_proj = build_module(
            submodules.k_proj,
            self.config.hidden_size,
            self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='k',
        )

        self.v_proj = build_module(
            submodules.v_proj,
            self.config.hidden_size,
            self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='v',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """Rewrite the QKV tensor generation logic."""
        if key_value_states is not None:
            raise ValueError("Self-Attention does not support key_value_states")

        # Independent Calculation of Q/K/V
        query, query_bias = self.q_proj(hidden_states)  # [sq, b, query_projection_size]
        key, key_bias = self.k_proj(hidden_states)  # [sq, b, kv_projection_size]
        value, value_bias = self.v_proj(hidden_states)  # [sq, b, kv_projection_size]

        new_query_shape = query.size()[:-1] + (
            self.num_query_groups_per_partition * (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition),
            self.hidden_size_per_attention_head,
        )
        query = query.view(*new_query_shape)

        new_kv_shape = key.size()[:-1] + (
            self.num_query_groups_per_partition,
            self.hidden_size_per_attention_head
        )
        key = key.view(*new_kv_shape)
        value = value.view(*new_kv_shape)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


class PatchSplitGateUpMLP(MegatronModule):
    """Implementation of Splitting gate_proj and up_proj Layer"""

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SplitUpGateMLPSubmodules,
            is_expert: bool = False,
            input_size: Optional[int] = None,
    ):
        super().__init__(config=config)
        self.config: TransformerConfig = config

        self.input_size = input_size if input_size else self.config.hidden_size

        # If this is a gated linear unit we double the output width
        if is_expert and self.config.moe_ffn_hidden_size:
            # Experts read ffn_hidden_size from config.moe_ffn_hidden_size
            ffn_hidden_size = self.config.moe_ffn_hidden_size
        else:
            # Normal MLPs read ffn_hidden_size from config.ffn_hidden_size
            ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        # The output size of gate/up is half of ffn_size.
        split_hidden_size = int(ffn_hidden_size // 2)

        self.activation_func = self.config.activation_func

        # Build independent gate/up projection layers
        self.gate_proj = build_module(
            submodules.gate_proj,
            self.input_size,
            split_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='gate',
        )

        self.up_proj = build_module(
            submodules.up_proj,
            self.input_size,
            split_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='up',
        )

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name='fc2',
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Rewrite the forward propagation using independent gate_proj and up_proj. """
        # Calculate the outputs of the gate and up layers.
        gate_parallel, gate_bias = self.gate_proj(hidden_states)  # [s, b, ffn_hidden_size]
        up_parallel, up_bias = self.up_proj(hidden_states)  # [s, b, ffn_hidden_size]

        if self.config.add_bias_linear:
            bias_parallel = torch.cat([gate_bias, up_bias], dim=-1) if gate_bias is not None else None
        else:
            bias_parallel = None

        # Activation Function and Fusion Logic (Reusing the original logic but with inputs changed to the split tensors)
        if self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # Fused Weighted Swiglu (requires merging gate and up as inputs)
                    intermediate_combined = torch.cat([gate_parallel, up_parallel], dim=-1)
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_combined,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of swiglu with per_token_scale in MLP.")
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        # Integrate GEGLU (requires merging gate and up as input)
                        intermediate_combined = torch.cat([gate_parallel, up_parallel], dim=-1)
                        intermediate_parallel = bias_geglu_impl(intermediate_combined, bias_parallel)
                    elif self.config.add_bias_linear:
                        # Standard GELU (directly using up_proj output)
                        intermediate_parallel = bias_gelu_impl(up_parallel, up_bias)
                    else:
                        raise ValueError("Only support gated_linear_unit or add_bias_linear in gelu.")
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    # Integrate Swiglu (requires merging gate and up as inputs)
                    intermediate_combined = torch.cat([gate_parallel, up_parallel], dim=-1)
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_combined,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            # Non-fusion path: Manual calculation of activation
            if self.config.add_bias_linear and gate_bias is not None:
                gate_parallel = gate_parallel + gate_bias
                up_parallel = up_parallel + up_bias

            if self.config.gated_linear_unit:
                # Gated Mode：activation(gate) * up
                intermediate_parallel = self.activation_func(gate_parallel) * up_parallel
            else:
                # Normal Mode: Directly activate up
                intermediate_parallel = self.activation_func(up_parallel)

            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)

        output, output_bias = self.linear_fc2(intermediate_parallel)

        if per_token_scale is not None:
            if output_bias is not None:
                raise ValueError("Bias is not supported with per_token_scale")

        return output, output_bias

    def sharded_state_dict(
            self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Rewrite the shard state dictionary to adapt to independent gate_proj and up_proj. """
        sharded_state_dict = {}
        for name, module in self._modules.items():
            if name in ['gate_proj', 'up_proj']:
                sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
                sharded_state_dict.update(sub_sd)
            else:
                sub_sd = module.sharded_state_dict(f'{prefix}{name}.', sharded_offsets, metadata)
                sharded_state_dict.update(sub_sd)
        return sharded_state_dict


class PatchViTSelfAttention(Qwen2vlVitSelfAttention):
    """Implementation of non-interleaved QKV Self-Attention Layer in ViT, which only rewrites the logic related to QKV projection."""

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """Derives `query`, `key` and `value` tensors from `hidden_states` using non-interleaved weight"""
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        sq, b, h = hidden_states.shape

        query, key, value = (
            mixed_qkv.reshape(sq, b, 3, self.num_attention_heads_per_partition, -1).permute(2, 0, 1, 3, 4).unbind(0)
        )

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value


def _patch_get_mlp_module_spec(
        use_te: Optional[bool] = True,
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        fp8: Optional[str] = None,
        moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_patch_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_patch_mlp_module_spec(
        use_te: Optional[bool] = True,
        num_experts: Optional[int] = None,
        moe_grouped_gemm: Optional[bool] = False,
        fp8: Optional[str] = None,  # pylint: disable=unused-arguments
        moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Rewrite helper function to get patch solit gate/up module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            ' and will be removed soon. Please update your code accordingly.'
        )

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=PatchSplitGateUpMLP,
            submodules=SplitUpGateMLPSubmodules(
                gate_proj=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                up_proj=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec(
            use_te=use_te,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
