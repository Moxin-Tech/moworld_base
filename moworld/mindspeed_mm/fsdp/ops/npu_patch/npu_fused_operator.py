# Copyright 2025 Bytedance Ltd. and/or its affiliates
import torch
import torch_npu
import torch.nn.functional as F
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.ops.swiglu import swiglu


_orig_gelu = F.gelu


def apply_gelu_npu(input_tensor, approximate='none'):
    """
    Wrap npu_gelu to match the F.gelu interface
    """
    device_type = input_tensor.device.type if hasattr(input_tensor, 'device') else 'cpu'

    valid_approximates = ['none', 'tanh']
    if approximate not in valid_approximates:
        import warnings
        warnings.warn(f"NPU GELU does not support approximate='{approximate}'. "f"Using approximate='none' instead.")
        approximate = 'none'

    if device_type == 'npu':
        return torch_npu.npu_gelu(input_tensor, approximate=approximate)
    else:
        return _orig_gelu(input_tensor, approximate=approximate)


# This api can improve performance on ASCEND NPU
def rms_norm_forward_npu(self, x):
    """NPU optimized implementation for RMSNorm."""
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def apply_transformers_rope_half_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for RoPE(half mode)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q.shape[0] * q.shape[1] <= 128:
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    else:
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_transformers_vision_rope_half_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """NPU optimized implementation for vision RoPE(half mode)."""
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q_4d = q.unsqueeze(0).float().contiguous()
    k_4d = k.unsqueeze(0).float().contiguous()
    cos_4d = cos.unsqueeze(0).unsqueeze(2).float()
    sin_4d = sin.unsqueeze(0).unsqueeze(2).float()
    q_embed_4d = torch_npu.npu_rotary_mul(q_4d, cos_4d, sin_4d)
    k_embed_4d = torch_npu.npu_rotary_mul(k_4d, cos_4d, sin_4d)
    q_embed = q_embed_4d.squeeze(0).to(orig_q_dtype)
    k_embed = k_embed_4d.squeeze(0).to(orig_k_dtype)
    q_embed = q_embed.reshape(orig_q_shape)
    k_embed = k_embed.reshape(orig_k_shape)
    return q_embed, k_embed


def silu_forward_npu(self, hidden_states):
    """NPU optimized implementation for SiLU in 'forward' func in MLP layer."""
    gate_up = torch.cat([self.gate_proj(hidden_states), self.up_proj(hidden_states)], dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def fused_moe_forward_npu(
    self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
) -> torch.Tensor:
    permuted_hidden_states, row_ids_map = permute(hidden_states, router_indices.to(torch.int32), fused=True)
    tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
    intermediate_hidden_states = grouped_matmul(permuted_hidden_states, self.gate_up_proj, tokens_per_expert, fused=True)
    intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=True)
    output = grouped_matmul(intermediate_activations, self.down_proj, tokens_per_expert, fused=True)
    next_states = unpermute(output, row_ids_map, probs=routing_weights, fused=True)
    return next_states


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
