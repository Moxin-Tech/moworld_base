from typing import List, Optional

import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


class GmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul([x], [weight], bias=None, group_list=group_list,
                                                  split_item=2, group_type=0, group_list_type=1)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul([grad_output], [weight], bias=None, group_list=group_list,
                                                  split_item=2, group_type=0, group_list_type=1)[0]

        grad_weight = torch_npu.npu_grouped_matmul([input_tensor.T], [grad_output], bias=None, group_list=group_list,
                                                   split_item=3, group_type=2, group_list_type=1)[0]

        return grad_input, grad_weight, None


def fused_group_gemm(x, weight, group_list):
    output = GmmFunction.apply(x, weight, group_list)
    return output


def eager_grouped_matmul(x, weight, group_list):
    """
    Grouped matrix multiplication that handles two weight tensor formats.

    Args:
        inputs: Tensor of shape [batch_size, input_dim]
        m_split: Tensor of group sizes that sum to batch_size
        weights: Weight tensor of either:
                 Format 1: [num_groups, input_dim, output_dim] - ready for matmul
                 Format 2: [num_groups, output_dim, input_dim] - needs transpose

    Returns:
        Tensor of shape [batch_size, output_dim]
    """
    inputs, m_split, weights = x, group_list, weight
    batch_size, input_dim = inputs.shape

    # Automatically detect and adjust weight format
    # Check if second dimension matches input dimension (Format 1)
    if weights.shape[1] == input_dim:
        # Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[2]
        # No transformation needed - weights are already in correct format
    else:
        # Format 2: [num_groups, output_dim, input_dim]
        # Transpose to convert to Format 1: [num_groups, input_dim, output_dim]
        output_dim = weights.shape[1]
        weights = weights.transpose(1, 2)

    # Initialize output tensor
    output_shape = (batch_size, output_dim)
    final_hidden_states = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)

    # Calculate group boundaries from cumulative sum
    group_list = [0] + torch.cumsum(m_split, dim=0).tolist()

    # Process each group separately
    for i in range(len(group_list) - 1):
        start_idx = group_list[i]
        end_idx = group_list[i + 1]

        # Matrix multiplication for current group
        # inputs[start_idx:end_idx, :] has shape [group_size, input_dim]
        # weights[i] has shape [input_dim, output_dim] (after format normalization)
        final_hidden_states[start_idx:end_idx, :] = torch.matmul(
            inputs[start_idx:end_idx, :],
            weights[i]
        )

    return final_hidden_states


def grouped_matmul(x, weight, group_list, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_group_gemm(x, weight, group_list)
    else:
        return eager_grouped_matmul(x, weight, group_list)
