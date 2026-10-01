# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional

import torch
import torch.distributed as dist
import torch_npu

from mindspeed_mm.models.common.communications import all_to_all_EP as all_to_all
from mindspeed_mm.models.common.gmm import npu_group_gemm


def fused_ep_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    hidden_states, unpermute_indices, post_dispatch_unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )

    intermediate_hidden_states = npu_group_gemm(hidden_states, fc1_weight, num_global_sum_tokens_per_local_expert)
    intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
    hidden_states = npu_group_gemm(
        intermediate_activations, fc2_weight, num_global_sum_tokens_per_local_expert
    )

    hidden_states = alltoall_combine(
        hidden_states,
        routing_weights,
        post_dispatch_unpermute_indices,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    if ep_group is None:
        ep_size = 1
        ep_rank = 0
    else:
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    num_local_experts = num_global_experts // ep_size

    num_local_tokens_per_expert = torch.bincount(selected_experts.view(-1), minlength=num_global_experts)

    if ep_group is None or ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    hidden_states, unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=input_splits, gather_sizes=output_splits)

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )

    _expert_ids_per_ep_rank = torch.arange(num_global_experts, dtype=torch.int32, device=hidden_states.device) % num_local_experts
    global_input_tokens_local_experts_indices = torch.repeat_interleave(_expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    hidden_states, post_dispatch_unpermute_indices = torch_npu.npu_moe_token_permute(hidden_states, global_input_tokens_local_experts_indices)

    return hidden_states, unpermute_indices, post_dispatch_unpermute_indices


def alltoall_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    post_dispatch_unpermute_indices: torch.Tensor,
    unpermute_indices: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )

    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, post_dispatch_unpermute_indices)

    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=output_splits, gather_sizes=input_splits)
    hidden_states = torch_npu.npu_moe_token_unpermute(hidden_states, unpermute_indices, probs=routing_weights)
    return hidden_states
