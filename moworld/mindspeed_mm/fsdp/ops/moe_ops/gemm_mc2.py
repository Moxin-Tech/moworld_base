# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


class AllToAllGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)
        group_list_tensor = recv_counts.reshape(ep_world_size, -1).sum(dim=0)

        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

        output, shared_output, permute_output = torch_npu.npu_alltoallv_gmm(inputs, weights, hcomm, ep_world_size,
                                                                           send_counts, recv_counts, mm_x=shared_inputs,
                                                                           mm_weight=shared_weight, permute_out_flag=True)

        ctx.save_for_backward(weights, shared_inputs, shared_weight, permute_output)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list_tensor = group_list_tensor
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        weights, shared_inputs, shared_weight, permute_output = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list_tensor = ctx.group_list_tensor

        inputs_grad, shared_inputs_grad = torch_npu.npu_gmm_alltoallv(output_grad, weights, hcomm, ep_world_size,
                                                                     recv_counts, send_counts, mm_x=shared_output_grad,
                                                                     mm_weight=shared_weight, trans_gmm_weight=True)

        weights_grad = torch_npu.npu_grouped_matmul([permute_output.T], [output_grad], bias=None, group_list=group_list_tensor,
                                                    split_item=3, group_type=2, group_list_type=1)[0]

        shared_weight_grad = None if shared_inputs is None else torch.matmul(shared_inputs.T, shared_output_grad)
        return inputs_grad, weights_grad, None, None, None, shared_inputs_grad, shared_weight_grad


class GroupedMatmulAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)
        group_list_tensor = send_counts.reshape(ep_world_size, -1).sum(dim=0)

        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

        output, shared_output = torch_npu.npu_gmm_alltoallv(inputs, weights, hcomm, ep_world_size, send_counts,
                                                        recv_counts, mm_x=shared_inputs, mm_weight=shared_weight)

        ctx.save_for_backward(inputs, weights, shared_inputs, shared_weight)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list_tensor = group_list_tensor
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        inputs, weights, shared_inputs, shared_weight = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list_tensor = ctx.group_list_tensor


        inputs_grad, shared_inputs_grad, permute_grad = torch_npu.npu_alltoallv_gmm(output_grad, weights, hcomm, ep_world_size,
                                                                        recv_counts, send_counts, mm_x=shared_output_grad,
                                                                        mm_weight=shared_weight, permute_out_flag=True,
                                                                        trans_gmm_weight=True)

        weights_grad = torch_npu.npu_grouped_matmul([inputs.T], [permute_grad], bias=None, group_list=group_list_tensor,
                                                    split_item=3, group_type=2, group_list_type=1)[0]


        shared_weight_grad = None if shared_inputs is None else torch.matmul(shared_inputs.T, shared_output_grad)
        return inputs_grad, weights_grad, None, None, None, shared_inputs_grad, shared_weight_grad


# send_counts: [num_global_experts], recv_counts: [num_global_experts]
def all2all_grouped_matmul(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None):
    output = AllToAllGroupedMatmul.apply(inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]


def grouped_matmul_all2all(inputs, weights, group, send_counts, recv_counts, shared_inputs=None, shared_weight=None):
    output = GroupedMatmulAllToAll.apply(inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight)
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]
