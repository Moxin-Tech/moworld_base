import torch
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


def npu_group_gemm(x, weight, group_list):
    output = GmmFunction.apply(x, weight, group_list)
    return output
