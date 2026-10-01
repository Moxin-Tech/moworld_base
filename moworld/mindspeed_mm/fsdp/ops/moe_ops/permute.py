import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


def eager_permute(tokens, indices, num_out_tokens=None):
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True).to(indices_dtype)
    if num_out_tokens:
        permuted_tokens = permuted_tokens[:num_out_tokens]
        sorted_indices1 = sorted_indices1[:num_out_tokens]
    return permuted_tokens, sorted_indices1


def fused_permute(tokens, indices, num_out_tokens=None):
    return torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=num_out_tokens)


def permute(tokens, indices, num_out_tokens=None, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_permute(tokens, indices, num_out_tokens)
    else:
        return eager_permute(tokens, indices)
