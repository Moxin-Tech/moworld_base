import torch
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from tests.ut_fsdp.utils.utils import judge_expression


def test_fused_vs_eager_consistency_for_permute():
    device = torch.device(get_device_type())
    num_tokens = 10
    hidden_size = 32
    topk_list = [1, 2]
    num_experts = 8

    tokens = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    for topk in topk_list:
        indices = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device)
        indices = indices.squeeze(1)

        eager_result = permute(tokens, indices, fused=False)
        fused_result = permute(tokens, indices, fused=True)
        judge_expression(len(eager_result) == len(fused_result))
        torch.testing.assert_close(eager_result[0], fused_result[0])
        torch.testing.assert_close(eager_result[1], fused_result[1])
