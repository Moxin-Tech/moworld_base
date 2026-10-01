import torch
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.ops.swiglu import swiglu
from tests.ut_fsdp.utils.utils import judge_expression

def test_eager_swiglu_basic():
    inputs = torch.randn(2, 4, 8, device=torch.device(get_device_type()))
    output = swiglu(inputs, fused=False)
    judge_expression(output.shape == (2, 4, 4))
    judge_expression(not torch.isnan(output).any())
    judge_expression(not torch.isinf(output).any())


def test_fused_swiglu_basic():
    inputs = torch.randn(4, 16, 8, device=torch.device(get_device_type()))
    output = swiglu(inputs, fused=True)
    judge_expression(output.shape == (4, 16, 4))
    judge_expression(not torch.isnan(output).any())
    judge_expression(not torch.isinf(output).any())


def test_fused_vs_eager_consistency_for_swiglu():
    test_shapes = [
        (2, 8),
        (4, 16, 8),
        (1, 32, 16, 8),
        (3, 6, 12, 24),
    ]
    test_dims = [-1, 0, 1, 2]

    for shape in test_shapes:
        for dim in test_dims:
            if abs(dim) >= len(shape) or shape[dim] % 2 != 0:
                continue

            inputs = torch.randn(shape, device=torch.device(get_device_type()), dtype=torch.bfloat16)
            out_eager = swiglu(inputs, fused=False)
            out_fused = swiglu(inputs, fused=True)

            torch.testing.assert_close(out_fused, out_eager)
            judge_expression(not torch.isnan(out_fused).any())
            judge_expression(not torch.isinf(out_fused).any())
