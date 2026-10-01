import mindspeed.megatron_adaptor
import torch
import torch_npu

from mindspeed_mm.models.common.embeddings.pos_embeddings import RoPE3DSORA, RoPE3DStepVideo
from tests.ut.utils import judge_expression


class TestRoPE3DSORA:
    def test_init_rope3dsora(self):
        rope = RoPE3DSORA(30000, freq=10000.0, interpolation_scale=(1, 1, 1))
        judge_expression(isinstance(rope, RoPE3DSORA))

    def test_check_type(self):
        rope = RoPE3DSORA(30000, freq=10000.0, interpolation_scale=(1, 1, 1))
        input_tensor = torch.tensor(3.14, requires_grad=True)
        result = rope.check_type(input_tensor)
        judge_expression(isinstance(result, float))

    def test_rotate_half(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        output_tensor = RoPE3DSORA.rotate_half(x)
        expected_tensor = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        judge_expression(torch.all(output_tensor == expected_tensor))

    def test_apply_rope1d_broadcasting(self):
        tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Shape [2, 2]
        freq = torch.tensor([0.1, 0.2])                  # Shape [2]
        rope = RoPE3DSORA(30000, freq=10000.0, interpolation_scale=(1, 1, 1))

        result = rope.apply_rope1d(tokens, freq)
        # Check output shape matches tokens
        judge_expression(result.shape == tokens.shape)

    def test_get_position(self):
        rope = RoPE3DSORA(30000, freq=10000.0, interpolation_scale=(1, 1, 1))
        b, t, h, w = 2, 3, 4, 5
        poses = rope.get_position(b, t, h, w, 'cpu')

        # Verify output structure
        judge_expression(isinstance(poses, tuple))


class TestRoPE3DStepVideo:
    def test_init_rope3dstepvideo(self):
        rope = RoPE3DStepVideo([64, 32, 32], freq=10000.0)
        judge_expression(isinstance(rope, RoPE3DStepVideo))
