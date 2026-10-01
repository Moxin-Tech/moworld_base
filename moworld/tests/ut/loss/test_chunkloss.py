import torch
import torch_npu

from mindspeed_mm.models.common.chunkloss import (
    chunk_loss,
    calculate_lm_loss
)
from tests.ut.utils import judge_expression


class TestChunkLoss:
    """
    Test ChunkLoss
    """

    device = "npu"
    dtype = torch.bfloat16

    micro_batch_size = 2
    grad_acc = 2
    seq_len = 8192
    chunk_size = 1024
    hidden_dim = 4096
    vocab_size = 151674
    mask_len = 200

    inputs = []
    shift_labels = []
    hidden_states = []
    loss_masks = []
    for _ in range(grad_acc):
        input = torch.rand(micro_batch_size, seq_len, hidden_dim, requires_grad=True, dtype=dtype, device=device)
        label = torch.randint(vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device)
        label[:, -200:] = -100
        shift_label = label[:, 1:].contiguous()
        hidden_state = input[:, :-1].contiguous()
        loss_mask = shift_label > -1
        inputs.append(input)
        shift_labels.append(shift_label)
        hidden_states.append(hidden_state)
        loss_masks.append(loss_mask)

    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, dtype=dtype).to(device)

    @staticmethod
    def _judge_result(no_chunk_forward, chunk_forward, no_chunk_grad, chunk_grad):
        judge_expression(torch.allclose(no_chunk_forward, chunk_forward, rtol=1e-5, atol=1e-6))
        judge_expression(torch.allclose(no_chunk_grad, chunk_grad, rtol=1e-4, atol=1e-5))

    def _loss_forward_backward_per_step(self, hidden_state, shift_label, alpha, reduction):
        no_chunk_forward, _ = calculate_lm_loss(
            hidden_states=hidden_state,
            head_weight=self.lm_head.weight,
            shift_labels=shift_label,
            alpha=alpha,
            ignore_index=-100,
            reduction=reduction
        )
        no_chunk_forward.backward()
        return no_chunk_forward

    def _chunk_loss_forward_backward_per_step(self, hidden_state, shift_label, alpha, reduction):
        chunk_labels = torch.split(shift_label, self.chunk_size, dim=1)
        loss_ctx_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": -100,
                "reduction": reduction,
                "alpha": alpha
            }
            for i in range(len(chunk_labels))
        ]

        chunk_forward = chunk_loss(
            hidden_states=hidden_state,
            head_weight=self.lm_head.weight,
            head_bias=None,
            loss_forward=calculate_lm_loss,
            loss_kwargs_chunks=loss_ctx_kwargs,
            chunk_size=self.chunk_size
        )

        chunk_forward.backward()
        return chunk_forward

    def _loss_forward_backward(self, alphas, reductions, per_step_func):
        """no chunk"""
        accumulated_forward = 0
        for i in range(self.grad_acc):
            loss_forward = per_step_func(
                self.hidden_states[i],
                self.shift_labels[i],
                alpha=alphas[i],
                reduction=reductions[i]
            )
            accumulated_forward += loss_forward

        grad = self.lm_head.weight.grad
        # reset grad
        self.lm_head.weight.grad = None

        return accumulated_forward, grad

    def test_default_vlm_loss(self):
        alphas = [self.loss_masks[i].sum() for i in range(self.grad_acc)]
        reductions = ["sum"] * self.grad_acc
        no_chunk_forward, no_chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._loss_forward_backward_per_step
        )
        chunk_forward, chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._chunk_loss_forward_backward_per_step
        )
        self._judge_result(no_chunk_forward, chunk_forward, no_chunk_grad, chunk_grad)

    def test_per_sample_vlm_loss(self):
        alphas = [self.loss_masks[i].sum(1) * self.loss_masks[i].shape[0] for i in range(self.grad_acc)]
        reductions = ["none"] * self.grad_acc
        no_chunk_forward, no_chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._loss_forward_backward_per_step
        )
        chunk_forward, chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._chunk_loss_forward_backward_per_step
        )
        self._judge_result(no_chunk_forward, chunk_forward, no_chunk_grad, chunk_grad)

    def test_per_token_vlm_loss(self):
        alphas = [sum([self.loss_masks[i].sum() for i in range(self.grad_acc)])] * self.grad_acc
        reductions = ["none"] * self.grad_acc
        no_chunk_forward, no_chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._loss_forward_backward_per_step
        )
        chunk_forward, chunk_grad = self._loss_forward_backward(
            alphas=alphas,
            reductions=reductions,
            per_step_func=self._chunk_loss_forward_backward_per_step
        )
        self._judge_result(no_chunk_forward, chunk_forward, no_chunk_grad, chunk_grad)
