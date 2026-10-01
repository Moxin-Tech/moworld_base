import os
import pytest
import torch
import inspect
from unittest.mock import patch, MagicMock


from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.loss.loss_func import (
    calculate_chunk_size,
    get_loss_func_params,
    build_loss_func,
)
from tests.ut_fsdp.utils.utils import judge_expression

@pytest.fixture(autouse=True)
def mock_parallel_state():
    with patch("mindspeed_mm.fsdp.loss.loss_func.get_parallel_state") as mock_ps_func:
        mock_ps = MagicMock()
        mock_ps.is_cp_enable.return_value = False
        mock_ps_func.return_value = mock_ps
        yield mock_ps_func


class TestCalculateChunkSize:
    """
    Test calculate_chunk_size function
    """

    def test_normal_case_power_of_two(self):
        """Test normal case where max_possible_chunk_size is a power of two"""
        result = calculate_chunk_size(batch_size=2, total_size=4096)
        judge_expression(result == 2048)  # 4096 // 2 = 2048 (2^11)

    def test_normal_case_not_power_of_two(self):
        """Test normal case where max_possible_chunk_size is not a power of two"""
        result = calculate_chunk_size(batch_size=3, total_size=4096)
        # 4096 // 3 = 1365, largest power of two <= 1365 is 1024 (2^10)
        judge_expression(result == 1024)

    def test_batch_size_equals_total_size(self):
        """Test when batch_size equals total_size"""
        result = calculate_chunk_size(batch_size=4096, total_size=4096)
        judge_expression(result == 1)

    def test_batch_size_exceeds_total_size(self):
        """Test when batch_size exceeds total_size"""
        result = calculate_chunk_size(batch_size=8192, total_size=4096)
        judge_expression(result == 1)

    def test_zero_batch_size(self):
        """Test when batch_size is zero"""
        result = calculate_chunk_size(batch_size=0, total_size=4096)
        judge_expression(result == 1)

    def test_zero_total_size(self):
        """Test when total_size is zero"""
        result = calculate_chunk_size(batch_size=2, total_size=0)
        judge_expression(result == 1)

    def test_negative_values(self):
        """Test negative input values"""
        result1 = calculate_chunk_size(batch_size=-1, total_size=4096)
        result2 = calculate_chunk_size(batch_size=2, total_size=-1)
        judge_expression(result1 == 1)
        judge_expression(result2 == 1)


class TestGetLossFuncParamsWithTotalChunkSize:
    """
    Test get_loss_func_params with total_chunk_size parameter
    """

    device = torch.device(get_device_type())

    def test_total_chunk_size_none_uses_passed_chunk_size(self):
        """Test when total_chunk_size is None, passed chunk_size is used"""
        labels = torch.randint(0, 1000, (2, 1024), device=self.device)

        result1 = get_loss_func_params(
            labels=labels,
            loss_type="default",
            ignore_index=-100,
            chunk_size=None,
        )
        result2 = get_loss_func_params(
            labels=labels,
            loss_type="default",
            ignore_index=-100,
            chunk_size=1024,
        )

        judge_expression(result1[0]["chunk_size"] is None)
        judge_expression(result2[0]["chunk_size"] == 1024)

    def test_total_chunk_size_with_different_batch_sizes(self):
        """Test chunk_size calculation with different batch sizes"""
        batch_sizes = [1, 2, 4, 8]
        total_chunk_size = 4096
        expected_chunk_sizes = [4096, 2048, 1024, 512]

        for batch_size, expected in zip(batch_sizes, expected_chunk_sizes):
            labels = torch.randint(0, 1000, (batch_size, 1024), device=self.device)

            result = get_loss_func_params(
                labels=labels,
                loss_type="default",
                ignore_index=-100,
                chunk_size=1024,
                total_chunk_size=total_chunk_size
            )

            judge_expression(result[0]["chunk_size"] == expected)


class TestBuildLossFuncBranch:
    """
    Test that build_loss_func enters the chunk loss branch when:
    1. chunk_size is not None, OR
    2. total_chunk_size is not None
    """
    device = torch.device(get_device_type())
    hidden_dim = 128
    vocab_size = 1000

    def _is_chunk_loss_func(self, loss_func):
        """
        Determine if the returned loss function is the chunk loss version.
        Chunk loss function has signature: loss_func(hidden_states, head_weight, head_bias, labels=None)
        Non-chunk loss function has signature: loss_func(logits, labels=None, vocab_size=None)
        """
        sig = inspect.signature(loss_func)
        params = list(sig.parameters.keys())

        # Chunk loss: (hidden_states, head_weight, head_bias, labels=None)
        # Non-chunk loss: (logits, labels=None, vocab_size=None)
        if params[:3] == ['hidden_states', 'head_weight', 'head_bias']:
            return True
        elif params[:1] == ['logits']:
            return False
        else:
            # try to call with chunk loss signature
            try:
                # Create dummy inputs
                hidden_states = torch.randn(1, 10, self.hidden_dim, device=self.device)
                head_weight = torch.randn(self.vocab_size, self.hidden_dim, device=self.device)
                head_bias = None
                labels = torch.randint(0, self.vocab_size, (1, 10), device=self.device)

                # This should work for chunk loss
                loss_func(hidden_states, head_weight, head_bias, labels)
                return True
            except TypeError:
                return False

    def test_chunk_size_non_zero_enters_chunk_branch(self):
        """Test that non-zero chunk_size enters chunk loss branch"""
        loss_func = build_loss_func(
            loss_type="default",
            chunk_size=1024,  # Non-zero
        )

        judge_expression(self._is_chunk_loss_func(loss_func))

    def test_total_chunk_size_non_zero_enters_chunk_branch(self):
        """Test that non-zero total_chunk_size enters chunk loss branch"""
        loss_func = build_loss_func(
            loss_type="default",
            chunk_size=None,
            total_chunk_size=2048  # Non-zero
        )

        judge_expression(self._is_chunk_loss_func(loss_func))

    def test_chunk_size_none_enters_non_chunk_branch(self):
        """Test that none chunk_size and None total_chunk_size enters non-chunk branch"""
        loss_func = build_loss_func(
            loss_type="default",
            chunk_size=None,
        )

        judge_expression(not self._is_chunk_loss_func(loss_func))
