import unittest
from unittest.mock import Mock, patch
import torch
import numpy as np

from mindspeed_mm.utils.utils import dist_sort, EncoderBalanceComm
from tests.ut.utils import judge_expression


class TestDistSort(unittest.TestCase):
    """Test cases for dist_sort function"""

    def test_dist_sort_equal_distribution(self):
        """Test dist_sort when images are equally distributed"""
        image_num_list = np.array([4, 4, 4, 4])
        transfer, target = dist_sort(image_num_list)

        judge_expression(np.sum(transfer) == 0)
        judge_expression(all(t == 4 for t in target))

    def test_dist_sort_unequal_distribution(self):
        """Test dist_sort when images need redistribution"""
        image_num_list = np.array([8, 2, 6, 0])
        transfer, target = dist_sort(image_num_list)

        total_before = np.sum(image_num_list)
        total_after = np.sum(target)

        judge_expression(total_before == total_after)
        judge_expression(all(t == 4 for t in target))
        judge_expression(transfer.shape == (4, 4))
        judge_expression(np.sum(transfer) > 0)

    def test_dist_sort_with_remainder(self):
        """Test dist_sort when total images don't divide evenly"""
        image_num_list = np.array([7, 1, 3, 2])
        _, target = dist_sort(image_num_list)

        total_target = np.sum(target)
        judge_expression(total_target == 13)

        avg = 13 // 4
        judge_expression(all(t >= avg and t <= avg + 1 for t in target))


class TestEncoderBalanceComm(unittest.TestCase):
    """Test cases for EncoderBalanceComm autograd function"""

    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.world_size = 4
        self.rank = 0

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    def test_forward_no_transfer_needed(self, mock_world_size, mock_rank):
        """Test forward when no load balancing is needed"""
        mock_rank.return_value = self.rank
        mock_world_size.return_value = self.world_size

        mock_group = Mock()

        input_tensor = torch.randn(4, 64, dtype=torch.float32)

        transfer = np.zeros((4, 4))
        target = [4, 4, 4, 4]

        result = EncoderBalanceComm.apply(input_tensor, mock_group, (transfer, target))

        judge_expression(torch.equal(result, input_tensor))

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    @patch('torch.distributed.all_to_all')
    def test_forward_with_transfer(self, mock_all_to_all, mock_world_size, mock_rank):
        """Test forward when load balancing is needed"""
        mock_rank.return_value = self.rank
        mock_world_size.return_value = self.world_size

        mock_group = Mock()

        input_tensor = torch.randn(8, 64, dtype=torch.float32)

        transfer = np.array([[0, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        target = [4, 4, 4, 4]

        def mock_all_to_all_side_effect(recv_list, send_list, group=None):
            for i, send_tensor in enumerate(send_list):
                if i < len(recv_list) and send_tensor.numel() > 0:
                    recv_list[i].resize_(send_tensor.shape)
                    recv_list[i].copy_(send_tensor)

        mock_all_to_all.side_effect = mock_all_to_all_side_effect

        result = EncoderBalanceComm.apply(input_tensor, mock_group, (transfer, target))

        judge_expression(isinstance(result, torch.Tensor))
        judge_expression(result.dim() == 2)
        judge_expression(result.shape[1] == 64)

    def test_forward_skip_mode(self):
        """Test forward in skip mode"""
        mock_group = Mock()

        input_tensor = torch.randn(4, 64, dtype=torch.float32)

        transfer = np.zeros((4, 4))
        target = [4, 4, 4, 4]

        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.get_world_size', return_value=4):

            result, returned_transfer = EncoderBalanceComm.apply(
                input_tensor, mock_group, (transfer, target), False, True
            )

            judge_expression(torch.equal(result, input_tensor))

    @patch('torch.distributed.get_rank')
    @patch('torch.distributed.get_world_size')
    def test_backward_no_transfer(self, mock_world_size, mock_rank):
        """Test backward when no transfer was needed"""
        mock_rank.return_value = self.rank
        mock_world_size.return_value = self.world_size

        mock_group = Mock()

        input_tensor = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)

        transfer = np.zeros((4, 4))
        target = [4, 4, 4, 4]

        result = EncoderBalanceComm.apply(input_tensor, mock_group, (transfer, target))

        grad_output = torch.randn_like(result)
        result.backward(grad_output)

        judge_expression(input_tensor.grad is not None)
        judge_expression(torch.equal(input_tensor.grad, grad_output))

    def test_nopadding_flag(self):
        """Test the nopadding flag functionality"""
        mock_group = Mock()

        input_tensor = torch.randn(6, 64, dtype=torch.float32)

        transfer = np.zeros((4, 4))
        target = [4, 4, 4, 4]

        with patch('torch.distributed.get_rank', return_value=0), \
             patch('torch.distributed.get_world_size', return_value=4):

            result = EncoderBalanceComm.apply(
                input_tensor, mock_group, (transfer, target), False, False
            )

            judge_expression(result.shape[0] <= target[0] or np.sum(transfer) > 0)
