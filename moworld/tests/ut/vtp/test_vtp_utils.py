# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP utility functions."""

import math
from unittest.mock import patch, MagicMock

import mindspeed.megatron_adaptor
import pytest
import torch

from mindspeed_mm.patchs.layerwise_disaggregated_training import utils_patch as mod
from tests.ut.utils import judge_expression

_PATCH_CUDA = patch("torch.cuda.current_device", return_value="cpu")
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/NPU required"
)


class TestAllreduceModelParallel:

    def test_vtp_enabled_calls_vtp_allreduce(self):
        with patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "vtp_allreduce") as mock_ar:
            tensor = torch.tensor([1.0])
            mod._ldt_allreduce_model_parallel(tensor, op=torch.distributed.ReduceOp.SUM, group=MagicMock())
            mock_ar.assert_called_once_with(tensor, op=torch.distributed.ReduceOp.SUM)

    def test_vtp_disabled_calls_standard_allreduce(self):
        with patch.object(mod, "is_vtp_enabled", return_value=False), \
             patch("torch.distributed.all_reduce") as mock_ar:
            tensor = torch.tensor([1.0])
            group = MagicMock()
            mod._ldt_allreduce_model_parallel(tensor, op=torch.distributed.ReduceOp.MAX, group=group)
            mock_ar.assert_called_once_with(tensor, op=torch.distributed.ReduceOp.MAX, group=group)


class TestLdtReduceMaxStat:

    def test_none_input_returns_none(self):
        with _PATCH_CUDA, \
             patch.object(mod, "_ldt_allreduce_model_parallel"), \
             patch.object(mod.mpu, "get_tensor_model_parallel_group", return_value=MagicMock()):
            result = mod.ldt_reduce_max_stat_across_model_parallel_group(None)
            judge_expression(result is None)

    def test_valid_stat_returns_value(self):
        def set_tensor(tensor, op, group=None):
            tensor.fill_(5.0)
        with _PATCH_CUDA, \
             patch.object(mod, "_ldt_allreduce_model_parallel", side_effect=set_tensor), \
             patch.object(mod.mpu, "get_tensor_model_parallel_group", return_value=MagicMock()):
            result = mod.ldt_reduce_max_stat_across_model_parallel_group(3.0)
            judge_expression(result == 5.0)


class TestLdtLogicalAnd:

    def test_true_input(self):
        with _PATCH_CUDA, \
             patch.object(mod, "_ldt_allreduce_model_parallel"), \
             patch.object(mod.mpu, "get_tensor_model_parallel_group", return_value=MagicMock()):
            result = mod.ldt_logical_and_across_model_parallel_group(True)
            judge_expression(result is True)

    def test_false_input(self):
        with _PATCH_CUDA, \
             patch.object(mod, "_ldt_allreduce_model_parallel"), \
             patch.object(mod.mpu, "get_tensor_model_parallel_group", return_value=MagicMock()):
            result = mod.ldt_logical_and_across_model_parallel_group(False)
            judge_expression(result is False)


class TestLdtGetGradNormFp32:

    def test_single_tensor_wrapped_to_list(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=False), \
             patch("torch.distributed.all_reduce"), \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=None), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            result = mod.ldt_get_grad_norm_fp32(torch.tensor([3.0, 4.0]), norm_type=float('inf'))
            judge_expression(isinstance(result, float))

    def test_norm_type_inf(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=False), \
             patch("torch.distributed.all_reduce"), \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=None), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            result = mod.ldt_get_grad_norm_fp32([torch.tensor([3.0, -4.0])], norm_type=float('inf'))
            judge_expression(math.isclose(result, 4.0))

    def test_norm_type_2(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "_ldt_allreduce_model_parallel") as mock_ar, \
             patch("torch.distributed.all_reduce"), \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=None), \
             patch.object(mod, "get_vdp_size", return_value=1), \
             patch.object(mod.mpu, "is_pipeline_first_stage", return_value=True), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            result = mod.ldt_get_grad_norm_fp32([torch.tensor([3.0, 4.0])], norm_type=2)
            judge_expression(isinstance(result, float))
            judge_expression(mock_ar.call_args[1]['op'] == torch.distributed.ReduceOp.SUM)

    def test_norm_type_other(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "_ldt_allreduce_model_parallel"), \
             patch("torch.distributed.all_reduce"), \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=None), \
             patch.object(mod, "get_vdp_size", return_value=1), \
             patch.object(mod.mpu, "is_pipeline_first_stage", return_value=True), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            result = mod.ldt_get_grad_norm_fp32([torch.tensor([3.0, 4.0])], norm_type=3)
            judge_expression(isinstance(result, float))

    def test_with_data_parallel_group(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=False), \
             patch("torch.distributed.all_reduce") as mock_dist_ar, \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=MagicMock()), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            mod.ldt_get_grad_norm_fp32([torch.tensor([3.0, -4.0])], norm_type=float('inf'))
            mock_dist_ar.assert_called()
            judge_expression(mock_dist_ar.call_count >= 1)

    def test_empty_grads_norm_type_2(self):
        with _PATCH_CUDA, \
             patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "_ldt_allreduce_model_parallel"), \
             patch("torch.distributed.all_reduce"), \
             patch.object(mod, "get_data_parallel_group_if_dtensor", return_value=None), \
             patch.object(mod, "get_vdp_size", return_value=1), \
             patch.object(mod.mpu, "is_pipeline_first_stage", return_value=True), \
             patch.object(mod, "to_local_if_dtensor", side_effect=lambda x: x):
            result = mod.ldt_get_grad_norm_fp32([], norm_type=2)
            judge_expression(isinstance(result, float))


class TestLdtVdpBarrierWrapper:

    def test_vtp_no_group_uses_hierarchical(self):
        with patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "vtp_hierarchical_barrier") as mock_barrier:
            wrapper = mod.ldt_vdp_barrier_wrapper(MagicMock())
            result = wrapper(group=None)
            mock_barrier.assert_called_once()
            judge_expression(result is None)

    def test_vtp_with_group_uses_original(self):
        original = MagicMock(return_value="ok")
        with patch.object(mod, "is_vtp_enabled", return_value=True), \
             patch.object(mod, "is_vdp_enabled", return_value=False):
            wrapper = mod.ldt_vdp_barrier_wrapper(original)
            result = wrapper(group=MagicMock())
            original.assert_called_once()
            judge_expression(result == "ok")

    def test_non_vtp_uses_original(self):
        original = MagicMock(return_value="ok")
        with patch.object(mod, "is_vtp_enabled", return_value=False), \
             patch.object(mod, "is_vdp_enabled", return_value=False):
            wrapper = mod.ldt_vdp_barrier_wrapper(original)
            wrapper(group=None)
            original.assert_called_once_with(group=None)


class TestVtpAllGatherWrapper:

    def test_vtp_no_group_copies(self):
        original = MagicMock(return_value=None)
        with patch.object(mod, "is_vtp_enabled", return_value=True):
            wrapper = mod.vtp_all_gather_into_tensor_wrapper(original)
            output = torch.zeros(4)
            input_t = torch.ones(4)
            result = wrapper(output, input_t, group=None)
            judge_expression(result is None)
            original.assert_called_once_with(output, input_t, group=None, async_op=False)

    def test_vtp_with_group_uses_original(self):
        original = MagicMock(return_value="ok")
        with patch.object(mod, "is_vtp_enabled", return_value=True):
            wrapper = mod.vtp_all_gather_into_tensor_wrapper(original)
            wrapper(torch.zeros(4), torch.ones(4), group=MagicMock())
            original.assert_called_once()

    def test_non_vtp_uses_original(self):
        original = MagicMock(return_value="ok")
        with patch.object(mod, "is_vtp_enabled", return_value=False):
            wrapper = mod.vtp_all_gather_into_tensor_wrapper(original)
            wrapper(torch.zeros(4), torch.ones(4), group=None, async_op=True)
            original.assert_called_once()
