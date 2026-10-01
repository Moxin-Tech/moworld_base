# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP-specific functions in schedules_patch."""

import os
from unittest.mock import patch, MagicMock

import mindspeed.megatron_adaptor
import torch

from mindspeed_mm.patchs.layerwise_disaggregated_training import parallel_state_patch as mod
from tests.ut.utils import judge_expression

mpu = mod.mpu


class TestVtpAllreduce:

    def test_full_3_step(self):
        with patch.object(mpu, "get_tensor_model_parallel_world_size", return_value=2), \
             patch.object(mpu, "get_tensor_model_parallel_group", return_value=MagicMock()), \
             patch.object(mpu, "get_pipeline_model_parallel_group", return_value=MagicMock()), \
             patch.object(mod.torch.distributed, "all_reduce") as mock_ar, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast, \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod, "get_vtp_stage_ranks", return_value=[[0, 1], [2, 3]]), \
             patch.object(mod, "get_vtp_my_stage_idx", return_value=0):
            mod.vtp_allreduce(torch.tensor([1.0]))
            judge_expression(mock_ar.call_count == 2)
            mock_bcast.assert_called_once()

    def test_tp1_no_intra_group(self):
        with patch.object(mpu, "get_tensor_model_parallel_world_size", return_value=1), \
             patch.object(mpu, "get_pipeline_model_parallel_group", return_value=MagicMock()), \
             patch.object(mod.torch.distributed, "all_reduce") as mock_ar, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast, \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mpu, "get_tensor_model_parallel_group", return_value=None):
            mod.vtp_allreduce(torch.tensor([1.0]))
            mock_ar.assert_called_once()
            mock_bcast.assert_not_called()

    def test_non_rank0_skips_pp(self):
        with patch.object(mpu, "get_tensor_model_parallel_world_size", return_value=1), \
             patch.object(mod.torch.distributed, "all_reduce") as mock_ar, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast, \
             patch.object(mod, "is_vtp_stage_rank0", return_value=False), \
             patch.object(mpu, "get_tensor_model_parallel_group", return_value=None):
            mod.vtp_allreduce(torch.tensor([1.0]))
            mock_ar.assert_not_called()
            mock_bcast.assert_not_called()


class TestVtpHierarchicalBarrier:

    def test_full_3_step_barrier(self):
        with patch.object(mpu, "get_tensor_model_parallel_world_size", return_value=2), \
             patch.object(mpu, "get_tensor_model_parallel_group", return_value=MagicMock()), \
             patch.object(mpu, "get_pipeline_model_parallel_group", return_value=MagicMock()), \
             patch.object(mod.torch.distributed, "barrier") as mock_barrier, \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod, "get_vtp_intra_stage_group", return_value=MagicMock()):
            mod.vtp_hierarchical_barrier()
            judge_expression(mock_barrier.call_count == 3)

    def test_tp1_non_rank0_no_intra(self):
        with patch.object(mpu, "get_tensor_model_parallel_world_size", return_value=1), \
             patch.object(mod.torch.distributed, "barrier") as mock_barrier, \
             patch.object(mod, "is_vtp_stage_rank0", return_value=False), \
             patch.object(mod, "get_vtp_intra_stage_group", return_value=None):
            mod.vtp_hierarchical_barrier()
            mock_barrier.assert_not_called()


class TestAutoDetectVtpSizes:

    def _make_args(self, world_size, tp, pp, cp=1, ep=1, dp=1):
        args = MagicMock()
        args.world_size = world_size
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.context_parallel_size = cp
        args.expert_model_parallel_size = ep
        args.data_parallel_size = dp
        return args

    def test_heterogeneous_edge_cloud(self):
        with patch.object(mod.torch.distributed, "all_gather") as mock_ag, \
             patch.object(mod.torch.distributed, "get_world_size", return_value=9), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "0"}):
            args = self._make_args(9, 4, 3, cp=1, ep=1, dp=1)

            def fill(gathered, local_tensor):
                vals = [1] + [8] * 8
                for i, t in enumerate(gathered):
                    t.fill_(vals[i])
            mock_ag.side_effect = fill
            result = mod._auto_detect_vtp_sizes(args)
            judge_expression(result == [1, 4, 4])

    def test_homogeneous_returns_none(self):
        with patch.object(mod.torch.distributed, "all_gather") as mock_ag, \
             patch.object(mod.torch.distributed, "get_world_size", return_value=8), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "8"}):
            args = self._make_args(8, 4, 2, cp=1, ep=1, dp=1)

            def fill(gathered, local_tensor):
                for t in gathered:
                    t.fill_(8)
            mock_ag.side_effect = fill
            result = mod._auto_detect_vtp_sizes(args)
            judge_expression(result == [4, 4])

    def test_not_enough_cards(self):
        with patch.object(mod.torch.distributed, "all_gather") as mock_ag, \
             patch.object(mod.torch.distributed, "get_world_size", return_value=2), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "1"}):
            args = self._make_args(2, 4, 4, cp=1, ep=1, dp=1)

            def fill(gathered, local_tensor):
                for t in gathered:
                    t.fill_(1)
            mock_ag.side_effect = fill
            result = mod._auto_detect_vtp_sizes(args)
            judge_expression(isinstance(result, list))
            judge_expression(result[0] < 0)

    def test_max_tp_mismatch_returns_none(self):
        with patch.object(mod.torch.distributed, "all_gather") as mock_ag, \
             patch.object(mod.torch.distributed, "get_world_size", return_value=4), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "2"}):
            args = self._make_args(4, 8, 2, cp=1, ep=1, dp=1)

            def fill(gathered, local_tensor):
                for t in gathered:
                    t.fill_(2)
            mock_ag.side_effect = fill
            result = mod._auto_detect_vtp_sizes(args)
            judge_expression(isinstance(result, list))
            judge_expression(result[0] < 0)


class TestPreValidateArgsForVtp:

    def test_divisible_noop(self):
        args = MagicMock()
        args.world_size = 8
        args.tensor_model_parallel_size = 2
        args.pipeline_model_parallel_size = 2
        args.context_parallel_size = 1
        mod.pre_validate_args_for_vtp(args)
        judge_expression(args.world_size == 8)

    def test_inflates_world_size(self):
        args = MagicMock()
        args.world_size = 9
        args.tensor_model_parallel_size = 4
        args.pipeline_model_parallel_size = 2
        args.context_parallel_size = 1
        mod.pre_validate_args_for_vtp(args)
        judge_expression(args._vtp_orig_world_size == 9)
        judge_expression(args.world_size == 8)

    def test_no_world_size_noop(self):
        args = MagicMock(spec=[])
        args.world_size = None
        mod.pre_validate_args_for_vtp(args)

    def test_cp_none_defaults_to_1(self):
        args = MagicMock()
        args.world_size = 9
        args.tensor_model_parallel_size = 4
        args.pipeline_model_parallel_size = 2
        args.context_parallel_size = None
        mod.pre_validate_args_for_vtp(args)
        judge_expression(args.world_size == 8)


class TestPostValidateArgsForVtp:

    _ENV = {
        "GROUP_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "2",
        "GROUP_WORLD_SIZE": "2",
    }

    def test_restores_world_size(self):
        args = MagicMock()
        args._vtp_orig_world_size = 9
        args.world_size = 8
        args.tensor_model_parallel_size = 2
        args.pipeline_model_parallel_size = 2
        args.context_parallel_size = 1
        with patch.dict(os.environ, self._ENV):
            mod.post_validate_args_for_vtp(args)
        judge_expression(args.world_size == 9)

    def test_no_attr_noop(self):
        args = MagicMock(spec=[])
        args.world_size = 8
        args.tensor_model_parallel_size = 2
        args.pipeline_model_parallel_size = 2
        args.context_parallel_size = 1
        with patch.dict(os.environ, self._ENV):
            mod.post_validate_args_for_vtp(args)
        judge_expression(args.world_size == 8)


class TestInitializeModelParallelWrapper:

    _ENV = {
        "GROUP_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "GROUP_WORLD_SIZE": "1",
    }

    @staticmethod
    def _reset_ldt_globals():
        """Reset LDT/VTP/VDP module-level globals that persist between tests."""
        mod._PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_CLOUD_TP = None
        mod._PIPELINE_MODEL_PARALLEL_GROUP_FOR_VDP_CROSS_EDGE_CLOUD = None
        mod._LAYERWISE_DISAGGREGATED_TRAINING = False
        mod._VDP_SIZE = 1
        mod._VDP_ENABLED = False
        mod._VTP_ENABLED = False
        mod._VTP_SIZE_LIST = None
        mod._VTP_STAGE_RANKS = None
        mod._VTP_INTRA_STAGE_GROUP = None
        mod._VTP_MY_STAGE_IDX = None
        mod._EDGE_TP_SIZE = 1
        mod._PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = None
        mod._PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = None
        mod._PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = None

    def setup_method(self):
        self._reset_ldt_globals()

    def test_non_ldt_calls_fn_only(self):
        args = MagicMock(layerwise_disaggregated_training=False, data_parallel_size=1)
        with patch.object(mod, "get_args", return_value=args), \
             patch.object(mod, "_auto_detect_vtp_sizes", return_value=None), \
             patch.object(mod.torch.distributed, "is_initialized", return_value=True), \
             patch.object(mod.torch.distributed, "get_world_size", return_value=8), \
             patch.object(mod.torch.distributed, "get_rank", return_value=0), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.object(mod.torch.distributed, "new_group"), \
             patch.object(mod.torch.distributed, "broadcast_object_list"), \
             patch.object(mod, "_init_vdp_state"), \
             patch.object(mod, "get_args", return_value=MagicMock(
                 layerwise_disaggregated_training=False, data_parallel_size=1, vdp=0,
                 vtp_size_list=None, vtp_stage_ranks=None, vdp_size=1,
                 context_parallel_size=1
             )), \
             patch.dict(os.environ, self._ENV):
            fn = MagicMock()
            wrapper = mod.initialize_model_parallel_wrapper(fn)
            wrapper(4, 2)
            fn.assert_called_once()

    def test_ldt_uniform_calls_fn_and_group_init(self):
        args = MagicMock(layerwise_disaggregated_training=True, data_parallel_size=1)
        with patch.object(mod, "get_args", return_value=args), \
             patch.object(mod, "_auto_detect_vtp_sizes", return_value=None), \
             patch.object(mod.torch.distributed, "is_initialized", return_value=True), \
             patch.object(mod.torch.distributed, "get_world_size", return_value=8), \
             patch.object(mod.torch.distributed, "get_rank", return_value=0), \
             patch.object(mod.torch.cuda, "current_device", return_value=0), \
             patch.object(mod.torch.distributed, "new_group"), \
             patch.object(mod.torch.distributed, "broadcast_object_list"), \
             patch.object(mod, "_init_vdp_state"), \
             patch.dict(os.environ, self._ENV):
            fn = MagicMock()
            wrapper = mod.initialize_model_parallel_wrapper(fn)
            wrapper(4, 2)
            fn.assert_called_once()

    def test_ldt_non_uniform_calls_vtp_static(self):
        args = MagicMock(layerwise_disaggregated_training=True)
        with patch.object(mod, "get_args", return_value=args), \
             patch.object(mod, "_auto_detect_vtp_sizes", return_value=[1, 4, 4]), \
             patch.object(mod, "is_vdp_enabled", return_value=False), \
             patch.object(mod, "_initialize_vtp_static_only_vtp") as mock_static, \
             patch.object(mod.torch.distributed, "get_world_size", return_value=9), \
             patch.dict(os.environ, self._ENV):
            fn = MagicMock()
            wrapper = mod.initialize_model_parallel_wrapper(fn)
            wrapper(4, 3)
            mock_static.assert_called_once()
            fn.assert_not_called()
