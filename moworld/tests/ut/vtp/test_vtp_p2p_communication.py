# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP P2P communication functions."""

from unittest.mock import patch, MagicMock

import mindspeed.megatron_adaptor
import torch

from mindspeed_mm.patchs.layerwise_disaggregated_training import p2p_communication_patch as mod
from tests.ut.utils import judge_expression


class TestVtpSendForward:

    def _setup_mocks(self, rank0=True):
        mocks = {
            "get_vtp_my_stage_idx": patch.object(mod, "get_vtp_my_stage_idx", return_value=1),
            "pp_group": patch.object(mod, "get_pipeline_model_parallel_group", return_value=MagicMock()),
            "next_rank": patch.object(mod, "get_pipeline_model_parallel_next_rank", return_value=5),
            "prev_rank": patch.object(mod, "get_pipeline_model_parallel_prev_rank", return_value=0),
        }
        return mocks

    def test_rank0_sends(self):
        mocks = self._setup_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["pp_group"], mocks["next_rank"], mocks["prev_rank"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "isend", return_value=MagicMock()) as mock_isend:
            result = mod.vtp_send_forward(torch.tensor([1.0]))
            mock_isend.assert_called_once()
            judge_expression(result is not None)

    def test_non_rank0_returns_none(self):
        with patch.object(mod, "is_vtp_stage_rank0", return_value=False):
            result = mod.vtp_send_forward(torch.tensor([1.0]))
            judge_expression(result is None)


class TestVtpSendBackward:

    def _setup_mocks(self, rank0=True):
        mocks = {
            "get_vtp_my_stage_idx": patch.object(mod, "get_vtp_my_stage_idx", return_value=1),
            "pp_group": patch.object(mod, "get_pipeline_model_parallel_group", return_value=MagicMock()),
            "next_rank": patch.object(mod, "get_pipeline_model_parallel_next_rank", return_value=5),
            "prev_rank": patch.object(mod, "get_pipeline_model_parallel_prev_rank", return_value=2),
        }
        return mocks

    def test_rank0_sends(self):
        mocks = self._setup_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["pp_group"], mocks["next_rank"], mocks["prev_rank"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "isend", return_value=MagicMock()) as mock_isend:
            result = mod.vtp_send_backward(torch.tensor([1.0]))
            mock_isend.assert_called_once()
            judge_expression(result is not None)

    def test_non_rank0_returns_none(self):
        with patch.object(mod, "is_vtp_stage_rank0", return_value=False):
            result = mod.vtp_send_backward(torch.tensor([1.0]))
            judge_expression(result is None)


class TestVTPRecvWork:

    def test_wait_with_irecv_and_broadcast(self):
        irecv_work = MagicMock()
        tensor = torch.tensor([1.0])
        intra_group = MagicMock()
        with patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            work = mod._VTPRecvWork(irecv_work, tensor, broadcast_src=0, intra_group=intra_group, dst_size=4)
            work.wait()
            irecv_work.wait.assert_called_once()
            mock_bcast.assert_called_once_with(tensor, src=0, group=intra_group)

    def test_wait_no_irecv(self):
        with patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            work = mod._VTPRecvWork(None, torch.tensor([1.0]), broadcast_src=0, intra_group=MagicMock(), dst_size=2)
            work.wait()
            mock_bcast.assert_called_once()

    def test_wait_dst_size_1_no_broadcast(self):
        with patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            work = mod._VTPRecvWork(MagicMock(), torch.tensor([1.0]), broadcast_src=0, intra_group=MagicMock(), dst_size=1)
            work.wait()
            mock_bcast.assert_not_called()

    def test_wait_no_intra_group_no_broadcast(self):
        with patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            work = mod._VTPRecvWork(MagicMock(), torch.tensor([1.0]), broadcast_src=0, intra_group=None, dst_size=4)
            work.wait()
            mock_bcast.assert_not_called()


class TestVtpRecvForward:

    def _setup_recv_mocks(self):
        return {
            "get_vtp_my_stage_idx": patch.object(mod, "get_vtp_my_stage_idx", return_value=1),
            "get_vtp_stage_ranks": patch.object(mod, "get_vtp_stage_ranks", return_value=[[0], [1, 2, 3, 4]]),
            "get_vtp_size_list": patch.object(mod, "get_vtp_size_list", return_value=[1, 4]),
            "intra_group": patch.object(mod, "get_tensor_model_parallel_group", return_value=MagicMock()),
            "pp_group": patch.object(mod, "get_pipeline_model_parallel_group", return_value=MagicMock()),
            "prev_rank": patch.object(mod, "get_pipeline_model_parallel_prev_rank", return_value=0),
            "next_rank": patch.object(mod, "get_pipeline_model_parallel_next_rank", return_value=1),
            "current_device": patch.object(mod.torch.cuda, "current_device", return_value=0),
        }

    def test_sync_rank0_recv_and_broadcast(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "irecv", return_value=MagicMock()) as mock_irecv, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            result = mod.vtp_recv_forward((2, 3), config, async_op=False)
            mock_irecv.assert_called_once()
            mock_bcast.assert_called_once()
            judge_expression(isinstance(result, torch.Tensor))

    def test_sync_non_rank0_broadcast_only(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=False), \
             patch.object(mod.torch.distributed, "irecv") as mock_irecv, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            mod.vtp_recv_forward((2, 3), config, async_op=False)
            mock_irecv.assert_not_called()
            mock_bcast.assert_called_once()

    def test_async_returns_tensor_and_reqs(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "irecv", return_value=MagicMock()):
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            tensor, reqs = mod.vtp_recv_forward((2, 3), config, async_op=True)
            judge_expression(isinstance(tensor, torch.Tensor))
            judge_expression("recv_prev" in reqs)


class TestVtpRecvBackward:

    def _setup_recv_mocks(self):
        return {
            "get_vtp_my_stage_idx": patch.object(mod, "get_vtp_my_stage_idx", return_value=1),
            "get_vtp_stage_ranks": patch.object(mod, "get_vtp_stage_ranks", return_value=[[0], [1, 2, 3, 4]]),
            "get_vtp_size_list": patch.object(mod, "get_vtp_size_list", return_value=[1, 4]),
            "intra_group": patch.object(mod, "get_tensor_model_parallel_group", return_value=MagicMock()),
            "pp_group": patch.object(mod, "get_pipeline_model_parallel_group", return_value=MagicMock()),
            "prev_rank": patch.object(mod, "get_pipeline_model_parallel_prev_rank", return_value=0),
            "next_rank": patch.object(mod, "get_pipeline_model_parallel_next_rank", return_value=5),
            "current_device": patch.object(mod.torch.cuda, "current_device", return_value=0),
        }

    def test_sync_rank0(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "irecv", return_value=MagicMock()) as mock_irecv, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            mod.vtp_recv_backward((2, 3), config, async_op=False)
            mock_irecv.assert_called_once()
            mock_bcast.assert_called_once()

    def test_async_returns_recv_next(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=True), \
             patch.object(mod.torch.distributed, "irecv", return_value=MagicMock()):
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            tensor, reqs = mod.vtp_recv_backward((2, 3), config, async_op=True)
            judge_expression("recv_next" in reqs)

    def test_sync_non_rank0(self):
        mocks = self._setup_recv_mocks()
        with mocks["get_vtp_my_stage_idx"], mocks["get_vtp_stage_ranks"], mocks["get_vtp_size_list"], \
             mocks["intra_group"], mocks["pp_group"], mocks["prev_rank"], mocks["next_rank"], mocks["current_device"], \
             patch.object(mod, "is_vtp_stage_rank0", return_value=False), \
             patch.object(mod.torch.distributed, "irecv") as mock_irecv, \
             patch.object(mod.torch.distributed, "broadcast") as mock_bcast:
            config = MagicMock()
            config.pipeline_dtype = torch.float32
            mod.vtp_recv_backward((2, 3), config, async_op=False)
            mock_irecv.assert_not_called()
            mock_bcast.assert_called_once()
