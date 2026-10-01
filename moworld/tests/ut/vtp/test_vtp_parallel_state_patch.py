# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP parallel state management."""

from unittest.mock import patch, MagicMock

import mindspeed.megatron_adaptor
import torch

from mindspeed_mm.patchs.layerwise_disaggregated_training import parallel_state_patch as mod
from tests.ut.utils import judge_expression


class TestInitVtpState:

    def test_sets_globals_and_finds_stage(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=3):
            mod._init_vtp_state(True, [2, 2, 2], [[0, 1], [2, 3], [4, 5]])
            judge_expression(mod._VTP_ENABLED is True)
            judge_expression(mod._VTP_SIZE_LIST == [2, 2, 2])
            judge_expression(mod._VTP_MY_STAGE_IDX == 1)

    def test_finds_first_stage(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=0):
            mod._init_vtp_state(True, [1, 4], [[0], [1, 2, 3, 4]])
            judge_expression(mod._VTP_MY_STAGE_IDX == 0)

    def test_finds_last_stage(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=4):
            mod._init_vtp_state(False, [1, 4], [[0], [1, 2, 3, 4]])
            judge_expression(mod._VTP_ENABLED is False)
            judge_expression(mod._VTP_MY_STAGE_IDX == 1)


class TestCreateVtpGroups:

    def test_creates_group_for_multi_rank_stage(self):
        mock_group = MagicMock()
        with patch.object(mod.torch.distributed, "get_rank", return_value=2), \
             patch.object(mod.torch.distributed, "new_group", return_value=mock_group):
            mod._VTP_INTRA_STAGE_GROUP = None
            mod._create_vtp_groups([[0], [1, 2, 3]], timeout=30, backend="nccl")
            mod.torch.distributed.new_group.assert_called_once()
            judge_expression(mod._VTP_INTRA_STAGE_GROUP is mock_group)

    def test_skips_single_rank_stage(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=0), \
             patch.object(mod.torch.distributed, "new_group") as mock_ng:
            mod._VTP_INTRA_STAGE_GROUP = None
            mod._create_vtp_groups([[0], [1]], timeout=30, backend="nccl")
            mock_ng.assert_not_called()
            judge_expression(mod._VTP_INTRA_STAGE_GROUP is None)

    def test_rank_not_in_multi_rank_stage(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=5), \
             patch.object(mod.torch.distributed, "new_group", return_value=MagicMock()) as mock_ng:
            mod._VTP_INTRA_STAGE_GROUP = None
            mod._create_vtp_groups([[0, 1], [2, 3]], timeout=30, backend="nccl")
            judge_expression(mock_ng.call_count == 2)
            judge_expression(mod._VTP_INTRA_STAGE_GROUP is None)


class TestGetters:

    def test_is_vtp_enabled(self):
        mod._VTP_ENABLED = True
        judge_expression(mod.is_vtp_enabled() is True)
        mod._VTP_ENABLED = False
        judge_expression(mod.is_vtp_enabled() is False)

    def test_get_vtp_size_list(self):
        mod._VTP_SIZE_LIST = [1, 4, 4]
        judge_expression(mod.get_vtp_size_list() == [1, 4, 4])

    def test_get_vtp_stage_ranks(self):
        ranks = [[0], [1, 2, 3, 4]]
        mod._VTP_STAGE_RANKS = ranks
        judge_expression(mod.get_vtp_stage_ranks() is ranks)

    def test_get_vtp_intra_stage_group(self):
        mock_group = MagicMock()
        mod._VTP_INTRA_STAGE_GROUP = mock_group
        judge_expression(mod.get_vtp_intra_stage_group() is mock_group)

    def test_get_vtp_my_stage_idx(self):
        mod._VTP_MY_STAGE_IDX = 2
        judge_expression(mod.get_vtp_my_stage_idx() == 2)


class TestIsVtpStageRank0:

    def test_returns_true_when_stage_ranks_none(self):
        mod._VTP_STAGE_RANKS = None
        mod._VTP_MY_STAGE_IDX = 0
        judge_expression(mod.is_vtp_stage_rank0() is True)

    def test_returns_true_when_stage_ranks_empty(self):
        mod._VTP_STAGE_RANKS = []
        mod._VTP_MY_STAGE_IDX = 0
        judge_expression(mod.is_vtp_stage_rank0() is True)

    def test_returns_true_when_stage_idx_none(self):
        mod._VTP_STAGE_RANKS = [[0, 1]]
        mod._VTP_MY_STAGE_IDX = None
        judge_expression(mod.is_vtp_stage_rank0() is True)

    def test_returns_true_when_rank_is_first(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=0):
            mod._VTP_STAGE_RANKS = [[0, 1], [2, 3]]
            mod._VTP_MY_STAGE_IDX = 0
            judge_expression(mod.is_vtp_stage_rank0() is True)

    def test_returns_false_when_rank_is_not_first(self):
        with patch.object(mod.torch.distributed, "get_rank", return_value=3):
            mod._VTP_STAGE_RANKS = [[0, 1], [2, 3]]
            mod._VTP_MY_STAGE_IDX = 1
            judge_expression(mod.is_vtp_stage_rank0() is False)
