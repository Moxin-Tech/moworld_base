"""Unit tests for FSDP MoE checkpoint planner helper behavior."""

import os
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class _FakeReadItem:
    storage_offsets: object
    lengths: object
    dest_offsets: object = None
    fqn: str = "param"


class _FakeMesh:
    def __init__(self, mesh_dim_names):
        self.mesh_dim_names = mesh_dim_names


class _FakeDTensor:
    def __init__(self, mesh_dim_names):
        self.device_mesh = _FakeMesh(mesh_dim_names)


class _FakeModel:
    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return iter(self._params)


class TestGetChunkReadItem:
    @pytest.mark.parametrize(
        "offsets,lengths,ep_rank,expected_offsets",
        [
            ([0], [1], 0, [0]),
            ([0], [1], 1, [1]),
            ([0], [1], 2, [2]),
            ([3], [7], 0, [3]),
            ([3], [7], 1, [10]),
            ([3], [7], 4, [31]),
            ([0, 0], [2, 5], 0, [0, 0]),
            ([0, 0], [2, 5], 1, [2, 0]),
            ([1, 3], [4, 8], 2, [9, 3]),
            ([5, 6, 7], [10, 11, 12], 3, [35, 6, 7]),
        ],
    )
    def test_get_chunk_readitem_offsets_first_dimension_by_ep_rank(
        self,
        offsets,
        lengths,
        ep_rank,
        expected_offsets,
    ):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.moe_utils import get_chunk_readitem

        readitem = _FakeReadItem(
            storage_offsets=torch.Size(offsets),
            lengths=torch.Size(lengths),
            dest_offsets=torch.Size([0] * len(offsets)),
        )

        chunked = get_chunk_readitem(readitem, ep_rank)

        assert chunked is not readitem
        assert chunked.storage_offsets == torch.Size(expected_offsets)
        assert chunked.lengths == torch.Size(lengths)
        assert chunked.dest_offsets == torch.Size([0] * len(offsets))

    @pytest.mark.parametrize(
        "operate_dim,expected_offsets",
        [
            (0, [10, 4, 6]),
            (1, [2, 24, 6]),
            (2, [2, 4, 36]),
        ],
    )
    def test_get_chunk_readitem_can_offset_selected_dimension(self, operate_dim, expected_offsets):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.moe_utils import get_chunk_readitem

        readitem = _FakeReadItem(
            storage_offsets=torch.Size([2, 4, 6]),
            lengths=torch.Size([4, 10, 15]),
        )

        chunked = get_chunk_readitem(readitem, ep_rank=2, operate_dim=operate_dim)

        assert chunked.storage_offsets == torch.Size(expected_offsets)

    def test_get_chunk_readitem_raises_when_offsets_and_lengths_have_different_rank(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.moe_utils import get_chunk_readitem

        readitem = _FakeReadItem(
            storage_offsets=torch.Size([0, 1]),
            lengths=torch.Size([2]),
        )

        with pytest.raises(ValueError, match="same size"):
            get_chunk_readitem(readitem, ep_rank=1)

    def test_get_chunk_readitem_preserves_non_offset_fields(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.moe_utils import get_chunk_readitem

        readitem = _FakeReadItem(
            storage_offsets=torch.Size([1, 2]),
            lengths=torch.Size([3, 4]),
            dest_offsets=torch.Size([5, 6]),
            fqn="moe.experts.weight",
        )

        chunked = get_chunk_readitem(readitem, ep_rank=3)

        assert chunked.fqn == "moe.experts.weight"
        assert chunked.dest_offsets == torch.Size([5, 6])
        assert chunked.lengths == torch.Size([3, 4])


class TestGetCheckMoeFunc:
    def test_check_moe_func_matches_dtensor_params_on_efsdp_mesh(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.checkpoint.moe_utils as moe_utils

        monkeypatch.setattr(moe_utils, "DTensor", _FakeDTensor)
        model = _FakeModel(
            [
                ("layers.0.mlp.experts.0.weight", _FakeDTensor(["dp", "efsdp"])),
                ("layers.0.mlp.shared.weight", object()),
                ("layers.1.attn.q_proj.weight", _FakeDTensor(["dp", "tp"])),
            ]
        )

        check_moe = moe_utils.get_check_moe_func(model)

        assert check_moe("layers.0.mlp.experts.0.weight") is True
        assert check_moe("module.layers.0.mlp.experts.0.weight") is True
        assert check_moe("layers.0.mlp.shared.weight") is False
        assert check_moe("layers.1.attn.q_proj.weight") is False

    def test_check_moe_func_strips_recompute_prefix_from_model_params(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.checkpoint.moe_utils as moe_utils

        monkeypatch.setattr(moe_utils, "DTensor", _FakeDTensor)
        model = _FakeModel(
            [
                ("_checkpoint_wrapped_module.layers.4.experts.down_proj.weight", _FakeDTensor(["efsdp"])),
            ]
        )

        check_moe = moe_utils.get_check_moe_func(model)

        assert check_moe("layers.4.experts.down_proj.weight") is True
        assert check_moe("_checkpoint_wrapped_module.layers.4.experts.down_proj.weight") is True
        assert check_moe("layers.4.router.weight") is False

    def test_check_moe_func_returns_false_when_no_efsdp_params_exist(self, monkeypatch):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.checkpoint.moe_utils as moe_utils

        monkeypatch.setattr(moe_utils, "DTensor", _FakeDTensor)
        model = _FakeModel(
            [
                ("layers.0.experts.weight", _FakeDTensor(["dp"])),
                ("layers.0.router.weight", _FakeDTensor(["tp"])),
                ("layers.0.norm.weight", object()),
            ]
        )

        check_moe = moe_utils.get_check_moe_func(model)

        assert check_moe("layers.0.experts.weight") is False
        assert check_moe("layers.0.router.weight") is False
        assert check_moe("anything") is False

    @pytest.mark.parametrize(
        "candidate,expected",
        [
            ("blocks.0.moe.experts.0.w1.weight", True),
            ("prefix.blocks.0.moe.experts.0.w1.weight", True),
            ("_checkpoint_wrapped_module.blocks.0.moe.experts.0.w1.weight", True),
            ("blocks.0.moe.experts.0.w2.weight", False),
            ("blocks.1.moe.experts.0.w1.weight", False),
            ("blocks.0.moe.router.weight", False),
        ],
    )
    def test_check_moe_func_uses_suffix_matching_for_wrapped_names(self, monkeypatch, candidate, expected):
        pytest.importorskip("torch")
        import mindspeed_mm.fsdp.checkpoint.moe_utils as moe_utils

        monkeypatch.setattr(moe_utils, "DTensor", _FakeDTensor)
        model = _FakeModel(
            [
                ("blocks.0.moe.experts.0.w1.weight", _FakeDTensor(["dp", "efsdp"])),
            ]
        )

        check_moe = moe_utils.get_check_moe_func(model)

        assert check_moe(candidate) is expected
