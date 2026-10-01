"""Unit tests for FSDP checkpoint load utility helpers."""

import os

import pytest


class TestChunkList:
    @pytest.mark.parametrize(
        "items,chunk_size,expected",
        [
            ([], 1, [[]]),
            ([], 3, [[], [], []]),
            ([1], 1, [[1]]),
            ([1], 2, [[1], []]),
            ([1, 2], 1, [[1, 2]]),
            ([1, 2], 2, [[1], [2]]),
            ([1, 2], 3, [[1], [2], []]),
            ([1, 2, 3], 2, [[1, 2], [3]]),
            ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
            ([1, 2, 3, 4, 5], 2, [[1, 2, 3], [4, 5]]),
            ([1, 2, 3, 4, 5], 3, [[1, 2], [3, 4], [5]]),
            ([1, 2, 3, 4, 5, 6, 7], 3, [[1, 2, 3], [4, 5], [6, 7]]),
            ([1, 2, 3, 4, 5, 6, 7], 4, [[1, 2], [3, 4], [5, 6], [7]]),
            (list(range(10)), 3, [list(range(4)), list(range(4, 7)), list(range(7, 10))]),
            (list(range(10)), 4, [list(range(3)), list(range(3, 6)), list(range(6, 8)), list(range(8, 10))]),
            (list(range(12)), 5, [list(range(3)), list(range(3, 6)), list(range(6, 8)), list(range(8, 10)), list(range(10, 12))]),
        ],
    )
    def test_chunk_list_balances_remainder_to_earlier_chunks(self, items, chunk_size, expected):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import chunk_list

        assert chunk_list(items, chunk_size) == expected

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 2, 3, 4, 5, 6, 7, 8],
    )
    def test_chunk_list_keeps_original_order(self, chunk_size):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import chunk_list

        items = [f"param_{idx}" for idx in range(17)]
        chunks = chunk_list(items, chunk_size)

        assert [item for chunk in chunks for item in chunk] == items

    @pytest.mark.parametrize(
        "length,chunk_size",
        [
            (0, 1),
            (0, 5),
            (1, 4),
            (2, 8),
            (5, 2),
            (5, 7),
            (16, 4),
            (17, 4),
            (31, 6),
            (64, 9),
        ],
    )
    def test_chunk_list_returns_requested_number_of_chunks(self, length, chunk_size):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import chunk_list

        chunks = chunk_list(list(range(length)), chunk_size)

        assert len(chunks) == chunk_size

    @pytest.mark.parametrize(
        "length,chunk_size",
        [
            (1, 2),
            (2, 3),
            (3, 2),
            (5, 2),
            (5, 3),
            (5, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (15, 6),
            (29, 7),
        ],
    )
    def test_chunk_list_chunk_lengths_differ_by_at_most_one(self, length, chunk_size):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import chunk_list

        chunks = chunk_list(list(range(length)), chunk_size)
        lengths = [len(chunk) for chunk in chunks]

        assert max(lengths) - min(lengths) <= 1

    def test_chunk_list_raises_when_chunk_size_is_zero(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import chunk_list

        with pytest.raises(ZeroDivisionError):
            chunk_list([1, 2, 3], 0)


class TestParamInfo:
    def test_param_info_defaults_to_empty_metadata(self):
        pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import ParamInfo

        info = ParamInfo()

        assert info.name is None
        assert info.shape is None
        assert info.dtype is None
        assert info.prefix is None

    def test_param_info_stores_tensor_metadata(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import ParamInfo

        info = ParamInfo(
            name="model.layers.0.weight",
            shape=torch.Size([2, 3]),
            dtype=torch.float32,
            prefix="model",
        )

        assert info.name == "model.layers.0.weight"
        assert info.shape == torch.Size([2, 3])
        assert info.dtype is torch.float32
        assert info.prefix == "model"

    def test_param_info_equality_uses_dataclass_value_semantics(self):
        torch = pytest.importorskip("torch")
        from mindspeed_mm.fsdp.checkpoint.load_utils import ParamInfo

        left = ParamInfo("weight", torch.Size([4]), torch.bfloat16, "optimizer")
        right = ParamInfo("weight", torch.Size([4]), torch.bfloat16, "optimizer")
        different = ParamInfo("bias", torch.Size([4]), torch.bfloat16, "optimizer")

        assert left == right
        assert left != different
