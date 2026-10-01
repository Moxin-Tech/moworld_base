from typing import Dict, List

import pytest
import torch

from checkpoint.vlm_model.operator import (
    interleaved_qkv_to_concated,
    concated_qkv_to_interleaved,
    merge_to_interleaved_qkv,
    get_layer_num,
    RenameOp,
    UpGateMergeOp,
    QKVMergeOp,
    RelocateOp,
    TieOp,
    ColSplit,
    RowSplit,
    GLUSplit,
)


@pytest.fixture
def sample_weights() -> Dict[str, torch.Tensor]:
    return {
        'layer.0.q': torch.randn(2, 3),
        'layer.0.k': torch.randn(2, 3),
        'layer.0.v': torch.randn(2, 3),
        'layer.1.q': torch.randn(2, 3),
        'layer.1.k': torch.randn(2, 3),
        'layer.1.v': torch.randn(2, 3),
        'word_embeddings.weight': torch.randn(10, 5),
        'output_layer.weight': torch.randn(5, 10),
    }


@pytest.mark.parametrize("megatron_qkv, num_key_value_heads, split_size, expected_tensor", [
    (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 4, [1, 1, 1],
     torch.tensor([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]))
])
def test_interleaved_qkv_to_concated(
        megatron_qkv: torch.Tensor,
        num_key_value_heads: int,
        split_size: List[int],
        expected_tensor: torch.Size,
) -> None:
    result = interleaved_qkv_to_concated(megatron_qkv, num_key_value_heads, split_size)
    assert torch.equal(result, expected_tensor)


@pytest.mark.parametrize("qkv, num_key_value_heads, split_size, expected_tensor", [
    (torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), 4, [3, 3, 3, 3],
     torch.tensor([1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12]))
])
def test_concated_qkv_to_interleaved(
        qkv: torch.Tensor,
        num_key_value_heads: int,
        split_size: List[int],
        expected_tensor: torch.Size,
) -> None:
    result = concated_qkv_to_interleaved(qkv, num_key_value_heads, split_size)
    assert torch.equal(result, expected_tensor)


@pytest.mark.parametrize("q, k, v, group, expected_tensor", [
    (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9]), 4,
     torch.tensor([1, 4, 7, 2, 5, 8, 3, 6, 9])
     ),
])
def test_merge_to_interleaved_qkv(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        group: int,
        expected_tensor: torch.Size,
) -> None:
    result = merge_to_interleaved_qkv(q, k, v, group)
    assert torch.equal(result, expected_tensor)


def test_get_layer_num() -> None:
    names = ['layer.0.weight', 'layer.1.weight', 'layer.2.weight']
    pattern = r'layer.(\d+).weight'
    assert get_layer_num(pattern, names) == 3


def test_rename_op() -> None:
    weights = {'visual.blocks.0.attn.proj.bias': torch.randn(2, 3),
               'visual.blocks.10.attn.proj.bias': torch.randn(2, 3)}
    rename_op = RenameOp(((r'^visual\.blocks\.(\d+)\.', r'image_encoder.encoder.blocks.layers.\1.'),))
    rename_op.apply(weights)
    assert 'image_encoder.encoder.blocks.layers.0.attn.proj.bias' in weights
    assert 'image_encoder.encoder.blocks.layers.10.attn.proj.bias' in weights
    assert 'visual.blocks.0.attn.proj.bias' not in weights


def test_up_gate_merge_op() -> None:
    weights = {
        'layer.0.gate': torch.tensor([1, 2]),
        'layer.0.up': torch.tensor([3, 4]),
    }
    up_gate_merge_op = UpGateMergeOp([r'layer.(\d+).gate', r'layer.(\d+).up'], r'layer.(\d+).mlp')
    up_gate_merge_op.apply(weights)
    assert 'layer.0.mlp' in weights
    assert 'layer.0.gate' not in weights
    assert 'layer.0.up' not in weights


def test_qkv_merge_op() -> None:
    weights = {
        'layer.0.q': torch.tensor([1, 2, 3, 4]),
        'layer.0.k': torch.tensor([5, 6, 7, 8]),
        'layer.0.v': torch.tensor([9, 10, 11, 12]),
    }
    qkv_merge_op = QKVMergeOp((r'layer.(\d+).q', r'layer.(\d+).k', r'layer.(\d+).v'), r'layer.(\d+).qkv', 4, 1, 1, 1)
    qkv_merge_op.apply(weights)
    assert 'layer.0.qkv' in weights
    assert 'layer.0.q' not in weights
    assert 'layer.0.k' not in weights
    assert 'layer.0.v' not in weights


def test_relocate_op() -> None:
    weights = {'layer.0.qkv': torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])}
    relocate_op = RelocateOp(r'layer.(\d+).qkv', r'new_layer.(\d+).qkv', group=4, split_size=[3, 3, 3, 3])
    relocate_op.apply(weights)
    assert 'layer.0.qkv' in weights


def test_tie_op() -> None:
    weights = {'word_embeddings.weight': torch.randn(10, 5)}
    tie_op = TieOp('word_embeddings.weight', 'output_layer.weight')
    tie_op.apply(weights)
    assert 'output_layer.weight' in weights
    assert torch.allclose(weights.get('output_layer.weight'), weights.get('word_embeddings.weight'))


@pytest.mark.parametrize("tp_size, tp_rank, value, expected_shape", [
    (2, 0, torch.randn(4, 6), (4, 3)),
    (2, 1, torch.randn(4, 6), (4, 3)),
])
def test_tp_split_col_weight(
        tp_size: int,
        tp_rank: int,
        value: torch.Tensor,
        expected_shape: torch.Size,
) -> None:
    result = ColSplit.split(tp_size, value)
    assert result[tp_rank].shape == expected_shape


@pytest.mark.parametrize("tp_size, tp_rank, value, expected_shape", [
    (2, 0, torch.randn(4, 6), (2, 6)),
    (2, 1, torch.randn(4, 6), (2, 6)),
])
def test_tp_split_row_bias(
        tp_size: int,
        tp_rank: int,
        value: torch.Tensor,
        expected_shape: torch.Size,
) -> None:
    result = RowSplit.split(tp_size, value)
    assert result[tp_rank].shape == expected_shape


@pytest.mark.parametrize("tp_size, tp_rank, value, expected_shape", [
    (2, 0, torch.randn(4, 6), (2, 6)),
    (2, 1, torch.randn(4, 6), (2, 6)),
])
def test_tp_split_row_weight(
        tp_size: int,
        tp_rank: int,
        value: torch.Tensor,
        expected_shape: torch.Size,
) -> None:
    result = RowSplit.split(tp_size, value)
    assert result[tp_rank].shape == expected_shape


@pytest.mark.parametrize("tp_size, tp_rank, value, expected_shape", [
    (2, 0, torch.randn(8, 6), (4, 6)),
    (2, 1, torch.randn(8, 6), (4, 6)),
])
def test_tp_split_glu_weight(
        tp_size: int,
        tp_rank: int,
        value: torch.Tensor,
        expected_shape: torch.Size,
) -> None:
    result = GLUSplit.split(tp_size, value)
    assert result[tp_rank].shape == expected_shape


@pytest.mark.parametrize("tp_size, tp_rank, value, expected_shape", [
    (2, 0, torch.randn(8), (4,)),
    (2, 1, torch.randn(8), (4,)),
])
def test_tp_split_glu_bias(
        tp_size: int,
        tp_rank: int,
        value: torch.Tensor,
        expected_shape: torch.Size,
) -> None:
    result = GLUSplit.split(tp_size, value)
    assert result[tp_rank].shape == expected_shape


@pytest.mark.parametrize("value, expected_shape", [
    ([torch.randn(4, 6), torch.randn(4, 6)], (8, 6)),
    ([torch.randn(3, 6), torch.randn(3, 6)], (6, 6)),
])
def test_tp_merge_row_weight(
        value: List[torch.Tensor],
        expected_shape: torch.Size,
) -> None:
    result = RowSplit.merge(value)
    assert result.shape == expected_shape


@pytest.mark.parametrize("value, expected_shape", [
    ([torch.randn(4,), torch.randn(4,)], (8,)),
    ([torch.randn(3,), torch.randn(3,)], (6,)),
])
def test_tp_merge_row_weight(
        value: List[torch.Tensor],
        expected_shape: torch.Size,
) -> None:
    result = RowSplit.merge(value)
    assert result.shape == expected_shape


@pytest.mark.parametrize("value, expected_shape", [
    ([torch.randn(4, 6), torch.randn(4, 6)], (4, 12)),
    ([torch.randn(3, 6), torch.randn(3, 6)], (3, 12)),
])
def test_tp_merge_col_weight(
        value: List[torch.Tensor],
        expected_shape: torch.Size,
) -> None:
    result = ColSplit.merge(value)
    assert result.shape == expected_shape


@pytest.mark.parametrize("value, expected_shape", [
    ([torch.randn(4, 6), torch.randn(4, 6)], (8, 6)),
    ([torch.randn(4,), torch.randn(4,)], (8,)),
])
def test_tp_merge_glu(
        value: List[torch.Tensor],
        expected_shape: torch.Size,
):
    result = GLUSplit.merge(value)
    assert result.shape == expected_shape
