import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from enum import Enum

import pytest
import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from checkpoint.vlm_model.converters.moe_expert import (
    merge_moe_expert_weights,
    split_moe_expert_weights,
    save_sharded_state_dict,
    moe_expert,
    ConfigType,
)


@pytest.fixture
def setup_test_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        (test_dir / "config.json").touch()
        (test_dir / "pytorch_model.bin").touch()
        yield test_dir


def test_merge_moe_expert_weights():
    # Create test state_dict
    state_dict = {
        "layer.0.experts.0.gate_proj.weight": torch.randn(2, 3),
        "layer.0.experts.0.up_proj.weight": torch.randn(2, 3),
        "layer.0.experts.0.down_proj.weight": torch.randn(2, 3),
        "layer.0.experts.1.gate_proj.weight": torch.randn(2, 3),
        "layer.0.experts.1.up_proj.weight": torch.randn(2, 3),
        "layer.0.experts.1.down_proj.weight": torch.randn(2, 3),
    }
    expert_start_layer = 0
    num_hidden_layers = 1
    num_experts = 2
    weight_path = "layer.{layer}.experts.{expert}"
    config_type = ConfigType.DEFAULT

    merge_moe_expert_weights(state_dict, num_hidden_layers, num_experts, expert_start_layer, config_type, weight_path)

    # Check merged weights
    assert "layer.0.experts.0.gate_proj.weight" not in state_dict
    assert "layer.0.experts.0.up_proj.weight" not in state_dict
    assert "layer.0.experts.0.down_proj.weight" not in state_dict
    assert "layer.0.experts.1.gate_proj.weight" not in state_dict
    assert "layer.0.experts.1.up_proj.weight" not in state_dict
    assert "layer.0.experts.1.down_proj.weight" not in state_dict

    assert "layer.0.experts.gate_up_proj" in state_dict
    assert "layer.0.experts.down_proj" in state_dict

    merged_gate_up_proj = state_dict.get("layer.0.experts.gate_up_proj")
    merged_down_proj = state_dict.get("layer.0.experts.down_proj")

    assert merged_gate_up_proj.shape == (6, 4)
    assert merged_down_proj.shape == (6, 2)


def test_split_moe_expert_weights():
    # Create test state_dict
    state_dict = {
        "layer.0.experts.gate_up_proj": torch.randn(6, 4),
        "layer.0.experts.down_proj": torch.randn(6, 2),
    }
    expert_start_layer = 0
    num_hidden_layers = 1
    num_experts = 2
    weight_path = "layer.{layer}.experts.{expert}"
    config_type = ConfigType.DEFAULT

    split_moe_expert_weights(state_dict, num_hidden_layers, num_experts, expert_start_layer, config_type, weight_path)
    assert "layer.0.experts.gate_up_proj" not in state_dict
    assert "layer.0.experts.down_proj" not in state_dict

    assert "layer.0.experts.0.gate_proj.weight" in state_dict
    assert "layer.0.experts.0.up_proj.weight" in state_dict
    assert "layer.0.experts.0.down_proj.weight" in state_dict
    assert "layer.0.experts.1.gate_proj.weight" in state_dict
    assert "layer.0.experts.1.up_proj.weight" in state_dict
    assert "layer.0.experts.1.down_proj.weight" in state_dict

    expert_0_gate_proj = state_dict.get("layer.0.experts.0.gate_proj.weight")
    expert_0_up_proj = state_dict.get("layer.0.experts.0.up_proj.weight")
    expert_0_down_proj = state_dict.get("layer.0.experts.0.down_proj.weight")
    expert_1_gate_proj = state_dict.get("layer.0.experts.1.gate_proj.weight")
    expert_1_up_proj = state_dict.get("layer.0.experts.1.up_proj.weight")
    expert_1_down_proj = state_dict.get("layer.0.experts.1.down_proj.weight")

    assert expert_0_gate_proj.shape == (2, 3)
    assert expert_0_up_proj.shape == (2, 3)
    assert expert_0_down_proj.shape == (2, 3)
    assert expert_1_gate_proj.shape == (2, 3)
    assert expert_1_up_proj.shape == (2, 3)
    assert expert_1_down_proj.shape == (2, 3)


def test_save_sharded_state_dict(setup_test_files):
    test_dir = setup_test_files
    state_dict = {
        "weight1": torch.randn(10, 10),
        "weight2": torch.randn(20, 20),
    }
    max_shard_size = "1MB"

    save_sharded_state_dict(state_dict, test_dir, max_shard_size, metadata={"format": "pt"})

    # Check if index file exists
    index_file = test_dir / SAFE_WEIGHTS_INDEX_NAME
    assert index_file.exists()

    # Check if shard files exist
    shard_files = list(test_dir.glob("*.safetensors"))
    assert len(shard_files) > 0


def test_moe_expert_merge(setup_test_files):
    # here we use mock to focus on moe_expert.py file only.
    test_dir = setup_test_files
    # notice, patch `moe_expert.AutoConfig.from_pretrained`, not `transformers.AutoConfig.from_pretrained`
    with patch("checkpoint.vlm_model.converters.moe_expert.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value = MagicMock(
            __class__=MagicMock(__name__="InternVLChatConfig"),
            llm_config=MagicMock(num_hidden_layers=1, num_experts=2)
        )
        # notice, patch `checkpoint.vlm_model.converters.moe_expert.load_from_hf`, not `hf_to_mm.load_from_hf`
        with patch("checkpoint.vlm_model.converters.moe_expert.load_from_hf") as mock_load:
            mock_load.return_value = {
                "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(2, 3),
                "language_model.model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(2, 3),
                "language_model.model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(2, 3),
                "language_model.model.layers.0.mlp.experts.1.gate_proj.weight": torch.randn(2, 3),
                "language_model.model.layers.0.mlp.experts.1.up_proj.weight": torch.randn(2, 3),
                "language_model.model.layers.0.mlp.experts.1.down_proj.weight": torch.randn(2, 3),
            }
            with tempfile.TemporaryDirectory() as temp_dir:
                save_dir = Path(temp_dir)
                moe_expert("merge", test_dir, save_dir)
                assert len(list(save_dir.glob("*.safetensors"))) > 0


def test_moe_expert_split(setup_test_files):
    test_dir = setup_test_files
    with patch("checkpoint.vlm_model.converters.moe_expert.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value = MagicMock(
            __class__=MagicMock(__name__="InternVLChatConfig"),
            llm_config=MagicMock(num_hidden_layers=1, num_experts=2)
        )
        with patch("checkpoint.vlm_model.converters.moe_expert.load_from_hf") as mock_load:
            mock_load.return_value = {
                "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.randn(8, 3),
                "language_model.model.layers.0.mlp.experts.down_proj": torch.randn(4, 3),
            }
            with tempfile.TemporaryDirectory() as temp_dir:
                save_dir = Path(temp_dir)
                moe_expert("split", test_dir, save_dir)
                assert len(list(save_dir.glob("*.safetensors"))) > 0
