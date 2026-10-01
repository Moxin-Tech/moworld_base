# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for LoRA with FSDP2.

This module provides comprehensive tests for LoRA integration
with FSDP2 distributed training, including:
- LoRA adapter injection
- Parameter validation
- Weight saving and loading
- Multi-card compatibility
"""

import os
import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.utils.lora_utils import (
    add_lora_to_model,
    freeze_parameters,
    load_state_dict,
    match_target_modules,
    validate_lora_config,
    get_lora_trainable_params,
)
from mindspeed_mm.fsdp.utils.lora_weight_manager import LoraWeightManager


class SimpleModel(nn.Module):
    """Simple test model for LoRA testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestLoraUtils:
    """Test LoRA utility functions."""

    def test_validate_lora_config_valid(self) -> None:
        """Test validation of valid LoRA configuration."""
        validate_lora_config(
            rank=8,
            alpha=16,
            target_modules=["q_proj", "k_proj"],
            dropout=0.05,
            init_lora_weights=True,
        )

    def test_validate_lora_config_invalid_rank(self) -> None:
        """Test validation of invalid LoRA rank."""
        with pytest.raises(ValueError):
            validate_lora_config(
                rank=0,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_validate_lora_config_invalid_alpha(self) -> None:
        """Test validation of invalid LoRA alpha."""
        with pytest.raises(ValueError):
            validate_lora_config(
                rank=8,
                alpha=0,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_validate_lora_config_invalid_dropout(self) -> None:
        """Test validation of invalid LoRA dropout."""
        with pytest.raises(ValueError):
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=1.5,
                init_lora_weights=True,
            )

    def test_match_target_modules_exact(self) -> None:
        """Test exact matching of target modules."""
        model = SimpleModel()
        patterns = ["linear1", "linear2"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 2
        assert "linear1" in matched
        assert "linear2" in matched

    def test_match_target_modules_wildcard(self) -> None:
        """Test wildcard matching of target modules."""
        model = SimpleModel()
        patterns = ["linear{*}"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 2
        assert "linear1" in matched
        assert "linear2" in matched

    def test_match_target_modules_no_match(self) -> None:
        """Test when no modules match the pattern."""
        model = SimpleModel()
        patterns = ["nonexistent"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 0

    def test_get_lora_trainable_params(self) -> None:
        """Test getting LoRA trainable parameters statistics."""
        model = SimpleModel()

        trainable, total, stats = get_lora_trainable_params(model)

        # For a SimpleModel without LoRA, all parameters are trainable by default
        # linear1: 10*20=200, linear2: 20*10=200, total=400
        assert trainable > 0  # Should be 400 (all params trainable)
        assert total > 0  # Should be 400
        assert stats["trainable_params"] == trainable
        assert stats["total_params"] == total
        assert stats["trainable_ratio"] > 0  # Should be 1.0 (100% trainable)
        assert stats["lora_params"] == 0  # No LoRA params in SimpleModel


class TestLoraWeightManager:
    """Test LoRA weight manager."""

    def test_init_weight_manager(self) -> None:
        """Test initialization of LoRA weight manager."""
        model = SimpleModel()
        manager = LoraWeightManager(model)

        assert manager.model is model
        assert isinstance(manager._is_distributed, bool)

    def test_get_lora_param_count(self) -> None:
        """Test getting LoRA parameter count."""
        model = SimpleModel()
        manager = LoraWeightManager(model)

        num_params, num_elements = manager.get_lora_param_count()

        assert num_params == 0
        assert num_elements == 0

    def test_get_lora_state_dict(self) -> None:
        """Test getting LoRA state dictionary."""
        model = SimpleModel()
        manager = LoraWeightManager(model)

        state_dict = manager.get_lora_state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) == 0

    def test_verify_lora_weights_no_lora(self) -> None:
        """Test verification when no LoRA weights exist."""
        model = SimpleModel()
        manager = LoraWeightManager(model)

        result = manager.verify_lora_weights()

        assert result is False


class TestLoraFSDP2Integration:
    """Integration tests for LoRA with FSDP2."""

    def test_lora_enable_with_config(self, tmp_path: pathlib.Path) -> None:
        """Test enabling LoRA with configuration file."""
        # Create a minimal test configuration
        config_path = os.path.join(tmp_path, "test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("""
parallel:
  tensor_parallel_size: 1
  fully_shard_parallel_size: 1
  fsdp_plan:
    apply_modules: []
    param_dtype: bf16
  recompute: false

data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    preprocess_parameters:
      model_name_or_path: ./test_model
      use_fast_tokenizer: true
    basic_parameters:
      cutoff_len: 1024
      template: qwen3_vl_nothink
      dataset_dir: ./data
      dataset: ./data/test.json
      cache_dir: ./cache_dir/
  dataloader_param:
    pin_memory: false
    shuffle: false
    dataloader_mode: sampler
    drop_last: true
    num_workers: 0

model:
  model_id: qwen3vl
  model_name_or_path: ./test_model
  trust_remote_code: true
  attn_implementation: sdpa
  loss_cfg:
    loss_type: default
    router_aux_loss_coef: 0.0

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 1
  seed: 42
  lr: 1e-4
  train_iters: 10
  clip_grad: 0.0
  init_model_with_meta_device: false
  optimizer: adamw
  adam_fused: false
  save_interval: 10
  no_load_opt: false
  no_load_rng: false
  no_save_optim: true
  no_save_rng: true
  load: null
  save: ./test_save
  use_deter_comp: false
  lora:
    enable: true
    rank: 4
    alpha: 8
    target_modules:
      - "linear1"
      - "linear2"
    dropout: 0.0
    init_lora_weights: true
    pretrained_lora_path: null

tools:
  profile:
    enable: false
  memory_profile:
    enable: false
""")

        # Note: This test requires actual model files, so we skip if they don't exist
        if not os.path.exists("./test_model"):
            pytest.skip("Test model not found")

        args = parse_args(Arguments, config_path)

        # Verify LoRA configuration is loaded
        assert args.training.lora.enable is True
        assert args.training.lora.rank == 4
        assert args.training.lora.alpha == 8
        assert len(args.training.lora.target_modules) == 2

    def test_lora_save_load_cycle(self, tmp_path: str) -> None:
        """Test saving and loading LoRA weights."""
        model = SimpleModel()
        manager = LoraWeightManager(model)

        # Mock LoRA parameters
        model.lora_A_weight = nn.Parameter(torch.randn(4, 10))
        model.lora_B_weight = nn.Parameter(torch.randn(10, 4))

        # Save LoRA weights
        save_path = os.path.join(tmp_path, "lora_test")
        num_saved, num_elements = manager.save_lora_only(save_path)

        assert num_saved == 2
        assert num_elements > 0

        # Verify file was created
        assert os.path.exists(os.path.join(save_path, "lora_adapter.safetensors"))

    @pytest.mark.skipif(
        not dist.is_initialized() or dist.get_world_size() < 2,
        reason="Requires at least 2 GPUs for multi-card test"
    )
    def test_lora_multicard_compatibility(self) -> None:
        """Test LoRA compatibility with multi-card training."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        model = SimpleModel()
        manager = LoraWeightManager(model)

        # Each rank should have the same model structure
        num_params, num_elements = manager.get_lora_param_count()

        # Gather results from all ranks
        gathered_params = [None] * world_size
        dist.all_gather_object(gathered_params, num_params)

        if rank == 0:
            for i, params in enumerate(gathered_params):
                assert params == num_params, f"Rank {i} has different parameter count"

        dist.barrier()


class TestAddLoraToModel:
    """Test add_lora_to_model function."""

    def test_lora_injection(self) -> None:
        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
        )

        lora_param_names = [
            n for n, _ in model.named_parameters() if "lora" in n
        ]
        assert len(lora_param_names) > 0

        has_lora_a = any("lora_A" in n for n in lora_param_names)
        has_lora_b = any("lora_B" in n for n in lora_param_names)
        assert has_lora_a
        assert has_lora_b

    def test_base_params_frozen_after_injection(self) -> None:
        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
        )

        for name, param in model.named_parameters():
            if "lora" in name:
                assert param.requires_grad is True
            else:
                assert param.requires_grad is False

    def test_lora_params_dtype_float32(self) -> None:
        model = SimpleModel()
        model = model.to(torch.float16)
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
        )

        for name, param in model.named_parameters():
            if "lora" in name:
                assert param.dtype == torch.float32

    def test_lora_target_modules_support_valid(self) -> None:
        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"],
            lora_target_modules_support=["linear1", "linear2", "linear3"],
        )

        lora_params = [n for n, _ in model.named_parameters() if "lora" in n]
        assert len(lora_params) > 0

    def test_lora_target_modules_support_invalid(self) -> None:
        model = SimpleModel()
        freeze_parameters(model)
        with pytest.raises(ValueError, match="not in lora_target_modules_support"):
            add_lora_to_model(
                model, lora_rank=4, lora_alpha=8,
                lora_target_modules=["linear1", "linear2"],
                lora_target_modules_support=["linear1"],
            )

    def test_lora_alpha_stored_on_model(self) -> None:
        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=32,
            lora_target_modules=["linear1"],
        )

        assert model.lora_alpha == 32

    def test_pretrained_lora_loading(self, tmp_path: pathlib.Path) -> None:
        from safetensors.torch import save_file

        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
        )

        lora_state = {}
        for name, param in model.named_parameters():
            if "lora" in name and "base_layer" not in name:
                lora_state[f"base_model.model.{name}"] = param.data.clone()

        pretrained_path = os.path.join(tmp_path, "pretrained_lora.safetensors")
        save_file(lora_state, pretrained_path)

        model2 = SimpleModel()
        freeze_parameters(model2)
        model2 = add_lora_to_model(
            model2, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
            pretrained_lora_path=pretrained_path,
        )

        for name, param in model2.named_parameters():
            if "lora" in name and "base_layer" not in name:
                assert param.numel() > 0


class TestLoraWeightManagerExtended:
    """Extended tests for LoraWeightManager."""

    def _create_lora_model(self) -> nn.Module:
        model = SimpleModel()
        freeze_parameters(model)
        model = add_lora_to_model(
            model, lora_rank=4, lora_alpha=8,
            lora_target_modules=["linear1", "linear2"], lora_dropout=0.0,
        )
        return model

    def test_verify_lora_weights_valid(self) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        assert manager.verify_lora_weights() is True

    def test_verify_lora_weights_with_nan(self) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.data.fill_(float("nan"))
                break

        assert manager.verify_lora_weights() is False

    def test_verify_lora_weights_with_inf(self) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data.fill_(float("inf"))
                break

        assert manager.verify_lora_weights() is False

    def test_get_lora_param_count_with_lora(self) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        num_params, num_elements = manager.get_lora_param_count()

        assert num_params > 0
        assert num_elements > 0

    def test_get_lora_state_dict_with_lora(self) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        state_dict = manager.get_lora_state_dict()

        assert len(state_dict) > 0
        for name in state_dict:
            assert "lora" in name
            assert "base_layer" not in name

    def test_save_lora_with_iteration(self, tmp_path: pathlib.Path) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        save_path = os.path.join(tmp_path, "lora_iter")
        num_saved, num_elements = manager.save_lora_only(save_path, iteration=5)

        assert num_saved > 0
        assert num_elements > 0
        assert os.path.exists(
            os.path.join(save_path, "lora_adapter_iteration_5.safetensors")
        )

    def test_save_lora_without_iteration(self, tmp_path: pathlib.Path) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        save_path = os.path.join(tmp_path, "lora_default")
        num_saved, _ = manager.save_lora_only(save_path)

        assert num_saved > 0
        assert os.path.exists(
            os.path.join(save_path, "lora_adapter.safetensors")
        )

    def test_load_lora_weights(self, tmp_path: pathlib.Path) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        save_path = os.path.join(tmp_path, "lora_save")
        manager.save_lora_only(save_path)

        lora_file = os.path.join(save_path, "lora_adapter.safetensors")
        loaded_state = manager._load_state_dict(lora_file)
        assert len(loaded_state) > 0
        for key in loaded_state:
            assert "lora" in key

        model2 = self._create_lora_model()
        manager2 = LoraWeightManager(model2)
        manager2.load_lora_weights(lora_file)

    def test_load_lora_weights_file_not_found(self) -> None:
        model = SimpleModel()
        manager = LoraWeightManager(model)

        with pytest.raises(FileNotFoundError):
            manager.load_lora_weights("/nonexistent/path/lora.safetensors")

    def test_merge_lora_to_base_non_peft_model(self) -> None:
        model = SimpleModel()
        manager = LoraWeightManager(model)

        manager.merge_lora_to_base()

    def test_save_load_roundtrip_values(self, tmp_path: pathlib.Path) -> None:
        model = self._create_lora_model()
        manager = LoraWeightManager(model)

        original_state = manager.get_lora_state_dict()

        save_path = os.path.join(tmp_path, "roundtrip")
        manager.save_lora_only(save_path)

        lora_file = os.path.join(save_path, "lora_adapter.safetensors")
        loaded_state = load_state_dict(lora_file)

        for name in original_state:
            assert name in loaded_state
            assert torch.allclose(original_state[name], loaded_state[name])


def run_tests() -> None:
    """Run all tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
