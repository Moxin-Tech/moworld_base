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

"""Unit tests for LoRA utility functions.

This module provides unit tests for LoRA utility functions
that don't require full MindSpeed-MM dependencies.
"""

import math
import os

import pytest
import torch
import torch.nn as nn

from mindspeed_mm.fsdp.utils.lora_utils import (
    match_target_modules,
    validate_lora_config,
    get_lora_trainable_params,
    is_pattern_match,
)


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


class TestValidateLoraConfig:
    """Test LoRA configuration validation."""

    def test_valid_config(self) -> None:
        """Test validation of valid LoRA configuration."""
        validate_lora_config(
            rank=8,
            alpha=16,
            target_modules=["q_proj", "k_proj"],
            dropout=0.05,
            init_lora_weights=True,
        )

    def test_invalid_rank_zero(self) -> None:
        """Test validation fails with rank=0."""
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_lora_config(
                rank=0,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_invalid_rank_negative(self) -> None:
        """Test validation fails with negative rank."""
        with pytest.raises(ValueError, match="rank must be positive"):
            validate_lora_config(
                rank=-1,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_invalid_alpha_zero(self) -> None:
        """Test validation fails with alpha=0."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            validate_lora_config(
                rank=8,
                alpha=0,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_invalid_alpha_negative(self) -> None:
        """Test validation fails with negative alpha."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            validate_lora_config(
                rank=8,
                alpha=-1,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_invalid_target_modules_empty(self) -> None:
        """Test validation fails with empty target_modules."""
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=[],
                dropout=0.05,
                init_lora_weights=True,
            )

    def test_invalid_dropout_negative(self) -> None:
        """Test validation fails with negative dropout."""
        with pytest.raises(ValueError, match="dropout must be in \\[0, 1\\)"):
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=-0.1,
                init_lora_weights=True,
            )

    def test_invalid_dropout_too_high(self) -> None:
        """Test validation fails with dropout >= 1.0."""
        with pytest.raises(ValueError, match="dropout must be in \\[0, 1\\)"):
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=1.0,
                init_lora_weights=True,
            )

    def test_invalid_init_method(self) -> None:
        """Test validation fails with invalid init method."""
        with pytest.raises(ValueError, match="init_lora_weights must be True, False, one of"):
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights="invalid_method",
            )

    def test_valid_init_methods(self) -> None:
        """Test validation passes with all valid init methods."""
        valid_methods = ["gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"]
        for method in valid_methods:
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=method,
            )

        # Test pissa_niter_[number] format
        pissa_niter_values = ["pissa_niter_0", "pissa_niter_5", "pissa_niter_10", "pissa_niter_100"]
        for value in pissa_niter_values:
            validate_lora_config(
                rank=8,
                alpha=16,
                target_modules=["q_proj"],
                dropout=0.05,
                init_lora_weights=value,
            )

        # Test bool values
        validate_lora_config(
            rank=8,
            alpha=16,
            target_modules=["q_proj"],
            dropout=0.05,
            init_lora_weights=True,
        )
        validate_lora_config(
            rank=8,
            alpha=16,
            target_modules=["q_proj"],
            dropout=0.05,
            init_lora_weights=False,
        )

    def test_invalid_pissa_niter_format(self) -> None:
        """Test validation fails with invalid pissa_niter format."""
        invalid_pissa_values = ["pissa_niter_", "pissa_niter_abc", "pissa_niter_-1", "pissa_niter_1.5"]
        for value in invalid_pissa_values:
            with pytest.raises(ValueError, match="init_lora_weights must be True, False, one of"):
                validate_lora_config(
                    rank=8,
                    alpha=16,
                    target_modules=["q_proj"],
                    dropout=0.05,
                    init_lora_weights=value,
                )


class TestIsPatternMatch:
    """Test pattern matching logic."""

    def test_exact_match(self) -> None:
        """Test exact string matching."""
        assert is_pattern_match("q_proj", "q_proj") is True
        assert is_pattern_match("k_proj", "k_proj") is True

    def test_suffix_match(self) -> None:
        """Test suffix matching."""
        assert is_pattern_match("model.linear1", "linear1") is True
        assert is_pattern_match("model.linear2", "linear2") is True

    def test_wildcard_match(self) -> None:
        """Test wildcard pattern matching."""
        assert is_pattern_match("layers.0.q_proj", "layers.{*}.q_proj") is True
        assert is_pattern_match("layers.5.k_proj", "layers.{*}.k_proj") is True
        assert is_pattern_match("model.layers.10.mlp.gate", "model.layers.{*}.mlp.gate") is True

    def test_wildcard_no_match(self) -> None:
        """Test wildcard pattern not matching."""
        assert is_pattern_match("layers.0.q_proj", "layers.{*}.v_proj") is False
        assert is_pattern_match("model.linear1", "layers.{*}.q_proj") is False

    def test_exact_no_match(self) -> None:
        """Test exact string not matching."""
        assert is_pattern_match("q_proj", "k_proj") is False
        assert is_pattern_match("linear1", "linear2") is False


class TestMatchTargetModules:
    """Test target module matching."""

    def test_match_exact_names(self) -> None:
        """Test matching exact module names."""
        model = SimpleModel()
        patterns = ["linear1", "linear2"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 2
        assert "linear1" in matched
        assert "linear2" in matched

    def test_match_wildcard_pattern(self) -> None:
        """Test matching with wildcard patterns."""
        model = SimpleModel()
        patterns = ["linear{*}"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 2
        assert "linear1" in matched
        assert "linear2" in matched

    def test_match_mixed_patterns(self) -> None:
        """Test matching with mixed exact and wildcard patterns."""
        model = SimpleModel()
        patterns = ["linear1", "linear{*}"]
        matched = match_target_modules(model, patterns)

        # Should match both linear1 and linear2
        assert len(matched) == 2
        assert "linear1" in matched
        assert "linear2" in matched

    def test_no_match(self) -> None:
        """Test when no modules match."""
        model = SimpleModel()
        patterns = ["nonexistent", "another_nonexistent"]
        matched = match_target_modules(model, patterns)

        assert len(matched) == 0

    def test_nested_model_matching(self) -> None:
        """Test matching in nested model structure."""

        class NestedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(10, 20),
                    nn.Linear(20, 10),
                ])

        model = NestedModel()
        patterns = ["layers.{*}"]
        matched = match_target_modules(model, patterns)

        # Should match both layers
        assert len(matched) == 2
        assert any("layers.0" in m for m in matched)
        assert any("layers.1" in m for m in matched)


class TestGetLoraTrainableParams:
    """Test getting LoRA trainable parameters statistics."""

    def test_all_params_trainable(self) -> None:
        """Test when all parameters are trainable."""
        model = SimpleModel()

        trainable, total, stats = get_lora_trainable_params(model)

        # SimpleModel has:
        # - linear1.weight: 10*20 = 200
        # - linear1.bias: 20
        # - linear2.weight: 20*10 = 200
        # - linear2.bias: 10
        # Total: 430 params
        assert trainable > 0
        assert total > 0
        assert trainable == total  # All params trainable by default
        assert stats["trainable_params"] == trainable
        assert stats["total_params"] == total
        assert math.isclose(stats["trainable_ratio"], 1.0)
        assert stats["lora_params"] == 0  # No LoRA params
        assert stats["base_params"] == trainable

    def test_no_params_trainable(self) -> None:
        """Test when no parameters are trainable."""
        model = SimpleModel()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        trainable, total, stats = get_lora_trainable_params(model)

        assert trainable == 0
        assert total > 0
        assert stats["trainable_params"] == 0
        assert stats["total_params"] == total
        assert math.isclose(stats["trainable_ratio"], 0.0)
        assert stats["lora_params"] == 0
        assert stats["base_params"] == 0

    def test_partial_trainable(self) -> None:
        """Test when some parameters are trainable."""
        model = SimpleModel()

        # Freeze only linear1
        for param in model.linear1.parameters():
            param.requires_grad = False

        trainable, total, stats = get_lora_trainable_params(model)

        assert trainable > 0
        assert total > 0
        assert trainable < total
        assert 0 < stats["trainable_ratio"] < 1.0
        assert stats["lora_params"] == 0
        assert stats["base_params"] == trainable

    def test_lora_params(self) -> None:
        """Test with LoRA parameters."""
        model = SimpleModel()

        # Add mock LoRA parameters
        model.lora_A_weight = nn.Parameter(torch.randn(4, 10))
        model.lora_B_weight = nn.Parameter(torch.randn(10, 4))

        # Freeze base model parameters
        for param in model.linear1.parameters():
            param.requires_grad = False
        for param in model.linear2.parameters():
            param.requires_grad = False

        trainable, total, stats = get_lora_trainable_params(model)

        # LoRA params: 4*10 + 10*4 = 80
        assert trainable == 80
        assert stats["lora_params"] == 80
        assert stats["base_params"] == 0
        assert stats["trainable_ratio"] > 0


class TestLoraUtilsIntegration:
    """Integration tests for LoRA utilities."""

    def test_full_workflow(self) -> None:
        """Test complete LoRA workflow."""
        model = SimpleModel()

        # Step 1: Validate configuration
        validate_lora_config(
            rank=8,
            alpha=16,
            target_modules=["linear1", "linear2"],
            dropout=0.05,
            init_lora_weights=True,
        )

        # Step 2: Match target modules
        matched = match_target_modules(model, ["linear1", "linear2"])
        assert len(matched) == 2

        # Step 3: Get parameter statistics
        trainable, total, stats = get_lora_trainable_params(model)
        assert trainable > 0
        assert total > 0

        # Step 4: Freeze base model
        for param in model.parameters():
            param.requires_grad = False

        # Verify all parameters are frozen
        trainable_after_freeze, _, _ = get_lora_trainable_params(model)
        assert trainable_after_freeze == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
