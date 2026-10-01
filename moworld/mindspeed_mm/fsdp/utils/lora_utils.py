# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""LoRA utilities for FSDP2 training framework.

This module provides utilities for integrating LoRA (Low-Rank Adaptation)
with FSDP2 distributed training, including:
- LoRA adapter injection
- Target module pattern matching
- Configuration validation
- Parameter statistics
"""

import fnmatch
import logging
import re
from typing import List, Optional, Set, Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def freeze_parameters(model: nn.Module) -> None:
    """Freeze all parameters in the model.

    Args:
        model: The PyTorch model to freeze.
    """
    model.requires_grad_(False)
    model.eval()
    model.train()


def match_target_modules(model: nn.Module, patterns: List[str]) -> List[str]:
    """Match target modules using wildcard patterns.

    Supports both exact matching and wildcard patterns (e.g., "language_model.layers.{*}.q_proj").

    Args:
        model: The PyTorch model to search.
        patterns: List of module name patterns to match.

    Returns:
        List of matched module names.

    Examples:
        >>> patterns = ["q_proj", "language_model.layers.{*}.self_attn"]
        >>> matched = match_target_modules(model, patterns)
    """
    matched_modules: List[str] = []

    for name, module in model.named_modules():
        for pattern in patterns:
            if is_pattern_match(name, pattern):
                matched_modules.append(name)
                break

    return matched_modules


def is_pattern_match(module_name: str, pattern: str) -> bool:
    """Check if a module name matches a pattern.

    Supports:
    - Exact matching: "q_proj" matches "q_proj"
    - Wildcard matching: "layers.{*}.q_proj" matches "layers.0.q_proj", "layers.1.q_proj"

    Args:
        module_name: Full module name to check.
        pattern: Pattern to match against.

    Returns:
        True if the module name matches the pattern.
    """
    if "{*}" in pattern:
        pattern_regex = pattern.replace("{*}", "*")
        return fnmatch.fnmatch(module_name, pattern_regex)
    else:
        return module_name == pattern or module_name.endswith("." + pattern)


def validate_lora_config(
    rank: int,
    alpha: int,
    target_modules: List[str],
    dropout: float,
    init_lora_weights: bool | str,
) -> None:
    """Validate LoRA configuration parameters.

    Args:
        rank: LoRA rank.
        alpha: LoRA alpha scaling factor.
        target_modules: List of target module patterns.
        dropout: Dropout rate.
        init_lora_weights: Weight initialization method (True, False, or str).

    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")

    if alpha <= 0:
        raise ValueError(f"LoRA alpha must be positive, got {alpha}")

    if not target_modules:
        raise ValueError("target_modules cannot be empty")

    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}")

    valid_init_methods = [
        "gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"
    ]
    pissa_niter_pattern = re.compile(r"^pissa_niter_\d+$")
    if isinstance(init_lora_weights, str):
        init_val = init_lora_weights.lower()
        if init_val not in valid_init_methods and not pissa_niter_pattern.match(init_val):
            raise ValueError(
                f"init_lora_weights must be True, False, one of {valid_init_methods}, "
                f"or 'pissa_niter_[number of iters]' (e.g., 'pissa_niter_5'), "
                f"got {init_lora_weights}"
            )
    elif not isinstance(init_lora_weights, bool):
        raise ValueError(
            f"init_lora_weights must be bool or str, got {type(init_lora_weights)}"
        )


def get_lora_trainable_params(model: nn.Module) -> Tuple[int, int, Dict[str, Any]]:
    """Get statistics about LoRA trainable parameters.

    Args:
        model: The PyTorch model with LoRA adapters.

    Returns:
        Tuple of (trainable_params, total_params, stats_dict) where:
        - trainable_params: Number of trainable parameters
        - total_params: Total number of parameters
        - stats_dict: Dictionary with detailed statistics
    """
    trainable_params = 0
    total_params = 0
    lora_params = 0
    base_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

            if "lora" in name:
                lora_params += param.numel()
            else:
                base_params += param.numel()

    stats_dict: Dict[str, Any] = {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
        "lora_params": lora_params,
        "base_params": base_params,
    }

    return trainable_params, total_params, stats_dict


def add_lora_to_model(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    init_lora_weights: bool | str = True,
    pretrained_lora_path: Optional[str] = None,
    lora_target_modules_support: Optional[List[str]] = None,
) -> nn.Module:
    """Add LoRA adapters to a PyTorch model.

    This function injects LoRA adapters into the specified target modules,
    optionally loads pretrained LoRA weights, and ensures proper dtype handling.

    Args:
        model: The PyTorch model to add LoRA to.
        lora_rank: LoRA rank (dimension of the low-rank matrices).
        lora_alpha: LoRA alpha scaling factor.
        lora_target_modules: List of target module names/patterns.
        lora_dropout: Dropout rate for LoRA layers.
        init_lora_weights: Weight initialization method (True, False, or str).
        pretrained_lora_path: Path to pretrained LoRA weights (optional).
        lora_target_modules_support: List of supported module types for validation.

    Returns:
        The model with LoRA adapters injected.

    Raises:
        ImportError: If PEFT library is not installed.
        ValueError: If target modules are not supported.
    """
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as e:
        raise ImportError(
            "PEFT library is required for LoRA training. "
            "Please install it with: pip install peft"
        ) from e

    model.lora_alpha = lora_alpha

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    if lora_target_modules_support is not None:
        for lora_target_module in lora_config.target_modules:
            if lora_target_module not in lora_target_modules_support:
                raise ValueError(
                    f"lora_target_module {lora_target_module} not in "
                    f"lora_target_modules_support"
                )

    model = inject_adapter_in_model(lora_config, model)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype=torch.float32)

    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        logger.info(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. "
            f"{num_unexpected_keys} parameters are unexpected."
        )

    return model


def load_state_dict(file_path: str, torch_dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Load state dictionary from a checkpoint file.

    Supports both safetensors (.safetensors) and PyTorch binary (.pt/.bin) formats.

    Args:
        file_path: Path to the checkpoint file.
        torch_dtype: Optional dtype to cast tensors to.

    Returns:
        State dictionary mapping parameter names to tensors.
    """
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load state dictionary from a safetensors file.

    Args:
        file_path: Path to the safetensors file.
        torch_dtype: Optional dtype to cast tensors to.

    Returns:
        State dictionary mapping parameter names to tensors.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "safetensors library is required. "
            "Please install it with: pip install safetensors"
        ) from e

    state_dict: Dict[str, torch.Tensor] = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Load state dictionary from a PyTorch binary file.

    Args:
        file_path: Path to the binary file.
        torch_dtype: Optional dtype to cast tensors to.

    Returns:
        State dictionary mapping parameter names to tensors.
    """
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def print_lora_config(
    rank: int,
    alpha: int,
    target_modules: List[str],
    dropout: float,
    init_lora_weights: bool | str,
    trainable_params: int,
    total_params: int,
) -> None:
    """Print LoRA configuration summary.

    Args:
        rank: LoRA rank.
        alpha: LoRA alpha.
        target_modules: List of target modules.
        dropout: Dropout rate.
        init_lora_weights: Initialization method (True, False, or str).
        trainable_params: Number of trainable parameters.
        total_params: Total number of parameters.
    """
    logger.info("=" * 60)
    logger.info("LoRA Configuration:")
    logger.info(f"  Enabled: True")
    logger.info(f"  Rank: {rank}")
    logger.info(f"  Alpha: {alpha}")
    logger.info(f"  Target modules: {target_modules}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Init weights: {init_lora_weights}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    if total_params > 0:
        logger.info(f"  Trainable ratio: {trainable_params / total_params:.2%}")
    logger.info("=" * 60)
