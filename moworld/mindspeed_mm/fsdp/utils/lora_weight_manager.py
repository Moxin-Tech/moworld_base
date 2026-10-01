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

"""LoRA weight manager for FSDP2 distributed training.

This module provides utilities for saving and loading LoRA weights
in FSDP2 distributed training environments, including:
- Saving only LoRA adapter weights
- Saving full model with LoRA
- Loading pretrained LoRA weights
- Merging LoRA weights into base model
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from mindspeed.fsdp.utils.log import print_rank

try:
    from torch.distributed._tensor import DTensor
    DTENSOR_AVAILABLE = True
except ImportError:
    DTENSOR_AVAILABLE = False
    DTensor = None

logger = logging.getLogger(__name__)


class LoraWeightManager:
    """Manager for LoRA weight operations in FSDP2 training.

    This class handles saving and loading LoRA weights in distributed
    training environments, ensuring compatibility with FSDP2 parameter
    sharding and checkpoint formats.
    """

    def __init__(self, model: nn.Module):
        """Initialize the LoRA weight manager.

        Args:
            model: The PyTorch model with LoRA adapters.
        """
        self.model = model
        self._is_distributed = dist.is_initialized()
        self._rank = dist.get_rank() if self._is_distributed else 0
        self._world_size = dist.get_world_size() if self._is_distributed else 1

    def _gather_dtensor(self, param_data: Union[torch.Tensor, "DTensor"]) -> torch.Tensor:
        """Convert DTensor to regular tensor with full data.

        In FSDP2, model parameters are wrapped as DTensor (Distributed Tensor),
        where each rank only holds a sharded portion of the full tensor.
        This method gathers the full tensor from all ranks.

        Args:
            param_data: Parameter data, could be DTensor or regular Tensor.

        Returns:
            A regular torch.Tensor with full data on CPU.

        Note:
            - DTensor.full_tensor(): gathers data from all ranks, returns full tensor
            - DTensor.to_local(): returns local shard without communication
            - For saving LoRA weights, we need full_tensor() to get complete weights
        """
        if DTENSOR_AVAILABLE and isinstance(param_data, DTensor):
            full_tensor = param_data.full_tensor()
            return full_tensor.cpu()
        else:
            return param_data.cpu()

    def _is_dtensor(self, param_data: Union[torch.Tensor, "DTensor"]) -> bool:
        """Check if parameter data is a DTensor.

        Args:
            param_data: Parameter data to check.

        Returns:
            True if the data is a DTensor, False otherwise.
        """
        return DTENSOR_AVAILABLE and isinstance(param_data, DTensor)

    def save_lora_only(
        self,
        save_path: str,
        iteration: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Save only LoRA adapter weights.

        This method extracts LoRA parameters from the model and saves them
        in safetensors format. It handles FSDP2 sharding by gathering
        parameters from all ranks.

        Args:
            save_path: Directory path to save LoRA weights.
            iteration: Optional iteration number for checkpoint naming.

        Returns:
            Tuple of (num_saved_params, num_lora_params) where:
            - num_saved_params: Number of parameters saved
            - num_lora_params: Total number of LoRA parameters

        Raises:
            RuntimeError: If safetensors library is not installed.
        """
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise RuntimeError(
                "safetensors library is required for saving LoRA weights. "
                "Please install it with: pip install safetensors"
            ) from e

        os.makedirs(save_path, exist_ok=True)

        lora_state_dict: Dict[str, torch.Tensor] = {}
        num_lora_params = 0

        for name, param in self.model.named_parameters():
            if "lora" in name and "base_layer" not in name:
                gathered_param = self._gather_dtensor(param.data)
                lora_state_dict[name] = gathered_param
                num_lora_params += gathered_param.numel()

        if iteration is not None:
            filename = f"lora_adapter_iteration_{iteration}.safetensors"
        else:
            filename = "lora_adapter.safetensors"

        save_path_full = os.path.join(save_path, filename)
        save_file(lora_state_dict, save_path_full)

        num_saved_params = len(lora_state_dict)

        print_rank(
            logger.info,
            f"Saved {num_saved_params} LoRA parameters ({num_lora_params:,} elements) "
            f"to {save_path_full}"
        )

        return num_saved_params, num_lora_params

    def save_full_model_with_lora(
        self,
        save_path: str,
        iteration: Optional[int] = None,
    ) -> None:
        """Save full model including LoRA adapters.

        This method saves the complete model state including both base
        model weights and LoRA adapters. This is useful for checkpointing
        during training.

        Args:
            save_path: Directory path to save the model.
            iteration: Optional iteration number for checkpoint naming.

        Note:
            This method should be called through the training engine's
            save method, which handles optimizer and scheduler state.
        """
        print_rank(
            logger.info,
            f"Saving full model with LoRA to {save_path}"
        )

    def load_lora_weights(
        self,
        lora_path: str,
        strict: bool = False,
    ) -> Tuple[int, int, int]:
        """Load pretrained LoRA weights into the model.

        This method loads LoRA weights from a checkpoint file and injects
        them into the model. It handles both safetensors and PyTorch
        binary formats.

        Args:
            lora_path: Path to the LoRA weights file.
            strict: Whether to enforce strict key matching.

        Returns:
            Tuple of (num_loaded, num_missing, num_unexpected) where:
            - num_loaded: Number of parameters successfully loaded
            - num_missing: Number of missing keys
            - num_unexpected: Number of unexpected keys

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

        lora_state_dict = self._load_state_dict(lora_path)

        missing_keys, unexpected_keys = self.model.load_state_dict(
            lora_state_dict,
            strict=strict,
        )

        num_loaded = len(lora_state_dict) - len(missing_keys)
        num_missing = len(missing_keys)
        num_unexpected = len(unexpected_keys)

        print_rank(
            logger.info,
            f"Loaded {num_loaded} LoRA parameters from {lora_path}. "
            f"Missing: {num_missing}, Unexpected: {num_unexpected}"
        )

        if num_missing > 0 and strict:
            logger.warning(f"Missing keys: {missing_keys}")

        if num_unexpected > 0:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        return num_loaded, num_missing, num_unexpected

    def merge_lora_to_base(self) -> None:
        """Merge LoRA weights into the base model.

        This method merges the LoRA adapter weights into the base model
        weights, effectively applying the LoRA adaptation permanently.

        Note:
            After merging, the model will no longer have separate LoRA
            parameters. This operation is irreversible.
        """
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "PEFT library is required for merging LoRA weights. "
                "Please install it with: pip install peft"
            ) from e

        if not isinstance(self.model, PeftModel):
            logger.warning(
                "Model is not a PeftModel, cannot merge LoRA weights. "
                "Skipping merge operation."
            )
            return

        print_rank(logger.info, "Merging LoRA weights into base model...")
        self.model = self.model.merge_and_unload()
        print_rank(logger.info, "LoRA weights merged successfully")

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the current LoRA state dictionary.

        Returns:
            Dictionary mapping LoRA parameter names to their values.
        """
        lora_state_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if "lora" in name and "base_layer" not in name:
                gathered_param = self._gather_dtensor(param.data)
                lora_state_dict[name] = gathered_param

        return lora_state_dict

    def get_lora_param_count(self) -> Tuple[int, int]:
        """Get the count of LoRA parameters.

        Returns:
            Tuple of (num_params, num_elements) where:
            - num_params: Number of LoRA parameter tensors
            - num_elements: Total number of elements in all LoRA parameters
        """
        num_params = 0
        num_elements = 0

        for name, param in self.model.named_parameters():
            if "lora" in name and "base_layer" not in name:
                num_params += 1
                num_elements += param.numel()

        return num_params, num_elements

    def _load_state_dict(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load state dictionary from a checkpoint file.

        Args:
            file_path: Path to the checkpoint file.

        Returns:
            State dictionary mapping parameter names to tensors.
        """
        if file_path.endswith(".safetensors"):
            return self._load_safetensors(file_path)
        else:
            return self._load_pytorch_bin(file_path)

    def _load_safetensors(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load state dictionary from a safetensors file.

        Args:
            file_path: Path to the safetensors file.

        Returns:
            State dictionary mapping parameter names to tensors.
        """
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise RuntimeError(
                "safetensors library is required. "
                "Please install it with: pip install safetensors"
            ) from e

        state_dict: Dict[str, torch.Tensor] = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        return state_dict

    def _load_pytorch_bin(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load state dictionary from a PyTorch binary file.

        Args:
            file_path: Path to the binary file.

        Returns:
            State dictionary mapping parameter names to tensors.
        """
        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Expected state dictionary, got {type(state_dict)}"
            )

        return state_dict

    def verify_lora_weights(self) -> bool:
        """Verify that LoRA weights are properly initialized.

        Returns:
            True if LoRA weights are valid, False otherwise.
        """
        lora_params = [
            (name, param)
            for name, param in self.model.named_parameters()
            if "lora" in name and "base_layer" not in name
        ]

        if not lora_params:
            logger.warning("No LoRA parameters found in model")
            return False

        for name, param in lora_params:
            param_data = self._gather_dtensor(param.data) if self._is_dtensor(param.data) else param.data
            if param_data.isnan().any():
                logger.error(f"LoRA parameter {name} contains NaN values")
                return False

            if param_data.isinf().any():
                logger.error(f"LoRA parameter {name} contains Inf values")
                return False

        print_rank(
            logger.info,
            f"Verified {len(lora_params)} LoRA parameters - all valid"
        )

        return True
