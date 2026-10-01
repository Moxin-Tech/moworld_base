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

"""LoRA configuration arguments for FSDP2 training.

This module defines the dataclass for LoRA-specific configuration
parameters used in FSDP2 distributed training.
"""

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from mindspeed_mm.config.arguments.base_args import BaseArguments


class LoraArguments(BaseArguments):
    """Configuration arguments for LoRA (Low-Rank Adaptation) training.

    This class contains all parameters needed to configure LoRA adapters
    for efficient fine-tuning of large models.

    Attributes:
        enable: Whether to enable LoRA fine-tuning.
        rank: Rank of the low-rank matrices.
        alpha: Scaling factor for LoRA weights.
        target_modules: List of target module names/patterns for LoRA.
        dropout: Dropout rate for LoRA layers.
        init_lora_weights: Weight initialization method.
        pretrained_lora_path: Path to pretrained LoRA weights (optional).
        lora_target_modules_support: List of supported module types.
    """
    enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA fine-tuning."},
    )
    rank: int = field(
        default=8,
        metadata={"help": "Rank of the low-rank matrices."},
    )
    alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA weights."},
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        metadata={
            "help": "List of target module names/patterns for LoRA. "
            "Supports wildcard patterns (e.g., 'language_model.layers.{*}.q_proj')."
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for LoRA layers."},
    )
    init_lora_weights: (
            bool
            | Literal[
                "gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    ) = field(
        default=True,
        metadata={
            "help": "How to initialize the weights of the LoRA layers. ",
        },
    )
    pretrained_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained LoRA weights to load."},
    )
    lora_target_modules_support: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of supported module types for validation. "
            "If None, validation is skipped."
        },
    )

    def model_post_init(self, __context):
        """Validate LoRA configuration after initialization."""
        if self.enable:
            if self.rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.rank}")

            if self.alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")

            if not self.target_modules:
                raise ValueError("target_modules cannot be empty when LoRA is enabled")

            if not 0.0 <= self.dropout < 1.0:
                raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")

            valid_init_methods = [
                "gaussian", "eva", "olora", "pissa", "corda", "loftq", "orthogonal"
            ]
            pissa_niter_pattern = re.compile(r"^pissa_niter_\d+$")
            if isinstance(self.init_lora_weights, str):
                init_val = self.init_lora_weights.lower()
                if init_val not in valid_init_methods and not pissa_niter_pattern.match(init_val):
                    raise ValueError(
                        f"init_lora_weights must be True, False, one of {valid_init_methods}, "
                        f"or 'pissa_niter_[number of iters]' (e.g., 'pissa_niter_5'), "
                        f"but got {self.init_lora_weights}"
                    )
            elif not isinstance(self.init_lora_weights, bool):
                raise ValueError(
                    f"init_lora_weights must be bool or str, got {type(self.init_lora_weights)}"
                )
