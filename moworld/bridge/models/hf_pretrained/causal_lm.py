#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)
from transformers.generation.utils import GenerateOutput

from bridge.models.hf_pretrained.base import PreTrainedBase
from bridge.models.hf_pretrained.safe_config_loader import safe_load_config_with_retry


# Python 3.12+ supports PEP 692 (TypedDict Unpack)
if sys.version_info >= (3, 12):
    from typing import TypedDict, Unpack
else:
    from typing_extensions import TypedDict, Unpack


CausalLMType = TypeVar("CausalLMType", bound=AutoModelForCausalLM)


class PreTrainedCausalLM(PreTrainedBase, Generic[CausalLMType]):
    """
    A generic class for Pretrained Causal Language Models with lazy loading.

    """

    ARTIFACTS = ["tokenizer"]
    OPTIONAL_ARTIFACTS = ["generation_config", "processor", "image_processor"]

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize a Pretrained Causal LM with lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on (e.g., 'cuda', 'cpu')
            torch_dtype: Data type to load model in (e.g., torch.float16)
            trust_remote_code: Whether to trust remote code when loading
            **kwargs: Additional arguments passed to from_pretrained methods
        """
        self._model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        super().__init__(**kwargs)
        # Store the original source path for custom modeling file preservation
        if model_name_or_path and trust_remote_code:
            self._original_source_path = model_name_or_path

    def _load_model(self) -> CausalLMType:
        """Load the model."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load model")

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            **self.init_kwargs,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        config = getattr(self, "_config", None)
        if config is not None:
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        model = model.to(self.device)

        generation_config = getattr(self, "_generation_config", None)
        if generation_config is not None and hasattr(model, "generation_config"):
            model.generation_config = generation_config
        return model

    def _load_config(self) -> AutoConfig:
        """Load the model config with thread-safety protection."""
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path must be provided to load config")
        return safe_load_config_with_retry(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            **self.init_kwargs,
        )

    @property
    def auto_map_model_class(self) -> Optional[str]:
        """Get the AutoModelForCausalLM class from the config."""
        config = self.config
        auto_map = getattr(config, "auto_map", None)
        if auto_map and "AutoModelForCausalLM" in auto_map:
            auto_map_class = auto_map["AutoModelForCausalLM"]
            return str(auto_map_class)

        return None


    @property
    def model_name_or_path(self) -> Optional[Union[str, Path]]:
        """Return the model name or path."""
        return self._model_name_or_path


    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "PreTrainedCausalLM[CausalLMType]":
        """
        Create a PreTrainedCausalLM instance for lazy loading.

        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to load model on
            torch_dtype: Data type to load model in
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for from_pretrained methods

        Returns:
            PreTrainedCausalLM instance configured for lazy loading
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Get model's dtype if loaded."""
        if self.has_model:
            try:
                return next(self.model.parameters()).dtype
            except StopIteration:
                return None
        return None

    @property
    def num_parameters(self) -> Optional[int]:
        """Get total number of parameters if model is loaded."""
        if self.has_model:
            return sum(p.numel() for p in self.model.parameters())
        return None
