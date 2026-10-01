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

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel

from bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict, StateSource


class PreTrainedBase(ABC):
    """
    Abstract base class for all pretrained models.

    """

    model_name_or_path: Union[str, Path]
    ARTIFACTS: ClassVar[List[str]] = []
    OPTIONAL_ARTIFACTS: ClassVar[List[str]] = []

    def __init__(self, **kwargs):
        self._state_dict_accessor: Optional[StateDict] = None
        self.init_kwargs = kwargs

        # File patterns used to copy files used for custom modeling, e.g.
        # modeling_*.py, configuration_*.py, tokenization_*.py,
        # processing_*.py, feature_extraction_*.py. HF repos
        # may have additional python files that are imported in the
        # aforementioned, so we just copy all python files
        # instead of above specific prefixes.
        # Currently, we can capture all json files via ARTIFACTS.
        self.custom_file_patterns = ["*.py"]

    def get_artifacts(self) -> Dict[str, str]:
        """Get the artifacts dictionary mapping artifact names to their attribute names."""
        return {artifact: f"_{artifact}" for artifact in self.ARTIFACTS}


    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Subclasses must implement this to load the main model."""
        pass

    @abstractmethod
    def _load_config(self) -> AutoConfig:
        """Subclasses must implement this to load the model config."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        """Lazily loads and returns the underlying model."""
        if not hasattr(self, "_model"):
            self._model = self._load_model()
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel):
        """Manually set the model."""
        self._model = value

    @property
    def config(self) -> AutoConfig:
        """Lazy load and return the model config."""
        if not hasattr(self, "_config"):
            self._config = self._load_config()
        return self._config

    @config.setter
    def config(self, value: AutoConfig):
        """Set the config manually."""
        self._config = value

    @property
    def state(self) -> StateDict:
        """
        Get the state dict accessor for pandas-like querying.

        This accessor can be backed by either a fully loaded model in memory
        or a ".safetensors" checkpoint on disk, enabling lazy loading of tensors.

        Examples:
            model.state()  # Get full state dict
            model.state["key"]  # Get single entry
            model.state[["key1", "key2"]]  # Get multiple entries
            model.state["*.weight"]  # Glob pattern
            model.state.regex(r".*\\.bias$")  # Regex pattern
        """
        if self._state_dict_accessor is None:
            source: Optional[Union[Dict[str, torch.Tensor], StateSource]] = None
            # Prioritize the loaded model's state_dict if available
            if hasattr(self, "_model") and self._model is not None:
                source = self.model.state_dict()
            elif hasattr(self, "model_name_or_path") and self.model_name_or_path:
                source = SafeTensorsStateSource(self.model_name_or_path)

            if source is None:
                raise ValueError(
                    "Cannot create StateDict accessor: model is not loaded and model_name_or_path is not set."
                )
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor
