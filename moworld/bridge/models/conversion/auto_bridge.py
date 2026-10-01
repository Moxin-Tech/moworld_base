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

import json
import os
from functools import cached_property
from pathlib import Path
from typing import Generic, Type, TypeVar, Union

import transformers
from megatron.core.transformer.module import MegatronModule
from transformers.configuration_utils import PretrainedConfig

from bridge.models.conversion import model_bridge
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

MegatronModelT = TypeVar("MegatronModelT", bound=MegatronModule)
DataclassT = TypeVar("DataclassT")

# Supported HuggingFace architecture suffixes for causal generation models
SUPPORTED_HF_ARCHITECTURES: tuple[str, ...] = (
    "ForCausalLM",
    "ForConditionalGeneration",
    "NemotronH_Nano_VL_V2",
)

CLASS_MODULE_MAPPING = {
    "WanTransformer3DModel": ("bridge.models", "WanTransformer3DModel"),
    "HunyuanVideo_1_5_DiffusionTransformer": ("bridge.models", "HunyuanVideo_1_5_DiffusionTransformer")
}


# Preformatted display string for error/help messages
SUPPORTED_HF_ARCHITECTURES_DISPLAY = " or ".join(f"'{s}'" for s in SUPPORTED_HF_ARCHITECTURES)


class AutoBridge(Generic[MegatronModelT]):

    def __init__(self, hf_pretrained: PreTrainedCausalLM | PretrainedConfig):
        if not isinstance(hf_pretrained, (PreTrainedCausalLM, PretrainedConfig)):
            raise ValueError("hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance")
        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = hf_pretrained

    @classmethod
    def from_hf_pretrained(cls, path: Union[str, Path], **kwargs) -> "AutoBridge":

        try:
            return cls(PreTrainedCausalLM.from_pretrained(path, **kwargs))
        except Exception as e:
            raise ValueError(f"Failed to load model with AutoBridge: {e}") from e

    def load_hf_weights(
            self,
            model: list[MegatronModelT],
            hf_path: str | Path | None = None,
            allowed_mismatched_params: list[str] | None = None,
    ) -> None:

        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance")
            pre_trained = self.hf_pretrained
        else:
            # Preserve trust_remote_code setting from the original bridge instance
            trust_remote_code = getattr(self.hf_pretrained, "trust_remote_code", False)
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path, trust_remote_code=trust_remote_code)
        self._model_bridge.load_weights_hf_to_megatron(
            pre_trained, model, allowed_mismatched_params=allowed_mismatched_params
        )

        return model

    @property
    def _model_bridge(self) -> "MegatronModelBridge":
        return model_bridge.get_model_bridge(self._model_architecture)

    @cached_property
    def _model_architecture(self):
        # Modification：Model Index for Generative Models
        config_path = os.path.join(self.hf_pretrained.model_name_or_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}

        if isinstance(config, dict) and "_class_name" in config:
            class_name = config["_class_name"]
            return self._resolve_generation_model_architecture(class_name, config)

        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            config = self.hf_pretrained.config
        else:
            config = self.hf_pretrained

        architectures = getattr(config, "architectures", [])

        if not architectures:
            raise ValueError(
                "\n✗ No architectures found in model config\n\n"
                "The model configuration does not specify any architectures.\n"
                "This is required for determining the model type."
            )

        causal_lm_arch = None
        for architecture_name in architectures:
            if architecture_name.endswith(SUPPORTED_HF_ARCHITECTURES):
                causal_lm_arch = architecture_name
                break

        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No CausalLM architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with {SUPPORTED_HF_ARCHITECTURES_DISPLAY}.\n"
                f"This bridge only supports causal language models.\n"
                f"For other model types, use a different bridge class."
            )

        try:
            return getattr(transformers, causal_lm_arch)
        except AttributeError as e:
            raise ValueError(
                f"\n✗ Architecture class '{causal_lm_arch}' not found in transformers\n\n"
                f"This could mean:\n"
                f"1. The model requires a newer version of transformers\n"
                f"2. The model uses a custom modeling file not in the standard library\n"
                f"3. There's a typo in the architecture name\n\n"
                f"Please verify your transformers installation and the model requirements."
            ) from e

    # Modification：Model Index for Generative Models
    def _resolve_generation_model_architecture(self, class_name: str, config) -> Type:
        if class_name not in CLASS_MODULE_MAPPING:
            raise KeyError(f"No mapping found for: {class_name}")

        try:
            import importlib
            module_name, actual_class_name = CLASS_MODULE_MAPPING[class_name]
            module = importlib.import_module(module_name)
            return getattr(module, actual_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Unable import {module_name}.{actual_class_name}: {e}") from e

    def _get_model_instance(self, model: list[MegatronModelT]) -> MegatronModelT:
        model_instance = model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        return model_instance

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        lines_for_build = []

        # Format hf_pretrained
        hf_repr_actual_lines = repr(self.hf_pretrained).splitlines()
        if hf_repr_actual_lines:
            # First line of hf_pretrained part
            lines_for_build.append(f"  (hf_pretrained): {hf_repr_actual_lines[0]}")
            # Subsequent lines of hf_pretrained part, indented
            for line in hf_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (hf_pretrained): ")  # Fallback for empty repr

        # Format model bridge
        mb_repr_actual_lines = repr(self._model_bridge).splitlines()
        if mb_repr_actual_lines:
            # First line of model bridge part
            lines_for_build.append(f"  (model_bridge): {mb_repr_actual_lines[0]}")
            # Subsequent lines of model bridge part, indented
            for line in mb_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (model_bridge): ")  # Fallback for empty repr

        return f"{class_name}(\n" + "\n".join(lines_for_build) + "\n)"
