# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "AutoBridge",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "ColumnParallelMapping",
    "GatedMLPMapping",
    "MegatronParamMapping",
    "QKVMapping",
    "ReplicatedMapping",
    "RowParallelMapping",
    "AutoMapping",
    # VL Models
    "Qwen25VLBridge",
    "Qwen3VLMoEBridge",
    "Qwen3VLBridge",
    # wan Models
    "Wan21Bridge",
    "WanTransformer3DModel",
    "HunyuanVideo_1_5_DiffusionTransformer"
]

from bridge.models.conversion.auto_bridge import AutoBridge
from bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from bridge.models.conversion.model_bridge import MegatronModelBridge
from bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from bridge.models.qwen_vl import (
    Qwen25VLBridge,
    Qwen3VLMoEBridge,
    Qwen3VLBridge,
)
from bridge.models.wan import (
    Wan21Bridge,
    WanTransformer3DModel,
)

from bridge.models.hunyuan import HunyuanVideo_1_5_DiffusionTransformer
