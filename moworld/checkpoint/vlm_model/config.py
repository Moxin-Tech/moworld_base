#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : config.py
@Time    : 2025/01/14
@Desc    : 模型相关的配置定义
"""
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional, List, Any, Tuple

from pydantic import BaseModel, DirectoryPath, PositiveInt, NonNegativeInt, model_validator, computed_field
from transformers import PretrainedConfig, AutoConfig


class ParallelConfig(BaseModel):
    """权模型切分配置，包括tp的size，以及pp切分时vit和llm在pp域每张卡上切分的层数"""

    llm_pp_layers: List[NonNegativeInt]
    """llm模块pipeline parallel切分每张卡上切分几层"""

    vit_pp_layers: List[NonNegativeInt]
    """vit模块pipeline parallel切分每张卡上切分几层"""

    audio_pp_layers: Optional[List[NonNegativeInt]] = None
    """audio模块pipeline parallel切分每张卡上切分几层"""

    tp_size: PositiveInt = 1
    """默认tensor parallel张量并行组，模型转换时不同的tp组要切分到不同的目录下"""

    vit_tp_size: Optional[NonNegativeInt] = 0
    """异构并行下,视频的tensor parallel张量并行组, 为0时使用默认tp_size配置"""

    audio_tp_size: Optional[NonNegativeInt] = 0
    """异构并行下,音频的tensor parallel张量并行组, 为0时使用默认tp_size配置"""

    ep_size: PositiveInt = 1
    """expert parallel张量并行组，模型转换时不同的ep组要切分到不同的目录下"""

    @computed_field
    @cached_property
    def pp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers)

    def is_pp(self) -> bool:
        return self.pp_size > 1

    @model_validator(mode='after')
    def validate_pp_layers(self) -> "ParallelConfig":
        if len(self.vit_pp_layers) != len(self.llm_pp_layers):
            warnings.warn("The pp_layers of vit and llm are inconsistent", UserWarning)
        if len(self.vit_pp_layers) < 1:
            raise ValueError("pp layers长度至少为1")
        return self


class VppParallelConfig(BaseModel):
    """权模型切分配置，包括tp的size，以及pp切分时vit和llm在pp域每张卡上切分的层数"""

    llm_pp_layers: List[List[NonNegativeInt]]
    """llm模块pipeline parallel切分每张卡上切分几层, vpp切分配置参考docs/zh/features/virtual_pipeline_parallel.md"""

    vit_pp_layers: List[List[NonNegativeInt]]
    """vit模块pipeline parallel切分每张卡上切分几层, vpp切分配置参考docs/zh/features/virtual_pipeline_parallel.md"""

    audio_pp_layers: Optional[List[List[NonNegativeInt]]] = None
    """audio模块pipeline parallel切分每张卡上切分几层, vpp切分配置参考docs/zh/features/virtual_pipeline_parallel.md"""

    tp_size: PositiveInt = 1
    """默认tensor parallel张量并行组，模型转换时不同的tp组要切分到不同的目录下"""

    vit_tp_size: Optional[NonNegativeInt] = 0
    """异构并行下,视频的tensor parallel张量并行组, 为0时使用默认tp_size配置"""

    audio_tp_size: Optional[NonNegativeInt] = 0
    """异构并行下,音频的tensor parallel张量并行组, 为0时使用默认tp_size配置"""

    ep_size: Optional[PositiveInt] = 1
    """expert parallel专家并行组，模型转换时不同的ep组要切分到不同的目录下"""

    @computed_field
    @cached_property
    def pp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers[0])

    @computed_field
    @cached_property
    def vit_pp_size(self) -> PositiveInt:
        return len(self.vit_pp_layers[0])

    def is_pp(self) -> bool:
        return self.pp_size > 1

    @computed_field
    @cached_property
    def vpp_size(self) -> PositiveInt:
        return len(self.llm_pp_layers)

    def is_vpp(self) -> bool:
        return self.vpp_size > 1

    @model_validator(mode='after')
    def validate_pp_layers(self) -> "VppParallelConfig":
        if len(self.vit_pp_layers) != len(self.llm_pp_layers):
            warnings.warn("The pp_layers of vit and llm are inconsistent", UserWarning)
        if len(self.vit_pp_layers) < 1:
            raise ValueError("pp layers长度至少为1")
        return self

    @model_validator(mode='after')
    def validate_vpp_layers(self) -> "VppParallelConfig":
        pp_size = self.pp_size
        vit_pp_size = self.vit_pp_size
        for vpp in self.llm_pp_layers:
            if len(vpp) != pp_size:
                raise ValueError("llm的每个vpp配置长度一定要一致")
        for vpp in self.vit_pp_layers:
            if len(vpp) != vit_pp_size:
                raise ValueError("vit的每个vpp配置长度一定要一致")
        return self


class HfConfig(BaseModel):
    """huggingface下载的开源权重的配置，主要包括路径校验及AutoConfig"""
    hf_dir: DirectoryPath
    """huggingface下载的路径"""

    @cached_property
    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(self.hf_dir, local_files_only=True)

    @model_validator(mode='after')
    def validate_hf_dir(self) -> "HfConfig":
        safetensors_files = list(self.hf_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No *.safetensors files found in {self.hf_dir}")
        if not list(self.hf_dir.glob("config.json")):
            raise FileNotFoundError(f"No config.json in {self.hf_dir}")
        return self


class HFLoRAConfig(BaseModel):
    """huggingface的LoRA权重的配置, 包括路径校验"""
    hf_dir: DirectoryPath
    """huggingface的LoRA权重路径"""

    @model_validator(mode='after')
    def validate_hf_dir(self) -> "HFLoRAConfig":
        safetensors_files = list(self.hf_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No *.safetensors files found in {self.hf_dir}")
        return self


# BaseModel/dataclasses注意要在field的下一行添加描述说明
class ConvertMMConfig(BaseModel):
    """huggingface权重转换为mindspeed-mm权重配置"""

    mm_dir: Path
    """mm保存的路径"""

    llm_hf_config: Optional[HfConfig] = None
    """hf下载的llm权重路径配置"""

    parallel_config: ParallelConfig
    """并行配置"""

    hf_config: HfConfig
    """hf下载的原始权重路径配置"""

    trust_remote_code: bool = False
    """trust_remote_code 默认设为False, 需要用户手动设为True"""


class ConvertHFConfig(ConvertMMConfig):
    """mindspeed-mm训练出来的权重转换为huggingface格式权重的配置"""
    save_hf_dir: Path
    """mm转回hf格式时保存的路径"""


class ConvertResplitConfig(BaseModel):
    """mindspeed-mm训练出来的权重pp重切分的配置，pp重切分一般用在推理时等设备变化的场景"""
    source_dir: DirectoryPath
    """原始训练出来的权重路径"""

    target_dir: Path
    """重切分后保存的权重路径"""

    source_parallel_config: ParallelConfig
    """原始训练出权重的并行配置"""

    target_parallel_config: ParallelConfig
    """重切分后的权重的并行配置"""

    @model_validator(mode='after')
    def validate_hf_dir(self) -> "ConvertResplitConfig":
        if sum(self.source_parallel_config.vit_pp_layers) != sum(self.target_parallel_config.vit_pp_layers):
            raise ValueError("vit pp layers not equal!")
        if sum(self.source_parallel_config.llm_pp_layers) != sum(self.target_parallel_config.llm_pp_layers):
            raise ValueError("llm pp layers not equal!")
        return self


class CommonModelConfig(BaseModel):
    """权重转换框架公共的模型配置"""

    num_experts: int = 0
    """专家并行数,表示模型中并行使用的专家数量"""
    #
    num_key_value_heads: int = 0
    """多头注意力机制中的键值头数量"""

    tie_word_embeddings: bool = False
    """是否将词嵌入层和输出层的权重进行绑定"""

    llm_hf_dir: Optional[Path] = None
    """LLM模型的Huggingface语言模型的存储目录路径"""

    llm_num_layers: int = 0
    """llm模型层数"""

    vit_num_layers: int = 0
    """vit模型层数"""

    audio_num_layers: Optional[int] = None
    """audio模型层数"""


# BaseModel/dataclasses注意要在field的下一行添加描述说明
class ConvertVppMMConfig(BaseModel):
    """huggingface权重转换为mindspeed-mm权重配置"""

    mm_dir: Path
    """mm保存的路径"""

    parallel_config: VppParallelConfig
    """并行配置"""

    hf_config: HfConfig
    """hf下载的原始权重路径配置"""

    llm_hf_config: Optional[HfConfig] = None
    """hf下载的llm权重路径配置"""

    save_vit_only: Optional[bool] = False
    """是否只保存vit部分（包含projector）的权重，默认为False，同时保存llm和vit的权重"""

    trust_remote_code: bool = False
    """trust_remote_code 默认设为False, 需要用户手动设为True"""

    common_model_config: CommonModelConfig = CommonModelConfig()
    """权重转换框架公共的模型配置"""


    @model_validator(mode='after')
    def validate_parallel_policies(self) -> "ConvertVppMMConfig":
        """
        1.Verify whether the TP segmentation meets the number of kv heads
        2.It is verified that the PP and VPP segmentation meet the requirements of the number of model layers
        """
        # Verify the tp split configuration
        if self.common_model_config.num_key_value_heads % self.parallel_config.tp_size != 0:
            raise ValueError(
                f"Number of key-value heads ({self.common_model_config.num_key_value_heads}) must be divisible by TP size ({self.parallel_config.tp_size})"
            )
        # Flatten the vit and llm layers for VPP
        if self.common_model_config.vit_num_layers is None or self.common_model_config.llm_num_layers is None:
            raise AttributeError("Required vision or LLM layer config not found in any model type.")

        vit_pipeline_num_layers_flat = [
            item
            for sublist in self.parallel_config.vit_pp_layers
            for item in sublist
        ]
        llm_pipeline_num_layers_flat = [
            item
            for sublist in self.parallel_config.llm_pp_layers
            for item in sublist
        ]
        # Validation for flattened lists
        expected_length = self.parallel_config.pp_size * self.parallel_config.vpp_size
        if len(vit_pipeline_num_layers_flat) != expected_length:
            warnings.warn("VIT use hetero PP layers...", UserWarning)
        if sum(vit_pipeline_num_layers_flat) != self.common_model_config.vit_num_layers:
            raise AssertionError(f'Sum of vit_pipeline_num_layers_flat must be equal to vit_num_layers, '
                                 f'but got {sum(vit_pipeline_num_layers_flat)} and {self.common_model_config.vit_num_layers}.')
        if len(llm_pipeline_num_layers_flat) != expected_length:
            raise AssertionError(f'Length of llm_pipeline_num_layers_flat must be equal to pp_size * vp_size, '
                                 f'but got {len(llm_pipeline_num_layers_flat)} and {expected_length}.')
        if sum(llm_pipeline_num_layers_flat) != self.common_model_config.llm_num_layers:
            raise AssertionError(f'Sum of llm_pipeline_num_layers_flat must be equal to llm_num_layers, '
                                 f'but got {sum(llm_pipeline_num_layers_flat)} and {self.common_model_config.llm_num_layers}.')
        if self.common_model_config.audio_num_layers is not None:
            audio_pipeline_num_layers_flat = [
                item
                for sublist in self.parallel_config.audio_pp_layers
                for item in sublist
            ]
            if len(audio_pipeline_num_layers_flat) != expected_length:
                warnings.warn("Audio model use hetero PP layers...", UserWarning)
            if sum(audio_pipeline_num_layers_flat) != self.common_model_config.audio_num_layers:
                raise AssertionError(f'Sum of audio_pipeline_num_layers_flat must be equal to audio_num_layers, '
                                     f'but got {sum(audio_pipeline_num_layers_flat)} and {self.common_model_config.audio_num_layers}.')
        return self


class ConvertHFLoRAConfig(BaseModel):
    mm_dir: Path
    """mm的LoRA权重保存路径"""

    parallel_config: VppParallelConfig
    """并行配置"""

    hf_config: HFLoRAConfig
    """hf的LoRA权重路径配置"""

    save_vit_only: Optional[bool] = False
    """是否只保存vit部分(包含projector)的权重, 默认为False, 同时保存llm和vit的权重"""


class ConvertTorchDCPConfig(BaseModel):
    """Config for convert huggingface ckpt to mindspeed-mm torch dcp ckpt"""

    mm_dir: Path

    hf_config: HfConfig

    # trust_remote_code default false, user need to set true when use
    trust_remote_code: bool = False

    # common config
    common_model_config: CommonModelConfig = CommonModelConfig()


def get_first_available(
        model_cfg: Any,
        candidates: List[Tuple[List[str], str]]
):
    """
    安全地按优先级尝试多个属性路径，返回第一个存在的属性值，否则返回 None。

    参数:
        model_cfg (Any):
            包含嵌套配置的模型配置对象（通常为 HuggingFace 模型的 config 对象）。
        candidates (List[Tuple[List[str], str]]):
            优先级排序的候选路径列表，每个元素为 (属性路径, 目标属性名)。
            属性路径是嵌套属性的层级列表，例如 ['vision_config', 'depth'] 对应 model_cfg.vision_config.depth。

    返回:
        Optional[Union[int, float, str, dict, list]]:
            第一个有效路径对应的属性值，如果所有路径均无效则返回 None。

    """
    for path, attr in candidates:
        current = model_cfg
        try:
            # 逐级访问嵌套属性
            for step in path:
                current = getattr(current, step)
            return getattr(current, attr)
        except AttributeError:
            continue
    return None


class ConvertVppHFConfig(ConvertVppMMConfig):
    """mindspeed-mm训练出来的权重转换为huggingface格式权重的配置"""
    save_hf_dir: Path
    """mm转回hf格式时保存的路径"""
