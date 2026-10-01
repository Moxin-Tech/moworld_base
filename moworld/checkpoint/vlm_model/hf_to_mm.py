#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : hf_to_mm.py
@Time    : 2025/01/14
@Desc    : qwen2vl huggingface模型转换成mindspeed-mm模型

huggingface模型目录：
Qwen2-VL-7B-Instruct/
├── chat_template.json
├── config.json
├── configuration.json
├── generation_config.json
├── LICENSE
├── merges.txt
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json
├── preprocessor_config.json
├── README.md
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json

mindspeed-mm模型目录(这里是tp1/pp4训练保存的模型)：
Qwen2-VL-7B-Instruct/
├── latest_checkpointed_iteration.txt
└── release
    ├── mp_rank_00_000
    │    └── model_optim_rng.pt
    ├── mp_rank_00_001
    │    └── model_optim_rng.pt
    ├── mp_rank_00_002
    │    └── model_optim_rng.pt
    └── mp_rank_00_003
        └── model_optim_rng.pt

"""
import re
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Callable, Any, List, Dict, Optional, Union, Tuple

import numpy as np
import torch
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.checkpoint import FileSystemWriter
from safetensors.torch import load_file
from tqdm import tqdm

from checkpoint.common.constant import LATEST_TXT, MEGATRON_CKPT_NAME, IMAGE_ENCODER, AUDIO_ENCODER, TEXT_DECODER
from checkpoint.common.mm_types import STATE_DICT_T, VPP_LAYER_NUM_T
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertTorchDCPConfig
from checkpoint.vlm_model.operator import Operator, TieOp, TP_PATTERN_T


CHECKPOINT_VERSION_KEY = "checkpoint_version"
CHECKPOINT_VERSION_VALUE = 3.0


@dataclass
class PPStageSchema:
    """When splitting different modules such as vit/lm/audio, the corresponding weight names are different,
    and it is necessary to distinguish between the first and last layers and the middle layer
    all_layer: The weights included in all layers
    """
    firsts: List[str]
    lasts: List[str]
    middle: str
    all_layer: List[str] = None


text_schema = PPStageSchema(
    firsts=['text_decoder.embedding.'],
    lasts=['text_decoder.decoder.final_layernorm.', 'text_decoder.output_layer.'],
    middle='text_decoder.decoder.layers.'
)
vision_schema = PPStageSchema(
    firsts=['image_encoder.encoder.patch_embed.'],
    lasts=['image_encoder.projector.'],
    middle='image_encoder.encoder.blocks.layers.'
)
audio_schema = PPStageSchema(
    firsts=['audio_encoder.encoder.conv', 'audio_encoder.encoder.audio_bos_eos_token'],
    lasts=['audio_encoder.encoder.proj', 'audio_encoder.encoder.ln_post'],
    middle='audio_encoder.encoder.blocks.layers.'
)


@dataclass
class PPRange:
    """For each rank of the pp group, we need know which layers of transformers correspond to it
    start. Each value in start defines the layer index at which the rank pp starts
    end. Each value in 'end' defines the layer index at which the rank pp ends
    Pp_first_rank. Defines the global pp_rank corresponding to the first layer of the transformer
    Pp_1ast_rank. Defines the global pp_rank corresponding to the last layer of the transformer
    """
    start: List[int]
    end: List[int]
    first_layer_rank: int
    last_layer_rank: int

    @property
    def pp_size(self) -> int:
        return len(self.start)


def partition_state_dict_by_pp(state_dict: STATE_DICT_T,
                               pp_ranges: List[PPRange],
                               stages: List[PPStageSchema]) -> List[STATE_DICT_T]:
    """For transformer structures of different modalities, use a universal PP splitting logic to split the
    model parameter state-dict into different PP ranks and reset the corresponding layer numbers. Supports
    hetero PP sizes with dp replication(e.g., VIT PP=1, Audio model PP=2, LLM PP =4)
    """
    global_pp_size = max(r.pp_size for r in pp_ranges)
    pp_weights = []
    for pp_rank in range(global_pp_size):
        pp_weight = {}
        for weight_name, weight_value in state_dict.items():
            for modality_stage, modality_pp_range in zip(stages, pp_ranges):
                offset = pp_rank - modality_pp_range.first_layer_rank
                if offset < 0:
                    continue
                local_pp_idx = offset % modality_pp_range.pp_size
                is_first_in_group = (local_pp_idx == modality_pp_range.first_layer_rank)
                is_last_in_group = (local_pp_idx == modality_pp_range.last_layer_rank)
                # 该模态首卡对应的权重
                if is_first_in_group:
                    for name_start in modality_stage.firsts:
                        if weight_name.startswith(name_start):
                            pp_weight[weight_name] = weight_value
                # 该模态尾卡对应的权重
                if is_last_in_group:
                    for name_start in modality_stage.lasts:
                        if weight_name.startswith(name_start):
                            pp_weight[weight_name] = weight_value
                # 该模态pp中间的卡对应的权重
                if weight_name.startswith(modality_stage.middle):
                    layer_start = modality_pp_range.start[local_pp_idx]
                    layer_end = modality_pp_range.end[local_pp_idx]
                    raw_layer_num, *remains = weight_name.replace(modality_stage.middle, "").split(".")
                    try:
                        raw_layer_num = int(raw_layer_num)
                        if layer_start <= raw_layer_num < layer_end:
                            new_layer_num = raw_layer_num - layer_start
                            new_weight_name = ".".join([modality_stage.middle[:-1], str(new_layer_num), *remains])
                            pp_weight[new_weight_name] = weight_value
                    except ValueError as e:
                        raise ValueError(
                            f"Failed to parse layer number from weight name: '{weight_name}'\n"
                            f"Modality: {modality_stage}, PP range: {modality_pp_range}\n"
                            f"Original error: {str(e)}"
                        ) from e
                # 该模态所有卡都包含的权重
                if modality_stage.all_layer:
                    for name_start in modality_stage.all_layer:
                        if weight_name.startswith(name_start):
                            pp_weight[weight_name] = weight_value

        pp_weights.append(pp_weight)
    return pp_weights


def save_by_vpp(state_dicts: List[Dict[str, torch.Tensor]],
                save_root_dir: Path,
                iteration: Optional[Union[str, int]] = 'release',
                pp_and_vpp_size: Tuple[int, int] = (1, 1),
                ep_size: int = 1,
                tp_rank: int = 0,
                ep_rank: int = 0):
    """获取pp_size和vpp_size"""
    pp_size, vpp_size = pp_and_vpp_size
    for pp_rank in tqdm(range(pp_size), desc="pp step"):
        # megatron格式权重目录的命名方式为 "mp_rank_{tp_rank}_{pp_rank}_{ep_rank}"
        name_parts = ["mp", "rank", f"{tp_rank:02d}"]
        if pp_size > 1:
            name_parts.append(f"{pp_rank:03d}")
        if ep_size > 1:
            name_parts.append(f"{ep_rank:03d}")
        iter_name = iteration if isinstance(iteration, str) else f"iter_{iteration:07d}"
        save_path = save_root_dir.joinpath(iter_name, "_".join(name_parts))
        save_path.mkdir(exist_ok=True, parents=True)
        if vpp_size > 1:
            # Collect VP state dicts for this PP rank
            save_dict = {f'model{vpp_idx}': state_dicts[vpp_idx * pp_size + pp_rank] for vpp_idx in range(vpp_size)}
        else:
            save_dict = {'model': state_dicts[pp_rank]}
        # 用于规避megatron对checkpoint_version的校验
        save_dict[CHECKPOINT_VERSION_KEY] = CHECKPOINT_VERSION_VALUE
        torch.save(save_dict, save_path.joinpath(MEGATRON_CKPT_NAME))
    save_root_dir.joinpath(LATEST_TXT).write_text(str(iteration))


def save_by_dcp(state_dict: STATE_DICT_T,
                save_root_dir: Path,
                iteration: Union[str, int] = 'release'):

    iter_name = iteration if isinstance(iteration, str) else f"iter_{iteration:07d}"
    save_path = save_root_dir.joinpath(iter_name)
    save_path.mkdir(exist_ok=True, parents=True)
    save_dict = {
        'model': state_dict,
        'checkpoint_version': CHECKPOINT_VERSION_VALUE
    }
    print(
        f"[dcp-save] writing {len(state_dict)} tensor(s) to {save_path} "
        "(the progress bar advances when the DCP writer finishes)",
        flush=True,
    )
    # DCP's writer is a single atomic operation from this process, so expose
    # its completion as a one-item progress bar instead of appearing idle.
    write_progress = tqdm(total=1, desc="Write DCP", unit="checkpoint")
    _save_state_dict(
        save_dict, storage_writer=FileSystemWriter(save_path), no_dist=True
    )
    write_progress.update(1)
    write_progress.close()
    save_root_dir.joinpath(LATEST_TXT).write_text(str(iteration))
    print(f"[dcp-save] complete: {save_root_dir}", flush=True)


def split_by_tp(state_dict: STATE_DICT_T, patterns: TP_PATTERN_T, tp_size: int = 0, vit_tp_size: int = 0,
                         audio_tp_size: int = 0) -> List[STATE_DICT_T]:
    """
    将状态字典按 TP 并行度切分
    :param state_dict: 原始状态字典
    :param patterns: 匹配模式到切分类的映射
    :param tp_size: 默认 TP 并行度
    :param vit_tp_size: vit TP 并行度
    :param audio_tp_size: audio TP 并行度
    :return: 切分后的状态字典列表
    """
    if tp_size == 1 and vit_tp_size <= 1 and audio_tp_size <= 1:
        return [state_dict.copy()]
    max_tp_size = max(tp_size, vit_tp_size, audio_tp_size)
    if any(
        size != 0 and max_tp_size % size != 0
        for size in [tp_size, vit_tp_size, audio_tp_size]
    ):
        raise ValueError('TP segmentation of multiple modules does not meet the requirements')

    # 初始化 TP 状态字典列表
    tp_dicts = [dict() for _ in range(max_tp_size)]

    def assign_split_values(tar_splitter, tar_tp_dicts, tar_tp_size, tar_key, tar_value):
        # 一次性获取所有 TP 的切分结果
        tar_split_values = tar_splitter.split(tar_tp_size, tar_value)
        # 遍历tar_tp_dicts中的每个字典
        for i, tar_tp_dict in enumerate(tar_tp_dicts):
            # 通过取模运算循环获取split_values中的值
            tar_tp_dict[tar_key] = tar_split_values[i % len(tar_split_values)].clone()

    # 遍历原始状态字典的每个键值对
    for key, value in state_dict.items():
        # 检查是否匹配任何模式
        for pattern, splitter in patterns.items():
            if re.match(pattern, key):
                if vit_tp_size != 0 and key.startswith(IMAGE_ENCODER):
                    assign_split_values(splitter, tp_dicts, vit_tp_size, key, value)
                    break
                if audio_tp_size != 0 and key.startswith(AUDIO_ENCODER):
                    assign_split_values(splitter, tp_dicts, audio_tp_size, key, value)
                    break
                if vit_tp_size == 0 or audio_tp_size == 0 or key.startswith(TEXT_DECODER):
                    assign_split_values(splitter, tp_dicts, tp_size, key, value)
                    break
        else:
            # 未匹配任何模式的值直接复制到所有 TP
            for tp_dict in tp_dicts:
                tp_dict[key] = value.clone()  # 避免共享内存
    return tp_dicts


def split_by_ep(_state_dict: STATE_DICT_T, _ep_size: int = 1, _num_experts: int = 0) -> List[Dict[str, torch.Tensor]]:
    if _ep_size == 1 or _num_experts == 0:
        return [_state_dict]

    per_ep_rank_experts = _num_experts // _ep_size
    ep_state_dicts = []
    for ep_rank in range(_ep_size):
        tmp_state_dict = {}
        for key, value in _state_dict.items():
            if "local_experts" in key:
                expert_idx = int(key.split(".")[
                                     7])  # 此处"7"表示expert_idx位于key的第（7+1）位, eg: key = "text_decoder.decoder.layers.1.mlp.experts.local_experts.*.linear_fc1.weight"
                if expert_idx >= ep_rank * per_ep_rank_experts and expert_idx < (ep_rank + 1) * per_ep_rank_experts:
                    local_expert_idx = expert_idx - ep_rank * per_ep_rank_experts
                    tmp_key_list = key.split(".")
                    tmp_key_list[7] = str(local_expert_idx)
                    new_key = ".".join(tmp_key_list)
                    tmp_state_dict[new_key] = value
            else:
                tmp_state_dict[key] = value

        ep_state_dicts.append(tmp_state_dict)

    return ep_state_dicts


def merge_llm_weights_to_state_dict(vl_state_dict: STATE_DICT_T, llm_state_dict: STATE_DICT_T) -> STATE_DICT_T:
    # 过滤掉vl_state_dict中llm相关的键
    for key in list(vl_state_dict.keys()):
        if key.startswith('model') or key.startswith("visual.merger"):
            vl_state_dict.pop(key)

    # 合并llm_state_dict到vl_state_dict
    vl_state_dict.update(llm_state_dict)

    return vl_state_dict


def filter_vit_keys(_state_dict: STATE_DICT_T):
    """过滤掉llm相关的键，只保留vit部分的键"""
    for key in list(_state_dict.keys()):
        if not key.startswith("visual"):
            _state_dict.pop(key)


def load_from_hf(hf_dir: Path, pt_path: Optional[Path] = None) -> STATE_DICT_T:
    # 注意AutoModel.from_pretrained转换成模型对象时，存在torch_dtype问题需确认，因此这里直接读取safetensors确保dtype一致
    state_dict = {}
    if pt_path:
        weight = torch.load(pt_path)
        state_dict.update(weight, device='cpu')
    else:
        files = list(hf_dir.glob("*.safetensors"))
        for safe_path in files:
            state_dict.update(load_file(str(safe_path), device='cpu'))
    return state_dict


def merge_pp_index(vit_pipeline_num_layers: List[int], llm_pipeline_num_layers: List[int]) -> List[Tuple[int, int]]:
    """返回每张卡上vit和llm各自的层数"""
    split_method = []
    for vit_num, llm_num in zip(vit_pipeline_num_layers, llm_pipeline_num_layers):
        split_method.append((vit_num, llm_num))
    return split_method


def merge_vpp_index(vit_pipeline_num_layers: VPP_LAYER_NUM_T,
                    llm_pipeline_num_layers: VPP_LAYER_NUM_T,
                    audio_pipeline_num_layers: VPP_LAYER_NUM_T) -> List[PPRange]:
    modalities_pp_range = []
    for modality in [vit_pipeline_num_layers, llm_pipeline_num_layers, audio_pipeline_num_layers]:
        modality_pp_flat = [item
                            for sublist in modality
                            for item in sublist]
        if not modality_pp_flat:
            continue
        modality_pp_acc = list(accumulate(modality_pp_flat))
        first_layer_rank, last_layer_rank = np.nonzero(np.array(modality_pp_flat))[0][[0, -1]]
        modalities_pp_range.append(PPRange(start=[0] + modality_pp_acc[:-1],
                                           end=modality_pp_acc,
                                           first_layer_rank=first_layer_rank,
                                           last_layer_rank=last_layer_rank))
    return modalities_pp_range


def convert(state_dict: STATE_DICT_T, ops: List[Operator], is_tie: bool, is_pp: bool) -> STATE_DICT_T:
    if is_tie and is_pp:
        # pp1时，output_layer从word_embedding处获取共享权重。pp>1时，流水线后面的卡无法获得word_embedding，因此需要加上该权重
        ops.append(TieOp(raw_name='text_decoder.embedding.word_embeddings.weight',
                         new_name='text_decoder.output_layer.weight'))
    for op in ops:
        op.apply(state_dict)
    return state_dict


def convert_hf_to_mm(convert_config: ConvertVppMMConfig, ops: List[Operator], tp_patterns: Dict[str, Callable],
                     stages: List[PPStageSchema]):
    pt_path = getattr(convert_config, 'pt_path', None)
    parallel_config = convert_config.parallel_config
    num_experts = convert_config.common_model_config.num_experts
    # 加载权重字典
    state_dict = load_from_hf(convert_config.hf_config.hf_dir, pt_path)

    # 如果有llm_config，则加载llm权重并合并到state_dict中
    if convert_config.common_model_config.llm_hf_dir is not None:
        llm_state_dict = load_from_hf(convert_config.common_model_config.llm_hf_dir)
        state_dict = merge_llm_weights_to_state_dict(state_dict, llm_state_dict)

    if convert_config.save_vit_only:
        # 如果只保存vit权重，则过滤掉非vit的权重
        filter_vit_keys(state_dict)

    # 权重转换、合并
    state_dict = convert(state_dict, ops, convert_config.common_model_config.tie_word_embeddings, parallel_config.is_pp())

    if getattr(convert_config, 'save_lora_only', False):
        state_dict = {k: v for k, v in state_dict.items() if "lora" in k}

    # 权重字典按ep域切分
    ep_state_dicts = split_by_ep(state_dict, parallel_config.ep_size, _num_experts=num_experts)

    # 权重字典按tp域切分
    ep_tp_state_dicts = []
    for ep_state_dict in ep_state_dicts:
        # 每个ep域对应的tp域拆分
        tp_state_dicts = split_by_tp(ep_state_dict, tp_patterns, parallel_config.tp_size, parallel_config.vit_tp_size, parallel_config.audio_tp_size)
        ep_tp_state_dicts.append(tp_state_dicts)

    # pp索引生成
    pp_ranges = merge_vpp_index(parallel_config.vit_pp_layers,
                                parallel_config.llm_pp_layers,
                                parallel_config.audio_pp_layers or [[]])
    for ep_rank, tp_state_dicts in enumerate(tqdm(ep_tp_state_dicts, desc="ep step")):
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = partition_state_dict_by_pp(tp_state_dict, pp_ranges, stages)
            save_by_vpp(pp_state_dicts, convert_config.mm_dir,
                        pp_and_vpp_size=(parallel_config.pp_size, parallel_config.vpp_size),
                        ep_size=parallel_config.ep_size, ep_rank=ep_rank, tp_rank=tp_rank)


def convert_hf_to_mm_dcp(convert_config: ConvertTorchDCPConfig, ops: List[Operator]):
    # load state dict from huggingface
    state_dict = load_from_hf(convert_config.hf_config.hf_dir)

    # load llm weight and merge to state_dict
    if convert_config.common_model_config.llm_hf_dir is not None:
        llm_state_dict = load_from_hf(convert_config.common_model_config.llm_hf_dir)
        state_dict = merge_llm_weights_to_state_dict(state_dict, llm_state_dict)

    # format and merge ckpt
    state_dict = convert(state_dict, ops, convert_config.common_model_config.tie_word_embeddings, is_pp=False)

    save_by_dcp(state_dict, convert_config.mm_dir)
