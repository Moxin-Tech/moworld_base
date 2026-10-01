#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from itertools import accumulate
from typing import List, Dict, Callable

import numpy as np
from tqdm import tqdm

from checkpoint.common.mm_types import STATE_DICT_T, VPP_LAYER_NUM_T
from checkpoint.vlm_model.config import ConvertVppMMConfig
from checkpoint.vlm_model.hf_to_mm import (
    PPRange, PPStageSchema,
    load_from_hf, merge_llm_weights_to_state_dict, filter_vit_keys,
    convert, split_by_ep, split_by_tp, save_by_vpp,
)
from checkpoint.vlm_model.operator import Operator


def partition_state_dict_by_pp_ldt(state_dict: STATE_DICT_T,
                                   pp_ranges: List[PPRange],
                                   stages: List[PPStageSchema]) -> List[STATE_DICT_T]:

    global_pp_size = max(r.pp_size for r in pp_ranges)
    pp_weights = []
    for pp_rank in range(global_pp_size):
        pp_weight = {}
        for weight_name, weight_value in state_dict.items():
            for modality_stage, modality_pp_range in zip(stages, pp_ranges):
                if pp_rank >= modality_pp_range.pp_size:
                    continue
                is_first_in_group = (pp_rank == modality_pp_range.first_layer_rank)
                is_last_in_group = (pp_rank == modality_pp_range.last_layer_rank)
                # 该模态首卡对应的权重（如 embedding）
                if is_first_in_group:
                    for name_start in modality_stage.firsts:
                        if weight_name.startswith(name_start):
                            pp_weight[weight_name] = weight_value
                # 该模态尾卡对应的权重（如 output_layer / final_layernorm）
                if is_last_in_group:
                    for name_start in modality_stage.lasts:
                        if weight_name.startswith(name_start):
                            pp_weight[weight_name] = weight_value
                # 该模态pp中间的卡对应的权重
                if weight_name.startswith(modality_stage.middle):
                    layer_start = modality_pp_range.start[pp_rank]
                    layer_end = modality_pp_range.end[pp_rank]
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
                    has_layers = modality_pp_range.start[pp_rank] < modality_pp_range.end[pp_rank]
                    if has_layers or is_first_in_group or is_last_in_group:
                        for name_start in modality_stage.all_layer:
                            if weight_name.startswith(name_start):
                                pp_weight[weight_name] = weight_value

        pp_weights.append(pp_weight)
    return pp_weights


def merge_vpp_index_ldt(vit_pipeline_num_layers: VPP_LAYER_NUM_T,
                        llm_pipeline_num_layers: VPP_LAYER_NUM_T,
                        audio_pipeline_num_layers: VPP_LAYER_NUM_T) -> List[PPRange]:

    modalities_pp_range = []
    modalities = [vit_pipeline_num_layers, llm_pipeline_num_layers, audio_pipeline_num_layers]
    is_llm_flags = [False, True, False]
    for modality, is_llm in zip(modalities, is_llm_flags):
        modality_pp_flat = [item
                            for sublist in modality
                            for item in sublist]
        if not modality_pp_flat:
            continue
        modality_pp_acc = list(accumulate(modality_pp_flat))
        if is_llm:
            # LLM的embedding层放在第一个有层数的VPP stage的PP0,
            # unembedding层固定在最后一个VPP stage的PP0(U形布局)
            # embedding跳过整个VPP stage全为0的情况
            pp_size = len(modality[0])
            vpp_size = len(modality)
            first_vpp_idx = next(i for i in range(vpp_size) if sum(modality[i]) > 0)
            first_layer_rank = first_vpp_idx * pp_size
            last_layer_rank = (vpp_size - 1) * pp_size
        else:
            first_layer_rank, last_layer_rank = np.nonzero(np.array(modality_pp_flat))[0][[0, -1]]
        modalities_pp_range.append(PPRange(start=[0] + modality_pp_acc[:-1],
                                           end=modality_pp_acc,
                                           first_layer_rank=first_layer_rank,
                                           last_layer_rank=last_layer_rank))
    return modalities_pp_range


def convert_hf_to_mm_ldt(convert_config: ConvertVppMMConfig, ops: List[Operator],
                          tp_patterns: Dict[str, Callable],
                          stages: List[PPStageSchema]):
    """LDT版convert_hf_to_mm,使用LDT变体的PP切分和VPP索引逻辑。"""
    pt_path = getattr(convert_config, 'pt_path', None)
    parallel_config = convert_config.parallel_config
    num_experts = convert_config.common_model_config.num_experts

    state_dict = load_from_hf(convert_config.hf_config.hf_dir, pt_path)

    if convert_config.common_model_config.llm_hf_dir is not None:
        llm_state_dict = load_from_hf(convert_config.common_model_config.llm_hf_dir)
        state_dict = merge_llm_weights_to_state_dict(state_dict, llm_state_dict)

    if convert_config.save_vit_only:
        filter_vit_keys(state_dict)

    state_dict = convert(state_dict, ops, convert_config.common_model_config.tie_word_embeddings, parallel_config.is_pp())

    if getattr(convert_config, 'save_lora_only', False):
        state_dict = {k: v for k, v in state_dict.items() if "lora" in k}

    ep_state_dicts = split_by_ep(state_dict, parallel_config.ep_size, _num_experts=num_experts)

    ep_tp_state_dicts = []
    for ep_state_dict in ep_state_dicts:
        tp_state_dicts = split_by_tp(ep_state_dict, tp_patterns, parallel_config.tp_size,
                                     parallel_config.vit_tp_size, parallel_config.audio_tp_size)
        ep_tp_state_dicts.append(tp_state_dicts)

    # 使用LDT变体VPP索引
    pp_ranges = merge_vpp_index_ldt(parallel_config.vit_pp_layers,
                                    parallel_config.llm_pp_layers,
                                    parallel_config.audio_pp_layers or [[]])
    for ep_rank, tp_state_dicts in enumerate(tqdm(ep_tp_state_dicts, desc="ep step")):
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            # 使用LDT变体PP切分
            pp_state_dicts = partition_state_dict_by_pp_ldt(tp_state_dict, pp_ranges, stages)
            save_by_vpp(pp_state_dicts, convert_config.mm_dir,
                        pp_and_vpp_size=(parallel_config.pp_size, parallel_config.vpp_size),
                        ep_size=parallel_config.ep_size, ep_rank=ep_rank, tp_rank=tp_rank)
