import json
import os
import re
import shutil
from pathlib import Path
from typing import List

import torch
from safetensors.torch import save_file

from checkpoint.common.constant import LATEST_TXT, MEGATRON_CKPT_NAME, IMAGE_ENCODER, AUDIO_ENCODER, TEXT_DECODER, \
    LORA_CKPT_NAME
from checkpoint.common.mm_types import STATE_DICT_T, PP_LAYER_NUM_T
from checkpoint.vlm_model.config import ConvertHFConfig
from checkpoint.vlm_model.hf_to_mm import load_from_hf
from checkpoint.vlm_model.operator import Operator, TP_PATTERN_T


def save_by_index_json(_state_dicts, _save_dir):
    metadata = {
        'format': 'pt'
    }
    for index, state_dict in enumerate(_state_dicts, start=1):
        name = f'model-{index:05}-of-{len(_state_dicts):05}.safetensors'
        save_file(state_dict, Path(_save_dir).joinpath(name), metadata=metadata)


def save_safetensors(_state_dicts, _save_dir):
    Path(_save_dir).mkdir(parents=True, exist_ok=True)

    metadata = {
        'format': 'pt'
    }
    save_file(_state_dicts, Path(_save_dir).joinpath(LORA_CKPT_NAME), metadata=metadata)


def split_by_index_json(state_dict: STATE_DICT_T, hf_dir: Path) -> List[STATE_DICT_T]:
    index_json_path = hf_dir.joinpath('model.safetensors.index.json')
    if not os.path.exists(index_json_path):
        raise ValueError(f"safetensors.index.json not in {index_json_path}")
    return_dicts = []
    weight_map = json.loads(index_json_path.read_text()).get('weight_map', {})
    for key, value in weight_map.items():
        index = int(value.split('-')[1])
        while index > len(return_dicts):
            return_dicts.append({})
        return_dicts[index - 1][key] = state_dict[key]
    return return_dicts


def copy_files_except_suffix(source_path: Path, target_path: Path, except_suffix: str = '.safetensors'):
    """拷贝源路径下除了以except_suffix为后缀的其他所有文件到目标路径，包含子目录"""
    target_path.mkdir(parents=True, exist_ok=True)
    for item in source_path.rglob('*'):
        if item.is_file() and item.suffix != except_suffix:
            relative_path = item.relative_to(source_path)
            destination = target_path / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)
            print(f"Copied: {item} -> {destination}")


def load_from_mm(load_dir: Path,
                 vit_pp_list: PP_LAYER_NUM_T,
                 llm_pp_list: PP_LAYER_NUM_T,
                 tp_size: int = 1,
                 audio_pp_list: PP_LAYER_NUM_T = None,
                 ep_size: int = 1,
                 num_experts: int = 1) -> List[STATE_DICT_T]:
    import mindspeed.megatron_adaptor  # noqa
    save_iteration = load_dir.joinpath(LATEST_TXT).read_text()
    save_dir = load_dir.joinpath(f"iter_{int(save_iteration):07}" if save_iteration != "release" else save_iteration)

    global_pp_size = max(
        len(vit_pp_list),
        len(llm_pp_list),
        len(audio_pp_list) if audio_pp_list else 0
    )

    state_dicts = []
    for tp_rank in range(tp_size):
        pp_state_dict = {}
        for pp_rank in range(global_pp_size):
            if ep_size > 1:
                for ep_rank in range(ep_size):
                    if global_pp_size > 1:
                        current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}_{int(ep_rank):03}")
                    else:
                        current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(ep_rank):03}")
                    pt_path = current_path.joinpath(MEGATRON_CKPT_NAME)
                    dict_ep = {}
                    for param, tensor in torch.load(pt_path, map_location='cpu', weights_only=False)['model'].items():
                        if tensor is not None:
                            new_key = rename_pp_ep_parameter(param, vit_pp_list, llm_pp_list, audio_pp_list, pp_rank, ep_rank, ep_size, num_experts)
                            dict_ep.update({new_key: tensor})
                    pp_state_dict.update(dict_ep)
            else:
                if global_pp_size > 1:
                    current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}_{int(pp_rank):03}")
                else:
                    current_path = save_dir.joinpath(f"mp_rank_{int(tp_rank):02}")
                pt_path = current_path.joinpath(MEGATRON_CKPT_NAME)
                print(str(pt_path).center(100, '_'))
                # 注意output_layer存在_extra_state其值为None
                pp_state_dict.update(
                    {rename_pp_parameter(param, vit_pp_list, llm_pp_list, audio_pp_list, pp_rank): tensor
                    for param, tensor in torch.load(pt_path, map_location='cpu', weights_only=False)['model'].items()
                    if tensor is not None})
        state_dicts.append(pp_state_dict)
    return state_dicts


def merge_by_tp(tp_state_dicts: List[STATE_DICT_T], patterns: TP_PATTERN_T, tp_size: int = 0, vit_tp_size: int = 0,
                         audio_tp_size: int = 0) -> STATE_DICT_T:
    """将多个TP分片的权重合并回完整权重"""
    if not tp_state_dicts:
        return {}
    merged_dict = {}
    max_tp_size = len(tp_state_dicts)
    if max_tp_size == 1:
        return tp_state_dicts[0]
    # 定义前缀和对应tp_size的映射
    tp_config = {
        IMAGE_ENCODER: vit_tp_size,
        AUDIO_ENCODER: audio_tp_size,
        TEXT_DECODER: tp_size
    }
    for key in tp_state_dicts[0].keys():
        # 收集所有分片的对应权重
        tp_values = [sd[key] for sd in tp_state_dicts]

        # 查找匹配的拆分函数，并获取其反向合并方法
        for pattern, merger in patterns.items():
            if re.match(pattern, key):
                for prefix, size in tp_config.items():
                    if key.startswith(prefix):
                        if size <= 0:
                            merged_dict[key] = merger.merge(tp_values)
                        elif size == 1:
                            merged_dict[key] = tp_values[0]
                        else:
                            merged_dict[key] = merger.merge(tp_values[:size])
                        break
                break
        else:
            merged_dict[key] = tp_values[0]
    return merged_dict


def rename_pp_ep_parameter(param_name: str,
                        vit_pp_list: List[int],
                        llm_pp_list: List[int],
                        audio_pp_list: List[int] = None,
                        pp_index: int = 0,
                        ep_rank: int = 0,
                        ep_size: int = 1,
                        num_experts: int = 16) -> str:
    pp_key = rename_pp_parameter(param_name, vit_pp_list, llm_pp_list, audio_pp_list, pp_index)
    per_ep_rank_experts = num_experts // ep_size
    offset = ep_rank * per_ep_rank_experts  # 原始专家索引的起始位置
    if "local_experts" in pp_key:
        # 解析原始 key 的结构
        parts = pp_key.split(".")
        if len(parts) < 8:
            raise ValueError(f"Invalid key format: {pp_key}")
        # 获取 local_expert_idx（即内部编号）
        local_expert_idx = int(parts[7])
        # 恢复为原始专家索引
        original_expert_idx = offset + local_expert_idx
        # 替换 parts[7] 为原始索引
        parts[7] = str(original_expert_idx)
        # 重构 key
        new_key = ".".join(parts)
    else:
        new_key = pp_key
    return new_key


def rename_pp_parameter(param_name: str,
                        vit_pp_list: List[int],
                        llm_pp_list: List[int],
                        audio_pp_list: List[int] = None,
                        pp_index: int = 0) -> str:
    # 计算偏移量：当前分片前的总层数
    def compute_offset(pp_list: List[int], idx: int) -> int:
        if not pp_list:
            return 0
        effective_idx = idx % len(pp_list)
        return sum(pp_list[:effective_idx]) if effective_idx > 0 else 0

    # 计算各模态的偏移量
    vit_offset = compute_offset(vit_pp_list, pp_index)
    llm_offset = compute_offset(llm_pp_list, pp_index)
    audio_offset = compute_offset(audio_pp_list, pp_index) if audio_pp_list is not None else 0

    # 定义模式列表：正则表达式和对应的偏移量
    patterns = [
        (r'^image_encoder\.encoder\.blocks\.layers\.(\d+)', vit_offset),
        # Internvl系列模型的VitTransformerBlock被命名为encoder
        (r'^image_encoder\.encoder\.encoder\.layers\.(\d+)', vit_offset),
        (r'^text_decoder\.decoder\.layers\.(\d+)', llm_offset),
        (r'^audio_encoder\.encoder\.blocks\.layers\.(\d+)', audio_offset)
    ]

    # 统一处理所有参数
    for pattern, offset in patterns:
        match = re.match(pattern, param_name)
        if match:
            # 提取原始层号
            layer_num = int(match.group(1))
            # 计算新层号
            new_layer_num = offset + layer_num
            # 替换层号
            return re.sub(r'\.\d+', f'.{new_layer_num}', param_name, count=1)

    # 不匹配任何模式则返回原参数名
    return param_name


def convert_mm_to_hf(convert_config: ConvertHFConfig,
                     ops: List[Operator],
                     tp_patterns: TP_PATTERN_T,
                     merge_source: bool = False):
    parallel_config = convert_config.parallel_config
    config = convert_config.hf_config.config
    # 找到最大的tp
    max_tp_size = max(parallel_config.tp_size, parallel_config.vit_tp_size, parallel_config.audio_tp_size)
    ep_size = parallel_config.ep_size if hasattr(parallel_config, 'ep_size') else 1
    if not hasattr(config, 'text_config'):
        num_experts = 1
    else:
        num_experts = config.text_config.num_experts if hasattr(config.text_config, 'num_experts') else 1
    # 加载权重字典
    state_dicts = load_from_mm(convert_config.mm_dir, parallel_config.vit_pp_layers, parallel_config.llm_pp_layers,
                               max_tp_size, parallel_config.audio_pp_layers, ep_size, num_experts)
    state_dict = merge_by_tp(state_dicts, tp_patterns, parallel_config.tp_size, parallel_config.vit_tp_size, parallel_config.audio_tp_size)
    for op in ops:
        op.revert(state_dict)  # 执行逆操作
    if merge_source:
        state_dict = {**load_from_hf(convert_config.hf_config.hf_dir), **state_dict}
    state_dicts = split_by_index_json(state_dict, convert_config.hf_config.hf_dir)
    copy_files_except_suffix(convert_config.hf_config.hf_dir, convert_config.save_hf_dir)
    save_by_index_json(state_dicts, convert_config.save_hf_dir)


def convert_lora_mm_to_hf(convert_config: ConvertHFConfig,
                          ops: List[Operator],
                          tp_patterns: TP_PATTERN_T):
    parallel_config = convert_config.parallel_config
    # 找到最大的tp
    max_tp_size = max(parallel_config.tp_size, parallel_config.vit_tp_size, parallel_config.audio_tp_size)
    # 加载权重字典
    state_dicts = load_from_mm(convert_config.mm_dir, parallel_config.vit_pp_layers, parallel_config.llm_pp_layers,
                               max_tp_size, parallel_config.audio_pp_layers)
    state_dict = merge_by_tp(state_dicts, tp_patterns)
    for op in ops:
        op.revert(state_dict)  # 执行逆操作
    save_safetensors(state_dict, convert_config.save_hf_dir)
