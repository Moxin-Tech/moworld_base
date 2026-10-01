from collections import OrderedDict
from pathlib import Path
from typing import Optional
import os
import copy

import torch
from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig, ParallelConfig
from checkpoint.sora_model.convert_utils.save_load_utils import (
    load_from_mm,
    load_from_hf,
    load_from_layerzero,
    save_as_mm,
    save_as_hf,
)
from checkpoint.sora_model.convert_utils.utils import (
    flip_mapping,
    replace_name,
    check_method_support,
    check_parallel_config_support
)
from checkpoint.sora_model.convert_utils.tp_patterns import TP_PARTTERN_MAPPING
from checkpoint.vlm_model.hf_to_mm import save_by_dcp


class DefaultLayerIndexConverter:
    @staticmethod
    def get_layer_index(name):
        parts = name.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return int(parts[1])
        return None

    @staticmethod
    def convert_layer_index(name, new_layer_index):
        parts = name.split(".")
        if len(parts) > 1:
            if parts[1].isdigit():
                parts[1] = str(new_layer_index)
                name = ".".join(parts)
        return name


class SoraModelConverter(Converter):
    """General converter for SoraModel"""
    _supported_methods = []
    _enable_tp = False
    _enable_pp = False
    _enable_vpp = False
    convert_mapping = OrderedDict() # origin to mm, mapping mode (recommend)
    str_replace_mapping = OrderedDict() # origin to mm, str_replace mode
    hf_to_mm_convert_mapping = OrderedDict() # hf to mm, mapping mode (recommend)
    hf_to_mm_str_replace_mapping = OrderedDict() # hf to mm, str_replace mode
    lora_hf_to_mm_str_replace_mapping = OrderedDict() # lora hf to mm, str_replace mode
    lora_target_modules = [] # lora modules

    # key: TP pattern name, values: state_dict.key
    tp_split_mapping = {
        "column_parallel_tp": [],
        "row_parallel_tp": [],
        "qkv_fused_column_tp": []
    }

    # Special TP pattern, key: custom TP pattern class, values: state_dict.key
    spec_tp_split_mapping = {}

    pre_process_weight_names = [] # pre_process layers for pp
    post_preprocess_weight_names = [] # post_process layers for pp
    layer_index_converter = DefaultLayerIndexConverter() # class to convert layer index, functions: get_layer_index, convert_layer_index

    @check_method_support
    def hf_to_mm(self, cfg: ConvertConfig):
        print(f"[converter] HF -> MM: loading {cfg.source_path}", flush=True)
        state_dict = load_from_hf(cfg.source_path)
        print(f"[converter] HF -> MM: renaming {len(state_dict)} tensor(s)", flush=True)
        state_dict = self._replace_state_dict(
            state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping
        )
        print("[converter] HF -> MM: splitting and writing shards", flush=True)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)
        print(f"[converter] HF -> MM: complete ({cfg.target_path})", flush=True)

    @check_method_support
    def mm_to_hf(self, cfg: ConvertConfig):
        import mindspeed.megatron_adaptor  # noqa
        state_dicts = load_from_mm(cfg.source_path)
        state_dict = self._mm_merge(state_dicts)
        state_dict = self._replace_state_dict(
            state_dict,
            flip_mapping(self.hf_to_mm_convert_mapping),
            flip_mapping(self.hf_to_mm_str_replace_mapping)
        )
        save_as_hf(state_dict, cfg.hf_dir, cfg.target_path)

    @check_method_support
    def mm_to_dcp(self, cfg: ConvertConfig):
        print(f"[converter] MM -> DCP: loading {cfg.source_path}", flush=True)
        state_dicts = load_from_mm(cfg.source_path)
        print("[converter] MM -> DCP: merging shards", flush=True)
        state_dict = self._mm_merge(state_dicts)
        print(f"[converter] MM -> DCP: writing {len(state_dict)} tensor(s) to {cfg.target_path}", flush=True)
        save_by_dcp(state_dict, Path(cfg.target_path))
        set_directory_permissions(Path(cfg.target_path))
        print(f"[converter] MM -> DCP: complete ({cfg.target_path})", flush=True)

    @check_method_support
    def lora_hf_to_mm(self, cfg: ConvertConfig):
        print(f"[converter] safetensors -> MM: loading {cfg.source_path}", flush=True)
        state_dict = load_from_hf(cfg.source_path)
        print(f"[converter] safetensors -> MM: renaming {len(state_dict)} tensor(s)", flush=True)
        state_dict = self._replace_state_dict(
            state_dict,
            str_replace_mapping=self.lora_hf_to_mm_str_replace_mapping
        )
        print("[converter] safetensors -> MM: splitting and writing shards", flush=True)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)
        print(f"[converter] safetensors -> MM: complete ({cfg.target_path})", flush=True)

    @check_method_support
    def resplit(self, cfg: ConvertConfig):
        state_dicts = load_from_mm(cfg.source_path)
        state_dict = self._mm_merge(state_dicts)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    @check_method_support
    def layerzero_to_mm(
        self,
        cfg: ConvertConfig,
        iteration: Optional[int] = None,
        prefix: str = "predictor",
        ema_model: bool = False
    ):
        state_dict = load_from_layerzero(
            cfg.source_path,
            iteration=iteration,
            prefix=prefix,
            ema_model=ema_model,
            for_release=True
        )
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    @check_method_support
    def merge_lora_to_base(
        self,
        cfg: ConvertConfig,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        use_npu: bool = True
    ):
        source_state_dicts = load_from_mm(cfg.source_path)
        source_state_dict = self._mm_merge(source_state_dicts)
        lora_state_dicts = load_from_mm(cfg.lora_path)
        lora_state_dict = self._mm_merge(lora_state_dicts)

        os.environ['JSONARGPARSE_DEPRECATION_WARNINGS'] = 'off'
        from checkpoint.common.merge_base_lora_weight import lora_merge_to_base

        if lora_rank == 0:
            raise ValueError(f"LoRA rank can not be 0")

        if use_npu:
            source_state_dict = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in source_state_dict.items()}
            lora_state_dict = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in lora_state_dict.items()}

        lora_merged_state_dict = lora_merge_to_base(
            source_state_dict,
            lora_state_dict,
            self.lora_target_modules,
            scaling=float(lora_alpha) / float(lora_rank)
        )

        if use_npu:
            torch.npu.empty_cache()
            lora_merged_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in lora_merged_state_dict.items()}

        state_dicts = self._mm_split(lora_merged_state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    @check_parallel_config_support
    def _mm_split(
        self,
        state_dict: dict,
        cfg: ParallelConfig, # target parallel config
    ):
        state_dicts = self._tp_split(state_dict, cfg)
        state_dicts = self._pp_vpp_split(state_dicts, cfg)
        return state_dicts

    def _mm_merge(
        self,
        state_dicts: list
    ):
        state_dicts = self._pp_vpp_merge(state_dicts)
        state_dict = self._tp_merge(state_dicts)
        return state_dict

    def _replace_state_dict(
        self,
        state_dict: dict,
        convert_mapping: dict = None,
        str_replace_mapping: dict = None
    ):
        if convert_mapping:
            for old_key, new_key in convert_mapping.items():
                if old_key not in state_dict.keys():
                    continue
                state_dict[new_key] = state_dict.pop(old_key)

        names = list(state_dict.keys())
        for name in tqdm(names, desc="Rename tensors", unit="tensor"):
            if "_extra_state" in name:
                state_dict.pop(name)
                continue
            if str_replace_mapping:
                weight = state_dict.pop(name)
                name = replace_name(name, str_replace_mapping)
                state_dict[name] = weight

        return state_dict

    def _tp_split(
        self,
        state_dict: dict,
        cfg: ParallelConfig
    ):
        if cfg.tp_size <= 1:
            return [state_dict]

        tp_state_dicts = [copy.deepcopy(state_dict) for _ in range(cfg.tp_size)]

        def _split(layer_names, tp_pattern, tp_size):
            for name in layer_names:
                split_values = tp_pattern.split(state_dict[name], tp_size)
                for tp_rank in range(tp_size):
                    tp_state_dicts[tp_rank][name] = split_values[tp_rank].clone()

        for tp_pattern, layer_names in self.tp_split_mapping.items():
            tp_pattern_class = TP_PARTTERN_MAPPING.get(tp_pattern, None)
            if tp_pattern_class:
                _split(layer_names, tp_pattern_class, cfg.tp_size)
            else:
                raise NotImplementedError(f"TP pattern {tp_pattern} is not found in common tp_pattern, only support: {TP_PARTTERN_MAPPING.keys()}")

        for spec_tp_pattern, layer_names in self.spec_tp_split_mapping.items():
            _split(layer_names, spec_tp_pattern, cfg.tp_size)

        return tp_state_dicts

    def _tp_merge(
        self,
        state_dicts: list
    ):
        if len(state_dicts) == 1:
            return state_dicts[0]

        tp_size = len(state_dicts)

        merged_state_dict = copy.deepcopy(state_dicts[0])

        def _merge(layer_names, tp_pattern, tp_size):
            for name in layer_names:
                merge_value = tp_pattern.merge(
                    [state_dicts[tp_rank][name] for tp_rank in range(tp_size)]
                )
                merged_state_dict[name] = merge_value

        for tp_pattern, layer_names in self.tp_split_mapping.items():
            tp_pattern_class = TP_PARTTERN_MAPPING.get(tp_pattern, None)
            if tp_pattern_class:
                _merge(layer_names, tp_pattern_class, tp_size)
            else:
                raise NotImplementedError(f"TP pattern {tp_pattern} is not found in common tp_pattern, only support: {TP_PARTTERN_MAPPING.keys()}")

        for spec_tp_parttern, layer_names in self.spec_tp_split_mapping.items():
            _merge(layer_names, spec_tp_parttern, tp_size)

        return merged_state_dict

    def _pp_vpp_split(
        self,
        state_dicts: list,
        cfg: ParallelConfig
    ):
        if len(cfg.pp_layers) == 0:
            return [state_dicts]

        enable_vpp = isinstance(cfg.pp_layers[0], list)
        if not enable_vpp and len(cfg.pp_layers) <= 1:
            return [state_dicts]

        if enable_vpp:
            pp_sizes_flat = [
                layers
                for vpp_layer in cfg.pp_layers
                for layers in vpp_layer
            ]
        else:
            pp_sizes_flat = cfg.pp_layers

        vpp_tp_state_dicts = [[None for _ in range(len(state_dicts))] for _ in range(len(pp_sizes_flat))]
        for vpp_rank, _ in enumerate(pp_sizes_flat):
            is_first = vpp_rank == 0
            is_last = vpp_rank == len(pp_sizes_flat) - 1
            start_layer = sum(pp_sizes_flat[:vpp_rank])
            end_layer = sum(pp_sizes_flat[:vpp_rank]) + pp_sizes_flat[vpp_rank]

            for tp_rank, state_dict in enumerate(state_dicts):
                pp_tp_param = dict()
                for k in state_dict.keys():
                    if k in self.pre_process_weight_names and is_first:
                        pp_tp_param[k] = state_dict[k]
                    elif k in self.post_preprocess_weight_names and is_last:
                        pp_tp_param[k] = state_dict[k]

                    layer_idx = self.layer_index_converter.get_layer_index(k)
                    if layer_idx is not None and start_layer <= layer_idx < end_layer:
                        new_k = self.layer_index_converter.convert_layer_index(k, layer_idx - start_layer)
                        pp_tp_param[new_k] = state_dict[k]
                vpp_tp_state_dicts[vpp_rank][tp_rank] = pp_tp_param

        if enable_vpp:
            # rearrange state_dict list by pp_rank
            vpp_size = len(cfg.pp_layers)
            pp_size = len(cfg.pp_layers[0])
            tp_size = len(state_dicts)
            pp_tp_state_dicts = [[None for _ in range(tp_size)] for _ in range(pp_size)]

            for pp_rank in range(pp_size):
                for tp_rank in range(tp_size):
                    pp_tp_state_dicts[pp_rank][tp_rank] = [
                        vpp_tp_state_dicts[vpp_rank * pp_size + pp_rank][tp_rank]
                        for vpp_rank in range(vpp_size)
                    ]
        else:
            pp_tp_state_dicts = vpp_tp_state_dicts

        return pp_tp_state_dicts

    def _pp_vpp_merge(
        self,
        state_dicts: list
    ):
        pp_size = len(state_dicts)
        if pp_size == 0:
            return state_dicts[0]

        tp_size = len(state_dicts[0])
        if isinstance(state_dicts[0][0], list):
            enable_vpp = True
            vpp_size = len(state_dicts[0][0])
        else:
            enable_vpp = False
            vpp_size = 1

        if not enable_vpp and pp_size == 1:
            return state_dicts[0]

        tp_state_dicts = []

        def _process_state_dict(state_dict, tp_state_dict, layer_start, layer_index_converter):
            max_layer_index = layer_start
            for key, value in state_dict.items():
                layer_index = layer_index_converter.get_layer_index(key)
                if layer_index is not None:
                    key = layer_index_converter.convert_layer_index(key, layer_index + layer_start)
                    max_layer_index = max(max_layer_index, layer_index + layer_start)
                tp_state_dict[key] = value
            return tp_state_dict, max_layer_index

        for tp_rank in range(tp_size):
            layer_start = 0
            tp_state_dict = {}
            for vpp_rank in range(vpp_size):
                for pp_rank in range(pp_size):
                    if enable_vpp:
                        state_dict = state_dicts[pp_rank][tp_rank][vpp_rank]
                    else:
                        state_dict = state_dicts[pp_rank][tp_rank]
                    tp_state_dict, max_layer_index = _process_state_dict(state_dict, tp_state_dict, layer_start, self.layer_index_converter)
                    layer_start = max_layer_index + 1
                    tp_state_dicts.append(tp_state_dict)
        return tp_state_dicts
