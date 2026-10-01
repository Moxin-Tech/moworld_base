from typing import List, Dict
from pathlib import Path
import re
import json
import torch
from transformers import AutoConfig, AutoProcessor

from checkpoint.common.converter import Converter
from checkpoint.common.merge_dcp_to_hf import merge_dcp_to_hf_sharded
from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded
from checkpoint.common.dcp_utils import append_state_dict_to_dcp
from checkpoint.vlm_model.converters.moe_expert import merge_moe_expert_weights, ConfigType


class Qwen35Converter(Converter):
    """
    DCP converter implementation for qwen3.5 model supporting HF ↔ DCP format conversion for multiple model architectures

    Supports:
    - HF → DCP conversion
    - DCP → HF merging
    - Placeholder methods for megatron format and resharding operations.
    """

    # MoE experts params
    expert_weight_name_patterns = [
        r"model\.language_model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
        r"model\.language_model\.layers\.\d+\.mlp\.experts\.down_proj",
        r"mtp\.layers\.\d+\.mlp\.experts\.gate_up_proj",
        r"mtp\.layers\.\d+\.mlp\.experts\.down_proj"
    ]

    mtp_expert_key_suffixes = (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")

    @staticmethod
    def is_mtp_expert_key(key: str) -> bool:
        return key.startswith("mtp.layers.") and ".mlp.experts." in key and key.endswith(Qwen35Converter.mtp_expert_key_suffixes)

    def hf_to_dcp(
        self,
        hf_dir: str = "",
        dcp_dir: str = "",
        dcp_prefix: str = "",
        hf_prefix: str = "",
        tie_weight_mapping: Dict[str, str] = None,
        fused_linear_names: List[str] = None,
    ):
        """
        Converts a Hugging Face formatted model checkpoint to torch-dcp format.

        Args:
            hf_dir (str): Input: Path to HF-format model directory
            dcp_dir (str): Output: Path to save DCP-format model
            dcp_prefix (str): Prefix to add for DCP format parameter names
            hf_prefix (str): Prefix to remove from Hugging Face parameter names
            tie_weight_mapping (str): Weight tying mapping in comma-separated format.
                Pairs follow "target1,source1,target2,source2,..." pattern.
                Used when output head shares weights with input embeddings.
            fused_linear_names (str): Names of MoE (Mixture of Experts) expert parameters
                in comma-separated format. These parameters will be reshaped during conversion.

        Steps:
        1. Load the state dict from HF format.
        2. Optionally tie weights (e.g., share lm_head and embed_tokens weights).
        3. Rename all keys by adding DCP prefix and removing HF prefix.
        4. Save the converted checkpoint in DCP format.
        5. Set proper directory permissions.
        """
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        num_experts = getattr(config.text_config, "num_experts", 0)
        mtp_num_hidden_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0)
        mtp_expert_collector: Dict[str, torch.Tensor] = {}

        def state_dict_convert_func(state_dict):
            if tie_weight_mapping:
                for tgt_weight, src_weight in tie_weight_mapping.items():
                    if src_weight in state_dict.keys():
                        state_dict[tgt_weight] = state_dict[src_weight]

            ori_keys = list(state_dict.keys())
            for ori_key in ori_keys:
                value = state_dict.pop(ori_key)
                if mtp_num_hidden_layers > 0 and self.is_mtp_expert_key(ori_key):
                    mtp_expert_collector[ori_key] = value
                    continue
                # permute expert weight gate_up_proj from (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) to (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
                # permute expert weight down_proj from (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) to (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
                # to meet gemm compute
                for pattern in self.expert_weight_name_patterns:
                    if re.fullmatch(pattern, ori_key):
                        value = value.permute(0, 2, 1).contiguous()

                # qwen3.5 currently does not support fused_linear_names
                # view experts weight: (expert_num, input_dim, output_dim) -> (expert_num * input_dim, output_dim)
                if fused_linear_names:
                    if any(fused_linear_name in ori_key for fused_linear_name in fused_linear_names):
                        value = value.view(-1, value.shape[-1])

                new_key = ori_key.replace(hf_prefix, dcp_prefix, 1) if len(hf_prefix) > 0 else f"{dcp_prefix}{ori_key}"
                state_dict[new_key] = value
            return state_dict

        hf_to_dcp_sharded(
            hf_dir=hf_dir,
            dcp_dir=dcp_dir,
            state_dict_convert_func=state_dict_convert_func
        )

        if mtp_num_hidden_layers > 0 and num_experts > 0 and mtp_expert_collector:
            mtp_weight_path = "mtp.layers.{layer}.mlp.experts.{expert}"
            merge_moe_expert_weights(
                mtp_expert_collector, mtp_num_hidden_layers, num_experts,
                expert_start_layer=0, config_type=ConfigType.QWEN3_5, weight_path=mtp_weight_path
            )
            merged_mtp_state_dict = {}
            for mtp_key, mtp_value in mtp_expert_collector.items():
                # qwen3.5 currently does not support fused_linear_names
                # view experts weight: (expert_num, input_dim, output_dim) -> (expert_num * input_dim, output_dim)
                if fused_linear_names:
                    if any(fused_linear_name in mtp_key for fused_linear_name in fused_linear_names):
                        mtp_value = mtp_value.view(-1, mtp_value.shape[-1])

                new_mtp_key = mtp_key.replace(hf_prefix, dcp_prefix, 1) if len(hf_prefix) > 0 else f"{dcp_prefix}{mtp_key}"
                merged_mtp_state_dict[new_mtp_key] = mtp_value
            del mtp_expert_collector
            append_state_dict_to_dcp(Path(dcp_dir), merged_mtp_state_dict)

    def dcp_to_hf(
        self,
        dcp_dir: str = "",
        save_hf_dir: Path = "",
        origin_hf_dir: str = "",
        dcp_prefix: str = "",
        hf_prefix: str = "",
        fused_linear_names: List[str] = None,
        trust_remote_code: bool = True,
        to_bf16: bool = False,
        keep_origin_mtp_weights: bool = False,
    ):
        """
        Merges torch-dcp shards and converts them back into standard Hugging Face format.

        This is typically used after training or inference in torch-dcp format to export
        a model that can be easily loaded with Hugging Face Transformers.
        Args:
            dcp_dir (str): Input: Directory containing DCP shards
            save_hf_dir (Path): Output: Directory to save merged HF model
            origin_hf_dir (str): Reference: Original HF model dir (for config/tokenizer)
            dcp_prefix (str): Prefix to remove from DCP format parameter names
            hf_prefix (str): Prefix to add for Hugging Face parameter names
            fused_linear_names (str): Names of MoE (Mixture of Experts) expert parameters
                in comma-separated format. These parameters need special reshaping during conversion.
            keep_origin_mtp_weights (bool): If True and DCP does not contain MTP weights,
                preserve the original MTP weights from origin_hf_dir. Default is False.
        """
        config = AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=trust_remote_code)
        num_experts = getattr(config.text_config, "num_experts", 0)
        mtp_num_hidden_layers = getattr(config.text_config, "mtp_num_hidden_layers", 0)
        has_mtp_weights_in_dcp = False

        def state_dict_convert_func(state_dict):
            nonlocal has_mtp_weights_in_dcp
            state_dict_keys = list(state_dict.keys())

            for key in state_dict_keys:
                # qwen3.5 currently does not support fused_linear_names
                # view experts weight: (expert_num * input_dim, output_dim) -> (expert_num, input_dim, output_dim)
                if fused_linear_names:
                    if num_experts and any(fused_linear_name in key for fused_linear_name in fused_linear_names):
                        state_dict[key] = state_dict[key].view(num_experts, -1, state_dict[key].shape[-1])
                value = state_dict.pop(key)
                new_key = key.replace(dcp_prefix, hf_prefix, 1) if key.startswith(dcp_prefix) else f"{hf_prefix}{key}"

                # permute expert weight gate_up_proj from (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) to (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
                # permute expert weight down_proj from (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) to (self.num_experts, self.hidden_dim, 2 * self.intermediate_dim)
                # to meet gemm compute
                for pattern in self.expert_weight_name_patterns:
                    if re.fullmatch(pattern, key):
                        value = value.permute(0, 2, 1).contiguous()

                state_dict[new_key] = value

                # Optionally convert the weights to BF16
                if to_bf16:
                    state_dict[new_key] = state_dict[new_key].to(dtype=torch.bfloat16)
                if not has_mtp_weights_in_dcp and new_key.startswith("mtp."):
                    has_mtp_weights_in_dcp = True

            return state_dict

        merge_dcp_to_hf_sharded(
            load_dir=Path(dcp_dir),
            save_dir=Path(save_hf_dir),
            model_assets_dir=Path(origin_hf_dir),
            select_key_convert_func=lambda key: f"model.{dcp_prefix}" + key,
            state_dict_convert_func=state_dict_convert_func
        )
        mtp_state_dict = None

        # dcp中不包含mtp层，但hf需要保留mtp原始权重，需要从原始文件加载
        if not has_mtp_weights_in_dcp and keep_origin_mtp_weights:
            mtp_state_dict = self._load_origin_mtp_weights(origin_hf_dir)
        # moe模型的dcp中包含mtp层，需要对专家权重做split
        elif has_mtp_weights_in_dcp and mtp_num_hidden_layers > 0 and num_experts > 0:
            mtp_state_dict = self._load_and_split_mtp_experts(
                dcp_dir, dcp_prefix, hf_prefix, mtp_num_hidden_layers, num_experts, fused_linear_names, to_bf16
            )

        if mtp_state_dict:
            from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
            from checkpoint.common.merge_dcp_to_hf import update_safetensors_files

            index_path = Path(save_hf_dir) / SAFE_WEIGHTS_INDEX_NAME
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            update_safetensors_files(Path(save_hf_dir), mtp_state_dict, weight_map)

    def _load_and_split_mtp_experts(
        self,
        dcp_dir: str,
        dcp_prefix: str,
        hf_prefix: str,
        mtp_num_layers: int,
        num_experts: int,
        fused_linear_names: List[str] = None,
        to_bf16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        from torch.distributed.checkpoint import FileSystemReader
        from checkpoint.common.dcp_utils import load_metadata, extract_metadata, partial_load_dcp_state_dict
        from checkpoint.vlm_model.converters.moe_expert import split_moe_expert_weights, ConfigType

        storage_reader = FileSystemReader(Path(dcp_dir))
        metadata = load_metadata(storage_reader)

        mtp_dcp_keys = [k for k in metadata.state_dict_metadata.keys() if "mtp.layers." in k and ".mlp.experts." in k]
        if not mtp_dcp_keys:
            return {}

        partial_metadata = extract_metadata(mtp_dcp_keys, metadata)
        mtp_state_dict = partial_load_dcp_state_dict(partial_metadata, storage_reader)
        mtp_state_dict = mtp_state_dict.get("model", mtp_state_dict)

        for mtp_key in list(mtp_state_dict.keys()):
            # qwen3.5 currently does not support fused_linear_names
            # view experts weight: (expert_num * input_dim, output_dim) -> (expert_num, input_dim, output_dim)
            if fused_linear_names:
                if any(fused_linear_name in mtp_key for fused_linear_name in fused_linear_names):
                    mtp_state_dict[mtp_key] = mtp_state_dict[mtp_key].view(num_experts, -1, mtp_state_dict[mtp_key].shape[-1])

            new_mtp_key = mtp_key.replace(dcp_prefix, hf_prefix, 1) if mtp_key.startswith(dcp_prefix) else f"{hf_prefix}{mtp_key}"
            if new_mtp_key != mtp_key:
                mtp_state_dict[new_mtp_key] = mtp_state_dict.pop(mtp_key)

        mtp_weight_path = "mtp.layers.{layer}.mlp.experts.{expert}"
        split_moe_expert_weights(
            mtp_state_dict, mtp_num_layers, num_experts,
            expert_start_layer=0, config_type=ConfigType.QWEN3_5, weight_path=mtp_weight_path
        )

        for key in mtp_state_dict:
            mtp_state_dict[key] = mtp_state_dict[key].contiguous()
            if to_bf16:
                mtp_state_dict[key] = mtp_state_dict[key].to(dtype=torch.bfloat16)

        return mtp_state_dict

    @staticmethod
    def _load_origin_mtp_weights(
        origin_hf_dir: str,
    ) -> Dict[str, torch.Tensor]:
        from safetensors.torch import load_file
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

        origin_dir = Path(origin_hf_dir)
        index_path = origin_dir / SAFE_WEIGHTS_INDEX_NAME
        if not index_path.exists():
            return {}

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
        if not mtp_keys:
            return {}

        mtp_files = set(weight_map[k] for k in mtp_keys)
        mtp_state_dict = {}
        for sf_file in mtp_files:
            file_state_dict = load_file(origin_dir / sf_file)
            for key in mtp_keys:
                if key in file_state_dict:
                    mtp_state_dict[key] = file_state_dict[key]

        return mtp_state_dict

    @staticmethod
    def hf_to_mm():
        pass

    @staticmethod
    def mm_to_hf():
        pass

    @staticmethod
    def resplit():
        pass
