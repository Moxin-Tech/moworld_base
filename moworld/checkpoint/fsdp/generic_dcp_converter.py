from typing import List, Dict
from pathlib import Path
from transformers import AutoConfig, AutoProcessor

from checkpoint.common.converter import Converter
from checkpoint.common.merge_dcp_to_hf import merge_dcp_to_hf_sharded
from checkpoint.common.hf_to_dcp import hf_to_dcp_sharded


class GenericDCPConverter(Converter):
    """
    Generic DCP converter implementation supporting HF ↔ DCP format conversion for multiple model architectures

    Supports:
    - HF → DCP conversion
    - DCP → HF merging
    - Placeholder methods for megatron format and resharding operations.
    """
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

        def state_dict_convert_func(state_dict):
            if tie_weight_mapping:
                for tgt_weight, src_weight in tie_weight_mapping.items():
                    if src_weight in state_dict.keys():
                        state_dict[tgt_weight] = state_dict[src_weight]

            ori_keys = list(state_dict.keys())
            for ori_key in ori_keys:
                value = state_dict.pop(ori_key)

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

    def dcp_to_hf(
        self,
        load_dir: str = "mm_save_dir/release",
        save_dir: Path = "",
        model_assets_dir: str = "",
        dcp_prefix: str = "",
        hf_prefix: str = "",
        fused_linear_names: List[str] = None,
        trust_remote_code: bool = True
    ):
        """
        Merges torch-dcp shards and converts them back into standard Hugging Face format.

        This is typically used after training or inference in torch-dcp format to export
        a model that can be easily loaded with Hugging Face Transformers.
        Args:
            load_dir (str): Input: Directory containing DCP shards
            save_dir (Path): Output: Directory to save merged HF model
            model_assets_dir (str): Reference: Original HF model dir (for config/tokenizer)
            dcp_prefix (str): Prefix to remove from DCP format parameter names
            hf_prefix (str): Prefix to add for Hugging Face parameter names
            fused_linear_names (str): Names of MoE (Mixture of Experts) expert parameters
                in comma-separated format. These parameters need special reshaping during conversion.
        """
        config = AutoConfig.from_pretrained(model_assets_dir, trust_remote_code=trust_remote_code)

        def get_text_config(config):
            if hasattr(config, "text_config"):
                return config.text_config
            elif hasattr(config, "thinker_config") and hasattr(config.thinker_config, "text_config"): # support qwen3-omni
                return config.thinker_config.text_config
            return config

        text_config = get_text_config(config)
        num_experts = getattr(text_config, "num_experts", None)

        def state_dict_convert_func(state_dict):
            state_dict_keys = list(state_dict.keys())

            for key in state_dict_keys:
                # view experts weight: (expert_num * input_dim, output_dim) -> (expert_num, input_dim, output_dim)
                if fused_linear_names:
                    if num_experts and any(fused_linear_name in key for fused_linear_name in fused_linear_names):
                        state_dict[key] = state_dict[key].view(num_experts, -1, state_dict[key].shape[-1])
                value = state_dict.pop(key)
                new_key = key.replace(dcp_prefix, hf_prefix, 1) if key.startswith(dcp_prefix) else key
                state_dict[new_key] = value

            return state_dict

        merge_dcp_to_hf_sharded(
            load_dir=Path(load_dir),
            save_dir=Path(save_dir),
            model_assets_dir=Path(model_assets_dir),
            select_key_convert_func=lambda key: f"model.{dcp_prefix}" + key.removeprefix(hf_prefix),
            state_dict_convert_func=state_dict_convert_func
        )

    @staticmethod
    def hf_to_mm():
        pass

    @staticmethod
    def mm_to_hf():
        pass

    @staticmethod
    def resplit():
        pass
