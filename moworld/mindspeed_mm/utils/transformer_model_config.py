from dataclasses import dataclass
from functools import wraps
from typing import Any
import torch.nn.functional as F

from megatron.core import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig
from megatron.training import get_args
from mindspeed.patch_utils import MindSpeedPatchesManager as pm

from mindspeed_mm.configs.config import ConfigReader
from .utils import get_dtype, quick_gelu, gelu_tanh


def get_class_variables(cls):
    all_members = dir(cls)
    filtered_members = [member for member in all_members if not member.startswith("__")]

    return filtered_members


def get_model_config(config):
    global_args = get_args()
    config_dict = config.to_dict()
    # for moe
    if "moe_intermediate_size" in config_dict:
        config_dict["moe_ffn_hidden_size"] = config_dict["moe_intermediate_size"]
    if "n_shared_experts" in config_dict:
        config_dict["moe_shared_expert_intermediate_size"] = (config_dict["n_shared_experts"] *
                                                              config_dict["moe_ffn_hidden_size"])

    t_config = dict()
    if getattr(global_args, "multi_latent_attention", False):
        tfc_variables = get_class_variables(MLATransformerConfig)
    else:
        tfc_variables = get_class_variables(TransformerConfig)
    mpc_variables = get_class_variables(ModelParallelConfig)
    for key in tfc_variables:
        if key in config_dict.keys():
            t_config[key] = config_dict[key]
        elif key in mpc_variables and hasattr(global_args, key):
            t_config[key] = getattr(global_args, key)

    t_config["params_dtype"] = get_dtype(t_config.get("params_dtype"))
    if t_config.get("activation_func") == "silu":
        t_config["activation_func"] = F.silu
    elif t_config.get("activation_func") == "quick_gelu":
        t_config["activation_func"] = quick_gelu
    elif t_config.get("activation_func") == "gelu_pytorch_tanh":
        t_config["activation_func"] = gelu_tanh
    else:
        t_config["activation_func"] = F.gelu

    if t_config.get("kv_channels") is None and t_config.get("hidden_size") and t_config.get("num_attention_heads"):
        t_config["kv_channels"] = t_config["hidden_size"] // t_config["num_attention_heads"]
    if t_config.get("ffn_hidden_size") is None and t_config.get("hidden_size"):
        t_config["ffn_hidden_size"] = 4 * t_config["hidden_size"]
    if t_config.get("num_attention_heads") is None:
        t_config["num_attention_heads"] = 0
    if t_config.get("num_query_groups") is None and t_config.get("num_attention_heads"):
        t_config["num_query_groups"] = t_config.get("num_attention_heads")
    if t_config.get("cp_comm_type") is None:
        t_config["cp_comm_type"] = None

    if getattr(global_args, "multi_latent_attention", False):
        t_config["rope_type"] = "rope"
        trans_config = MLATransformerConfig(**t_config)
    else:
        trans_config = TransformerConfig(**t_config)

    # Update config dict from TransformerConfig
    for key in tfc_variables:
        config_dict[key] = getattr(trans_config, key)

    # Add MindSpeedArgs that needed from global args
    mindspeed_variables = get_class_variables(MindSpeedArgsRequired)
    for key in mindspeed_variables:
        if key not in config_dict:
            config_dict[key] = getattr(global_args, key)

    new_config = ConfigReader(config_dict)

    return new_config


@dataclass
class MindSpeedArgsRequired:
    """Base configuration for MindSpeed Core"""

    # Train basic
    micro_batch_size: Any = None

    # Flash attention
    pre_tockens: Any = None
    next_tockens: Any = None
    sparse_mode: Any = None
    use_fusion_attn_v2: Any = None

    # Distributed args
    overlap_param_gather: Any = None
    overlap_grad_reduce: Any = None

    # alibi args
    alibi_fusion_attn_type: Any = None

    # 2d tp
    tp_2d: Any = None

    # CP args
    context_parallel_algo: Any = None
    context_parallel_kv_cache_policy: Any = None
    context_parallel_cache_interval: Any = None
    use_ulysses_allgather_kv: Any = None
    megatron_cp_in_bnsd: Any = None
    attention_mask_type: Any = None
    use_cp_send_recv_overlap: Any = None

    # Unaligned SP
    variable_seq_lengths: Any = None

    # Ring Attention CP
    reset_attention_mask: Any = None

    # CoC configs
    use_ascend_coc: Any = None
    coc_mode: Any = None
    coc_parallel_num: Any = None
    coc_fused_kernel: Any = None


def tranformer_config_post_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        """Modified from __post_init__ method of TransformerConfig to adapt MLP config"""
        if self.kv_channels is None and self.num_attention_heads == 0:
            self.kv_channels = 0
        if self.pipeline_dtype is None:
            self.pipeline_dtype = self.params_dtype
        fn(self)
    return wrapper


pm.register_patch("megatron.core.transformer.transformer_config.TransformerConfig.__post_init__", tranformer_config_post_init_wrapper)
pm.apply_patches()
