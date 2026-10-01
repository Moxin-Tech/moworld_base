# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import torch.nn.functional as F
from megatron.core import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig, MLATransformerConfig
from megatron.training import get_args

from mindspeed_mm.configs.config import ConfigReader
from mindspeed_mm.utils.utils import get_dtype, quick_gelu, gelu_tanh
from mindspeed_mm.utils.transformer_model_config import get_class_variables, MindSpeedArgsRequired


def get_model_config(config):
    global_args = get_args()
    config_dict = config.to_dict()
    if "model_id" in config_dict and config_dict["model_id"] == "InternVLMLP":
        config_dict["params_dtype"] = "bf16"
        config_dict["hidden_size"] = 4096
        config_dict["num_attention_heads"] = 1
        config_dict["num_layers"] = 1
    if "model_id" in config_dict and config_dict["model_id"] == "Qwen2.5llm":
        config_dict["use_repeat_kv"] = True
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
        t_config["kv_channels"] = t_config.get("hidden_size") // t_config.get("num_attention_heads")
    if t_config.get("ffn_hidden_size") is None and t_config.get("hidden_size"):
        t_config["ffn_hidden_size"] = 4 * t_config.get("hidden_size")
    if t_config.get("num_query_groups") is None and t_config.get("num_attention_heads") != 1:
        t_config["num_query_groups"] = t_config.get("num_attention_heads")

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
