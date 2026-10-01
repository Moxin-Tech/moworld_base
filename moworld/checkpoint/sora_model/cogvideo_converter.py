from typing_extensions import Literal
import torch

from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.save_load_utils import (
    load_pt,
    load_from_hf,
    save_as_mm
)
from checkpoint.sora_model.convert_utils.utils import check_method_support


class LayerIndexConverter:
    @staticmethod
    def get_layer_index(name):
        if name.startswith("videodit_blocks"):
            idx = int(name.split('.')[1])
            return idx
        return None

    @staticmethod
    def convert_layer_index(name, new_layer_index):
        if name.startswith("videodit_blocks"):
            parts = name.split('.')
            parts[1] = str(new_layer_index)
            return ".".join(parts)
        return name


class CogVideoConverter(SoraModelConverter):
    """Converter for CogVideo"""

    _supported_methods = ["hf_to_mm", "resplit", "source_to_mm", "layerzero_to_mm", "merge_lora_to_base"]
    _enable_tp = True
    _enable_pp = True
    _enable_vpp = False

    convert_mapping = {
        "model.diffusion_model.time_embed.0.bias": "time_embed.time_embed.0.bias",
        "model.diffusion_model.time_embed.0.weight": "time_embed.time_embed.0.weight",
        "model.diffusion_model.time_embed.2.bias": "time_embed.time_embed.2.bias",
        "model.diffusion_model.time_embed.2.weight": "time_embed.time_embed.2.weight",
        "model.diffusion_model.mixins.patch_embed.proj.bias": "patch_embed.proj.bias",
        "model.diffusion_model.mixins.patch_embed.proj.weight": "patch_embed.proj.weight",
        "model.diffusion_model.mixins.patch_embed.text_proj.bias": "caption_projection.bias",
        "model.diffusion_model.mixins.patch_embed.text_proj.weight": "caption_projection.weight",
        "model.diffusion_model.mixins.pos_embed.freqs_cos": "pos_embed.freqs_cos",
        "model.diffusion_model.mixins.pos_embed.freqs_sin": "pos_embed.freqs_sin",
        "model.diffusion_model.transformer.final_layernorm.weight": "norm_final.weight",
        "model.diffusion_model.transformer.final_layernorm.bias": "norm_final.bias",
        "model.diffusion_model.mixins.final_layer.norm_final.weight": "norm_out.weight",
        "model.diffusion_model.mixins.final_layer.norm_final.bias": "norm_out.bias",
        "model.diffusion_model.mixins.final_layer.linear.weight": "proj_out_linear.weight",
        "model.diffusion_model.mixins.final_layer.linear.bias": "proj_out_linear.bias",
        "model.diffusion_model.mixins.final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
        "model.diffusion_model.mixins.final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
    }

    hf_to_mm_convert_mapping = {
        "time_embedding.linear_1.bias": "time_embed.time_embed.0.bias",
        "time_embedding.linear_1.weight": "time_embed.time_embed.0.weight",
        "time_embedding.linear_2.bias": "time_embed.time_embed.2.bias",
        "time_embedding.linear_2.weight": "time_embed.time_embed.2.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.text_proj.bias": "caption_projection.bias",
        "patch_embed.text_proj.weight": "caption_projection.weight",
        "mixins.pos_embed.freqs_cos": "pos_embed.freqs_cos",
        "mixins.pos_embed.freqs_sin": "pos_embed.freqs_sin",
        "norm_final.weight": "norm_final.weight",
        "norm_final.bias": "norm_final.bias",
        "norm_out.norm.weight": "norm_out.weight",
        "norm_out.norm.bias": "norm_out.bias",
        "proj_out.weight": "proj_out_linear.weight",
        "proj_out.bias": "proj_out_linear.bias",
        "norm_out.linear.weight": "adaLN_modulation.1.weight",
        "norm_out.linear.bias": "adaLN_modulation.1.bias"
    }

    pre_process_weight_names = [
        "time_embed.time_embed.0.bias", "time_embed.time_embed.0.weight",
        "time_embed.time_embed.2.bias", "time_embed.time_embed.2.weight",
        "patch_embed.proj.bias", "patch_embed.proj.weight",
        "caption_projection.bias", "caption_projection.weight"
    ] # pre_process layers for pp

    post_preprocess_weight_names = [
        "norm_final.weight", "norm_final.bias",
        "norm_out.weight", "norm_out.bias",
        "proj_out_linear.weight", "proj_out_linear.bias",
        "adaLN_modulation.1.weight", "adaLN_modulation.1.bias"
    ] # post_process layers for pp

    layer_index_converter = LayerIndexConverter()

    tp_split_mapping = {
        "column_parallel_tp": ["adaLN_modulation.1.weight", "adaLN_modulation.1.bias"],
        "row_parallel_tp": [],
        "qkv_fused_column_tp": []
    }

    def __init__(self, version: Literal["t2v", "i2v"] = "t2v", remove_pos_emb: bool = False):
        self.version = version
        self.remove_pos_emb = remove_pos_emb
        if version == "i2v":
            self.convert_mapping.update({
                "model.diffusion_model.mixins.pos_embed.pos_embedding": "pos_embed.pos_embedding",
                "model.diffusion_model.ofs_embed.0.bias": "ofs_embed.0.bias",
                "model.diffusion_model.ofs_embed.0.weight": "ofs_embed.0.weight",
                "model.diffusion_model.ofs_embed.2.bias": "ofs_embed.2.bias",
                "model.diffusion_model.ofs_embed.2.weight": "ofs_embed.2.weight"
            })

        num_layers = 42
        self.num_layers = num_layers

        for i in range(num_layers):
            self.convert_mapping.update({
                f"model.diffusion_model.mixins.adaln_layer.adaLN_modulations.{i}.1.bias": f"videodit_blocks.{i}.scale_shift_table.1.bias",
                f"model.diffusion_model.mixins.adaln_layer.adaLN_modulations.{i}.1.weight": f"videodit_blocks.{i}.scale_shift_table.1.weight",
                f"model.diffusion_model.mixins.adaln_layer.query_layernorm_list.{i}.bias": f"videodit_blocks.{i}.self_atten.q_norm.bias",
                f"model.diffusion_model.mixins.adaln_layer.query_layernorm_list.{i}.weight": f"videodit_blocks.{i}.self_atten.q_norm.weight",
                f"model.diffusion_model.mixins.adaln_layer.key_layernorm_list.{i}.bias": f"videodit_blocks.{i}.self_atten.k_norm.bias",
                f"model.diffusion_model.mixins.adaln_layer.key_layernorm_list.{i}.weight": f"videodit_blocks.{i}.self_atten.k_norm.weight",
                f"model.diffusion_model.transformer.layers.{i}.input_layernorm.bias": f"videodit_blocks.{i}.norm1.bias",
                f"model.diffusion_model.transformer.layers.{i}.input_layernorm.weight": f"videodit_blocks.{i}.norm1.weight",
                f"model.diffusion_model.transformer.layers.{i}.attention.dense.bias": f"videodit_blocks.{i}.self_atten.proj_out.bias",
                f"model.diffusion_model.transformer.layers.{i}.attention.dense.weight": f"videodit_blocks.{i}.self_atten.proj_out.weight",
                f"model.diffusion_model.transformer.layers.{i}.post_attention_layernorm.bias": f"videodit_blocks.{i}.norm2.bias",
                f"model.diffusion_model.transformer.layers.{i}.post_attention_layernorm.weight": f"videodit_blocks.{i}.norm2.weight",
                f"model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.bias": f"videodit_blocks.{i}.ff.net.0.proj.bias",
                f"model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.weight": f"videodit_blocks.{i}.ff.net.0.proj.weight",
                f"model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.bias": f"videodit_blocks.{i}.ff.net.2.bias",
                f"model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.weight": f"videodit_blocks.{i}.ff.net.2.weight",
                f"model.diffusion_model.transformer.layers.{i}.attention.query_key_value.weight": f"videodit_blocks.{i}.self_atten.proj_qkv.weight",
                f"model.diffusion_model.transformer.layers.{i}.attention.query_key_value.bias": f"videodit_blocks.{i}.self_atten.proj_qkv.bias"
            })

            self.hf_to_mm_convert_mapping.update({
                f"transformer_blocks.{i}.attn1.norm_q.bias": f"videodit_blocks.{i}.self_atten.q_norm.bias",
                f"transformer_blocks.{i}.attn1.norm_q.weight": f"videodit_blocks.{i}.self_atten.q_norm.weight",
                f"transformer_blocks.{i}.attn1.norm_k.bias": f"videodit_blocks.{i}.self_atten.k_norm.bias",
                f"transformer_blocks.{i}.attn1.norm_k.weight": f"videodit_blocks.{i}.self_atten.k_norm.weight",
                f"transformer_blocks.{i}.norm1.norm.bias": f"videodit_blocks.{i}.norm1.bias",
                f"transformer_blocks.{i}.norm1.norm.weight": f"videodit_blocks.{i}.norm1.weight",
                f"transformer_blocks.{i}.attn1.to_out.0.bias": f"videodit_blocks.{i}.self_atten.proj_out.bias",
                f"transformer_blocks.{i}.attn1.to_out.0.weight": f"videodit_blocks.{i}.self_atten.proj_out.weight",
                f"transformer_blocks.{i}.norm2.norm.bias": f"videodit_blocks.{i}.norm2.bias",
                f"transformer_blocks.{i}.norm2.norm.weight": f"videodit_blocks.{i}.norm2.weight",
                f"transformer_blocks.{i}.ff.net.0.proj.bias": f"videodit_blocks.{i}.ff.net.0.proj.bias",
                f"transformer_blocks.{i}.ff.net.0.proj.weight": f"videodit_blocks.{i}.ff.net.0.proj.weight",
                f"transformer_blocks.{i}.ff.net.2.bias": f"videodit_blocks.{i}.ff.net.2.bias",
                f"transformer_blocks.{i}.ff.net.2.weight": f"videodit_blocks.{i}.ff.net.2.weight"
            })

            self.tp_split_mapping["column_parallel_tp"] += [
                f"videodit_blocks.{i}.ff.net.0.proj.weight",
                f"videodit_blocks.{i}.ff.net.0.proj.bias",
                f"videodit_blocks.{i}.scale_shift_table.1.weight",
                f"videodit_blocks.{i}.scale_shift_table.1.bias"
            ]

            self.tp_split_mapping["row_parallel_tp"] += [
                f"videodit_blocks.{i}.self_atten.proj_out.weight",
                f"videodit_blocks.{i}.ff.net.2.weight",
            ]

            self.tp_split_mapping["qkv_fused_column_tp"] += [
                f"videodit_blocks.{i}.self_atten.proj_qkv.weight",
                f"videodit_blocks.{i}.self_atten.proj_qkv.bias"
            ]

        if not remove_pos_emb:
            self.pre_process_weight_names += ["pos_embed.freqs_cos", "pos_embed.freqs_sin"]
            if version == "i2v":
                self.pre_process_weight_names += ["pos_embed.pos_embedding"]

    def _remove_state_dict(self, state_dict, remove_keys):
        for remove_key in remove_keys:
            if remove_key in state_dict.keys():
                state_dict.pop(remove_key)
        return state_dict

    def _remove_pos_emb(self, state_dict: dict):
        remove_keys = ["pos_embed.freqs_cos", "pos_embed.freqs_sin"]
        if self.version == "i2v":
            remove_keys.append("pos_embed.pos_embedding")
        state_dict = self._remove_state_dict(state_dict, remove_keys)
        return state_dict

    @check_method_support
    def source_to_mm(self, cfg: ConvertConfig):
        state_dict = load_pt(cfg.source_path, module_name="module")
        state_dict = self._replace_state_dict(
            state_dict,
            self.convert_mapping,
            self.str_replace_mapping
        )
        if self.remove_pos_emb:
            state_dict = self._remove_pos_emb(state_dict)
        # remove dummy layers
        remove_keys = set(state_dict.keys()) - set(self.convert_mapping.values())
        state_dict = self._remove_state_dict(state_dict, remove_keys)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    @check_method_support
    def hf_to_mm(self, cfg: ConvertConfig):
        state_dict = load_from_hf(cfg.source_path)
        state_dict = self._replace_state_dict(
            state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping
        )
        state_dict = self._hf_to_mm_replace_state_dict(state_dict)
        if self.remove_pos_emb:
            state_dict = self._remove_pos_emb(state_dict)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    def _hf_to_mm_replace_state_dict(self, state_dict: dict):
        for i in range(self.num_layers):
            # fuse qkv proj linear
            q_weight = state_dict.pop(f"transformer_blocks.{i}.attn1.to_q.weight")
            k_weight = state_dict.pop(f"transformer_blocks.{i}.attn1.to_k.weight")
            v_weight = state_dict.pop(f"transformer_blocks.{i}.attn1.to_v.weight")
            state_dict[f"videodit_blocks.{i}.self_atten.proj_qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)
            q_bias = state_dict.pop(f"transformer_blocks.{i}.attn1.to_q.bias")
            k_bias = state_dict.pop(f"transformer_blocks.{i}.attn1.to_k.bias")
            v_bias = state_dict.pop(f"transformer_blocks.{i}.attn1.to_v.bias")
            state_dict[f"videodit_blocks.{i}.self_atten.proj_qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)

            # permute scale_shift_table
            norm1_linear_bias = state_dict.pop(f"transformer_blocks.{i}.norm1.linear.bias")
            norm2_linear_bias = state_dict.pop(f"transformer_blocks.{i}.norm2.linear.bias")
            norm1_linear_weight = state_dict.pop(f"transformer_blocks.{i}.norm1.linear.weight")
            norm2_linear_weight = state_dict.pop(f"transformer_blocks.{i}.norm2.linear.weight")
            norm1_bias = torch.chunk(norm1_linear_bias, 6, dim=0)
            norm2_bias = torch.chunk(norm2_linear_bias, 6, dim=0)
            norm1_weight = torch.chunk(norm1_linear_weight, 6, dim=0)
            norm2_weight = torch.chunk(norm2_linear_weight, 6, dim=0)
            state_dict[f"videodit_blocks.{i}.scale_shift_table.1.bias"] = torch.cat(
                norm1_bias[0:3] + norm2_bias[0:3] + norm1_bias[3:6] + norm2_bias[3:6]
            )
            state_dict[f"videodit_blocks.{i}.scale_shift_table.1.weight"] = torch.cat(
                norm1_weight[0:3] + norm2_weight[0:3] + norm1_weight[3:6] + norm2_weight[3:6]
            )

        return state_dict
