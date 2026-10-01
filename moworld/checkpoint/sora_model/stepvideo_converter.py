import torch
from checkpoint.sora_model.convert_utils.tp_patterns import TPPattern
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.utils import check_method_support
from checkpoint.sora_model.convert_utils.save_load_utils import load_from_hf, save_as_mm


class KVfusedColumnTP(TPPattern):
    @staticmethod
    def split(weight, tp_size):
        wk, wv = torch.chunk(weight, 2, dim=0)
        wks = torch.chunk(wk, tp_size, dim=0)
        wvs = torch.chunk(wv, tp_size, dim=0)
        weights = [torch.cat([wks[i], wvs[i]], dim=0) for i in range(tp_size)]
        return weights

    @staticmethod
    def merge(weights):
        chunked_weights = [torch.chunk(weight, 2, dim=0) for weight in weights]

        wks = [chunk[0] for chunk in chunked_weights]
        wvs = [chunk[1] for chunk in chunked_weights]

        weight = torch.cat([
            torch.cat(wks, dim=0),
            torch.cat(wvs, dim=0)
        ], dim=0)
        return weight


class LayerIndexConverter:
    @staticmethod
    def get_layer_index(name):
        if name.startswith("transformer_blocks"):
            idx = int(name.split('.')[1])
            return idx
        return None

    @staticmethod
    def convert_layer_index(name, new_layer_index):
        if name.startswith("transformer_blocks"):
            parts = name.split('.')
            parts[1] = str(new_layer_index)
            return ".".join(parts)
        return name


class StepVideoConverter(SoraModelConverter):
    """Converter for StepVideo"""

    _supported_methods = ["hf_to_mm", "resplit"]
    _enable_tp = True
    _enable_pp = True
    _enable_vpp = False

    hf_to_mm_convert_mapping = {
        "pos_embed.proj.bias": "pos_embed.proj.bias",
        "pos_embed.proj.weight": "pos_embed.proj.weight",
        "scale_shift_table": "scale_shift_table",
        "adaln_single.emb.timestep_embedder.linear_1.bias": "adaln_single.emb.timestep_embedder.linear_1.bias",
        "adaln_single.emb.timestep_embedder.linear_1.weight": "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias": "adaln_single.emb.timestep_embedder.linear_2.bias",
        "adaln_single.emb.timestep_embedder.linear_2.weight": "adaln_single.emb.timestep_embedder.linear_2.weight",
        "caption_projection.linear_1.bias": "caption_projection.linear_1.bias",
        "caption_projection.linear_1.weight": "caption_projection.linear_1.weight",
        "caption_projection.linear_2.bias": "caption_projection.linear_2.bias",
        "caption_projection.linear_2.weight": "caption_projection.linear_2.weight",
        "clip_projection.bias": "clip_projection.bias",
        "clip_projection.weight": "clip_projection.weight",
        "proj_out.bias": "proj_out.bias",
        "proj_out.weight": "proj_out.weight"
    }

    pre_process_weight_names = [
        "pos_embed.proj.bias", "pos_embed.proj.weight",
        "adaln_single.emb.timestep_embedder.linear_1.bias", "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_2.bias", "adaln_single.emb.timestep_embedder.linear_2.weight",
        "adaln_single.linear.weight", "adaln_single.linear.bias",
        "caption_projection.linear_1.bias", "caption_projection.linear_1.weight",
        "caption_projection.linear_2.bias", "caption_projection.linear_2.weight",
        "clip_projection.bias", "clip_projection.weight"
    ]

    post_preprocess_weight_names = [
        "scale_shift_table",
        "proj_out.bias", "proj_out.weight"
    ]

    tp_split_mapping = {
        "column_parallel_tp": [
            "adaln_single.linear.weight",
            "adaln_single.linear.bias",
        ],
        "row_parallel_tp": [],
        "qkv_fused_column_tp": []
    }

    kv_fused_column_tp = KVfusedColumnTP()
    spec_tp_split_mapping = {kv_fused_column_tp: []}
    layer_index_converter = LayerIndexConverter()

    def __init__(self) -> None:
        super().__init__()

        num_layers = 48
        self.num_heads = 48
        for index in range(num_layers):
            self.hf_to_mm_convert_mapping.update({
                f"transformer_blocks.{index}.attn1.k_norm.weight": f"transformer_blocks.{index}.attn1.k_norm.weight",
                f"transformer_blocks.{index}.attn1.q_norm.weight": f"transformer_blocks.{index}.attn1.q_norm.weight",
                f"transformer_blocks.{index}.attn1.wo.weight": f"transformer_blocks.{index}.attn1.proj_out.weight",
                f"transformer_blocks.{index}.attn1.wqkv.weight": f"transformer_blocks.{index}.attn1.proj_qkv.weight",
                f"transformer_blocks.{index}.attn2.k_norm.weight": f"transformer_blocks.{index}.attn2.k_norm.weight",
                f"transformer_blocks.{index}.attn2.q_norm.weight": f"transformer_blocks.{index}.attn2.q_norm.weight",
                f"transformer_blocks.{index}.attn2.wkv.weight": f"transformer_blocks.{index}.attn2.proj_kv.weight",
                f"transformer_blocks.{index}.attn2.wo.weight": f"transformer_blocks.{index}.attn2.proj_out.weight",
                f"transformer_blocks.{index}.attn2.wq.weight": f"transformer_blocks.{index}.attn2.proj_q.weight",
                f"transformer_blocks.{index}.ff.net.0.proj.weight": f"transformer_blocks.{index}.ff.net.0.proj.weight",
                f"transformer_blocks.{index}.ff.net.2.weight": f"transformer_blocks.{index}.ff.net.2.weight",
                f"transformer_blocks.{index}.norm1.bias": f"transformer_blocks.{index}.norm1.bias",
                f"transformer_blocks.{index}.norm1.weight": f"transformer_blocks.{index}.norm1.weight",
                f"transformer_blocks.{index}.norm2.bias": f"transformer_blocks.{index}.norm2.bias",
                f"transformer_blocks.{index}.norm2.weight": f"transformer_blocks.{index}.norm2.weight",
                f"transformer_blocks.{index}.scale_shift_table": f"transformer_blocks.{index}.scale_shift_table"
            })

            self.tp_split_mapping["column_parallel_tp"] += [
                f"transformer_blocks.{index}.ff.net.0.proj.weight",
                f"transformer_blocks.{index}.attn2.proj_q.weight",
            ]

            self.tp_split_mapping["row_parallel_tp"] += [
                f"transformer_blocks.{index}.attn1.proj_out.weight",
                f"transformer_blocks.{index}.attn2.proj_out.weight",
                f"transformer_blocks.{index}.ff.net.2.weight",
            ]

            self.tp_split_mapping["qkv_fused_column_tp"] += [
                f"transformer_blocks.{index}.attn1.proj_qkv.weight",
            ]

            self.spec_tp_split_mapping[self.kv_fused_column_tp] += [
                f"transformer_blocks.{index}.attn2.proj_kv.weight",
            ]

    @check_method_support
    def hf_to_mm(self, cfg: ConvertConfig):
        state_dict = load_from_hf(cfg.source_path)
        state_dict = self._replace_state_dict(
            state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping
        )
        state_dict = self._xfuse_to_mm(state_dict)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    def _xfuse_to_mm(self, state_dict):

        def head_weight_permute(weight, fuse_num):
            weight_per_heads = torch.chunk(weight, self.num_heads)
            part_weight_per_heads = [
                torch.chunk(weight_per_head, fuse_num, dim=0)
                for weight_per_head in weight_per_heads
            ]

            part_weights = []
            for i in range(fuse_num):
                part_weights.append(
                    torch.cat([part_weight_per_head[i] for part_weight_per_head in part_weight_per_heads], dim=0)
                )
            weight = torch.cat(part_weights, dim=0).clone()
            return weight

        keys = state_dict.keys()
        for key in keys:
            if key in self.tp_split_mapping["qkv_fused_column_tp"]:
                state_dict[key] = head_weight_permute(state_dict[key], fuse_num=3)
            elif key in self.spec_tp_split_mapping[self.kv_fused_column_tp]:
                state_dict[key] = head_weight_permute(state_dict[key], fuse_num=2)

        return state_dict
