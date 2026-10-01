from typing_extensions import Literal
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.save_load_utils import (
    load_pt,
    save_as_pt,
    save_as_mm
)
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.utils import check_method_support


class LayerIndexConverter:
    @staticmethod
    def get_layer_index(name):
        if name.startswith("videodit_sparse_blocks"):
            idx = int(name.split('.')[1])
            return idx
        return None

    @staticmethod
    def convert_layer_index(name, new_layer_index):
        if name.startswith("videodit_sparse_blocks"):
            parts = name.split('.')
            parts[1] = str(new_layer_index)
            return ".".join(parts)
        return name


class OpenSoraPlanConverter(SoraModelConverter):
    """Converter for OpenSoraPlan"""

    def __init__(self, version: Literal["v1.2", "v1.3", "v1.5"] = "v1.5") -> None:
        super().__init__()
        self.version = version

        if self.version == "v1.2":
            self._supported_methods = ["hf_to_mm", "resplit"]
            self.hf_to_mm_str_replace_mapping = {
                "transformer_blocks": "videodit_blocks",
                "attn1": "self_atten",
                "attn2": "cross_atten",
                "to_q": "proj_q",
                "to_k": "proj_k",
                "to_v": "proj_v",
                "to_out.0": "proj_out",
                "to_out.1": "dropout"
            }

            self._enable_tp = True
            num_layers = 32
            for i in range(num_layers):
                self.tp_split_mapping["column_parallel_tp"] += [
                    f"videodit_blocks.{i}.self_atten.proj_q.weight",
                    f"videodit_blocks.{i}.self_atten.proj_q.bias",
                    f"videodit_blocks.{i}.cross_atten.proj_q.weight",
                    f"videodit_blocks.{i}.cross_atten.proj_q.bias",
                    f"videodit_blocks.{i}.self_atten.proj_k.weight",
                    f"videodit_blocks.{i}.self_atten.proj_k.bias",
                    f"videodit_blocks.{i}.cross_atten.proj_k.weight",
                    f"videodit_blocks.{i}.cross_atten.proj_k.bias",
                    f"videodit_blocks.{i}.self_atten.proj_v.weight",
                    f"videodit_blocks.{i}.self_atten.proj_v.bias",
                    f"videodit_blocks.{i}.cross_atten.proj_v.weight",
                    f"videodit_blocks.{i}.cross_atten.proj_v.bias",
                    f"videodit_blocks.{i}.ff.net.0.proj.weight",
                    f"videodit_blocks.{i}.ff.net.0.proj.bias"
                ]
                self.tp_split_mapping["row_parallel_tp"] += [
                    f"videodit_blocks.{i}.self_atten.proj_out.weight",
                    f"videodit_blocks.{i}.cross_atten.proj_out.weight",
                    f"videodit_blocks.{i}.ff.net.2.weight"
                ]

        elif self.version == "v1.3":
            self._supported_methods = ["hf_to_mm", "resplit"]
            self.hf_to_mm_str_replace_mapping = {
                "transformer_blocks": "videodit_sparse_blocks",
                "attn1": "self_atten",
                "attn2": "cross_atten",
                "to_q": "proj_q",
                "to_k": "proj_k",
                "to_v": "proj_v",
                "to_out.0": "proj_out",
                "to_out.1": "dropout",
            }

            self._enable_tp = True
            num_layers = 32
            for i in range(num_layers):
                self.tp_split_mapping["column_parallel_tp"] += [
                    f"videodit_sparse_blocks.{i}.self_atten.proj_q.weight",
                    f"videodit_sparse_blocks.{i}.self_atten.proj_q.bias",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_q.weight",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_q.bias",
                    f"videodit_sparse_blocks.{i}.self_atten.proj_k.weight",
                    f"videodit_sparse_blocks.{i}.self_atten.proj_k.bias",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_k.weight",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_k.bias",
                    f"videodit_sparse_blocks.{i}.self_atten.proj_v.weight",
                    f"videodit_sparse_blocks.{i}.self_atten.proj_v.bias",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_v.weight",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_v.bias",
                    f"videodit_sparse_blocks.{i}.ff.net.0.proj.weight",
                    f"videodit_sparse_blocks.{i}.ff.net.0.proj.bias",
                ]
                self.tp_split_mapping["row_parallel_tp"] += [
                    f"videodit_sparse_blocks.{i}.self_atten.proj_out.weight",
                    f"videodit_sparse_blocks.{i}.cross_atten.proj_out.weight",
                    f"videodit_sparse_blocks.{i}.ff.net.2.weight",
                ]

            self._enable_pp = True
            self._enable_vpp = True
            self.pre_process_weight_names = [
                "adaln_single.emb.timestep_embedder.linear_1.bias",
                "adaln_single.emb.timestep_embedder.linear_1.weight",
                "adaln_single.emb.timestep_embedder.linear_2.bias",
                "adaln_single.emb.timestep_embedder.linear_2.weight",
                "adaln_single.linear.bias",
                "adaln_single.linear.weight",
                "caption_projection.linear_1.bias",
                "caption_projection.linear_1.weight",
                "caption_projection.linear_2.bias",
                "caption_projection.linear_2.weight",
                "pos_embed.proj.bias",
                "pos_embed.proj.weight",
            ]
            self.post_preprocess_weight_names = ['scale_shift_table', 'proj_out.weight', 'proj_out.bias']
            self.layer_index_converter = LayerIndexConverter()

        elif self.version == "v1.5":
            self._supported_methods = ["source_to_mm", "resplit"]
            self.str_replace_mapping = {
                "attn1.norm_q": "attn1.norm_proj_q",
                "attn1.norm_k": "attn1.norm_proj_k",
                "attn1.to_q": "attn1.proj_q",
                "attn1.to_k": "attn1.proj_k",
                "attn1.to_v": "attn1.proj_v",
                "attn1.add_q_proj": "attn1.added_proj_q",
                "attn1.add_k_proj": "attn1.added_proj_k",
                "attn1.add_v_proj": "attn1.added_proj_v",
                "attn1.to_out.0": "attn1.proj_out",
                "attn1.to_add_out": "attn1.added_proj_out",
                "attn1.norm_added_q": "attn1.norm_added_proj_q",
                "attn1.norm_added_k": "attn1.norm_added_proj_k",
            }

            self._enable_tp = True
            num_layers = [2, 4, 6, 8, 6, 4, 2]
            self.tp_split_mapping["column_parallel_tp"] += [
                "norm_out.linear.weight",
                "norm_out.linear.bias"
            ]
            for i, nums in enumerate(num_layers):
                for j in range(nums):
                    self.tp_split_mapping["column_parallel_tp"] += [
                        f"transformer_blocks.{i}.{j}.attn1.proj_q.weight",
                        f"transformer_blocks.{i}.{j}.attn1.proj_q.bias",
                        f"transformer_blocks.{i}.{j}.attn1.proj_k.weight",
                        f"transformer_blocks.{i}.{j}.attn1.proj_k.bias",
                        f"transformer_blocks.{i}.{j}.attn1.proj_v.weight",
                        f"transformer_blocks.{i}.{j}.attn1.proj_v.bias",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_q.weight",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_q.bias",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_k.weight",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_k.bias",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_v.weight",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_v.bias",
                        f"transformer_blocks.{i}.{j}.ff.net.0.proj.weight",
                        f"transformer_blocks.{i}.{j}.ff_enc.net.0.proj.weight",
                        f"transformer_blocks.{i}.{j}.norm1.linear.weight",
                        f"transformer_blocks.{i}.{j}.norm1.linear.bias",
                        f"transformer_blocks.{i}.{j}.norm2.linear.weight",
                        f"transformer_blocks.{i}.{j}.norm2.linear.bias",
                    ]
                    self.tp_split_mapping["row_parallel_tp"] += [
                        f"transformer_blocks.{i}.{j}.attn1.proj_out.weight",
                        f"transformer_blocks.{i}.{j}.attn1.added_proj_out.weight",
                        f"transformer_blocks.{i}.{j}.ff.net.2.weight",
                        f"transformer_blocks.{i}.{j}.ff_enc.net.2.weight",
                    ]
        else:
            raise NotImplementedError(f"version: {version} is not supported for OpenSoraPlanConverter")

    @check_method_support
    def source_to_mm(self, cfg: ConvertConfig):
        state_dict = load_pt(cfg.source_path)
        state_dict = self._replace_state_dict(
            state_dict,
            self.convert_mapping,
            self.str_replace_mapping
        )
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    def _replace_state_dict(
        self,
        state_dict: dict,
        convert_mapping: dict = None,
        str_replace_mapping: dict = None,
    ):
        state_dict = state_dict.get("ema_state_dict", state_dict)
        return super()._replace_state_dict(state_dict, convert_mapping, str_replace_mapping)

    def vae_convert(
        self,
        cfg: ConvertConfig,
        use_ema_model: bool = True
    ):
        state_dict = load_pt(cfg.source_path)
        if (
            "ema_state_dict" in state_dict
            and len(state_dict["ema_state_dict"]) > 0
            and use_ema_model
        ):
            state_dict = state_dict["ema_state_dict"]
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        elif "state_dict" in state_dict:
            if "gen_model" in state_dict["state_dict"]:
                state_dict = state_dict["state_dict"]["gen_model"]
            else:
                state_dict = state_dict["state_dict"]
        save_as_pt(state_dict, cfg.target_path)
