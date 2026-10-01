from typing_extensions import Literal
import torch
from safetensors.torch import load_file
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.tp_patterns import TPPattern
from checkpoint.sora_model.convert_utils.save_load_utils import load_pt, save_as_mm
from checkpoint.sora_model.convert_utils.utils import check_method_support


class XMLPFusedRowParallelTP(TPPattern):
    def __init__(self, hidden_size=3096) -> None:
        self.hidden_size = hidden_size

    def split(self, weight, tp_size):
        hidden_size = self.hidden_size
        wx = weight[:, :hidden_size]
        wmlp = weight[:, hidden_size:]
        wxs = torch.chunk(wx, tp_size, dim=1)
        wmlps = torch.chunk(wmlp, tp_size, dim=1)
        weights = [torch.cat([wxs[i], wmlps[i]], dim=1) for i in range(tp_size)]
        return weights

    def merge(self, weights):
        hidden_size = self.hidden_size // len(weights)
        wxs = [weight[:, :hidden_size] for weight in weights]
        wmlps = [weight[:, hidden_size:] for weight in weights]
        weight = torch.cat([
            torch.cat(wxs, dim=1),
            torch.cat(wmlps, dim=1)
        ], dim=1)
        return weight


class HunyuanVideoConverter(SoraModelConverter):
    """Converter for HunyuanVideo"""

    _supported_methods = ["resplit", "layerzero_to_mm", "merge_lora_to_base"]
    _enable_tp = True

    convert_mapping = {
        "txt_in.t_embedder.mlp.0.weight": "txt_in.t_embedder.time_embed.0.weight",
        "txt_in.t_embedder.mlp.0.bias": "txt_in.t_embedder.time_embed.0.bias",
        "txt_in.t_embedder.mlp.2.weight": "txt_in.t_embedder.time_embed.2.weight",
        "txt_in.t_embedder.mlp.2.bias": "txt_in.t_embedder.time_embed.2.bias",
        "time_in.mlp.0.weight": "time_in.time_embed.0.weight",
        "time_in.mlp.0.bias": "time_in.time_embed.0.bias",
        "time_in.mlp.2.weight": "time_in.time_embed.2.weight",
        "time_in.mlp.2.bias": "time_in.time_embed.2.bias",
        "vector_in.in_layer.weight": "vector_in.fc1.weight",
        "vector_in.in_layer.bias": "vector_in.fc1.bias",
        "vector_in.out_layer.weight": "vector_in.fc2.weight",
        "vector_in.out_layer.bias": "vector_in.fc2.bias",
        "guidance_in.mlp.0.weight": "guidance_in.time_embed.0.weight",
        "guidance_in.mlp.0.bias": "guidance_in.time_embed.0.bias",
        "guidance_in.mlp.2.weight": "guidance_in.time_embed.2.weight",
        "guidance_in.mlp.2.bias": "guidance_in.time_embed.2.bias",
        "final_layer.linear.weight": "proj_out.weight",
        "final_layer.linear.bias": "proj_out.bias",
        "final_layer.adaLN_modulation.1.weight": "adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias": "adaLN_modulation.1.bias"
    }

    tp_split_mapping = {
        "column_parallel_tp": [
            "vector_in.fc1.weight",
            "vector_in.fc1.bias",
            "proj_out.weight",
            "proj_out.bias",
            "adaLN_modulation.1.weight",
            "adaLN_modulation.1.bias",
        ],
        "row_parallel_tp": [
            "vector_in.fc2.weight"
        ],
        "qkv_fused_column_tp": []
    }

    spec_tp_split_mapping = {}

    lora_target_modules = [
        "linear", "fc1", "fc2", "img_attn_qkv", "img_attn_proj", "txt_attn_qkv", "txt_attn_proj", "linear1_qkv", "linear1_mlp", "linear2", "proj_out"
    ]

    def __init__(self, version: Literal["t2v", "i2v", "t2v-lora", "i2v-lora"] = "t2v") -> None:
        self.version = version
        self.double_stream_layers = 20
        self.single_stream_layers = 40
        self.num_heads = 24
        self.head_dim = 128

        x_mlp_fused_row_parallel_tp_pattern = XMLPFusedRowParallelTP(hidden_size=self.num_heads * self.head_dim)

        self.spec_tp_split_mapping = {x_mlp_fused_row_parallel_tp_pattern: []}

        for index in range(self.double_stream_layers):
            self.tp_split_mapping["column_parallel_tp"] += [
                f"double_blocks.{index}.img_mod.linear.weight",
                f"double_blocks.{index}.img_mod.linear.bias",
                f"double_blocks.{index}.img_mlp.fc1.weight",
                f"double_blocks.{index}.img_mlp.fc1.bias",
                f"double_blocks.{index}.txt_mod.linear.weight",
                f"double_blocks.{index}.txt_mod.linear.bias",
                f"double_blocks.{index}.txt_mlp.fc1.weight",
                f"double_blocks.{index}.txt_mlp.fc1.bias",
            ]
            self.tp_split_mapping["row_parallel_tp"] += [
                f"double_blocks.{index}.img_attn_proj.weight",
                f"double_blocks.{index}.img_mlp.fc2.weight",
                f"double_blocks.{index}.txt_attn_proj.weight",
                f"double_blocks.{index}.txt_mlp.fc2.weight",
            ]
            self.tp_split_mapping["qkv_fused_column_tp"] += [
                f"double_blocks.{index}.img_attn_qkv.weight",
                f"double_blocks.{index}.img_attn_qkv.bias",
                f"double_blocks.{index}.txt_attn_qkv.weight",
                f"double_blocks.{index}.txt_attn_qkv.bias",
            ]

        for index in range(self.single_stream_layers):
            self.tp_split_mapping["column_parallel_tp"] += [
                f"single_blocks.{index}.modulation.linear.weight",
                f"single_blocks.{index}.modulation.linear.bias",
                f"single_blocks.{index}.linear1_mlp.weight",
                f"single_blocks.{index}.linear1_mlp.bias"
            ]

            self.tp_split_mapping["qkv_fused_column_tp"] += [
                f"single_blocks.{index}.linear1_qkv.weight",
                f"single_blocks.{index}.linear1_qkv.bias"
            ]

            self.spec_tp_split_mapping[x_mlp_fused_row_parallel_tp_pattern] += [
                f"single_blocks.{index}.linear2.weight"
            ]

        if self.version == "t2v":
            self._supported_methods = ["resplit", "layerzero_to_mm", "merge_lora_to_base", "source_to_mm"]
            self._enable_tp = True
            self._enable_pp = False
            self._enable_vpp = False

        elif self.version == "i2v":
            self._supported_methods = ["resplit", "layerzero_to_mm", "merge_lora_to_base", "source_to_mm"]
            self._enable_tp = False
            self._enable_pp = False
            self._enable_vpp = False

        elif self.version == "t2v-lora":
            self._supported_methods = []
            self._enable_tp = False
            self._enable_pp = False
            self._enable_vpp = False

        elif self.version == "i2v-lora":
            self._supported_methods = ["source_to_mm"]
            self._enable_tp = False
            self._enable_pp = False
            self._enable_vpp = False

            i2v_source_lora_prefix = "Hunyuan_video_I2V"
            self.str_replace_mapping = {
                f"{i2v_source_lora_prefix}_lora_": "",
                "single_blocks_": "single_blocks.",
                "double_blocks_": "double_blocks.",
                "_img_attn_proj": ".img_attn_proj",
                "_img_attn_qkv": ".img_attn_qkv",
                "_img_mlp_fc": ".img_mlp.fc",
                "_txt_mlp_fc": ".txt_mlp.fc",
                "_img_mod": ".img_mod",
                "_txt": ".txt",
                "_modulation": ".modulation",
                "_linear": ".linear",
                "lora_down": "lora_A.default",
                "lora_up": "lora_B.default",
                "_individual_token_refiner_blocks_": ".individual_token_refiner.blocks.",
                "_mlp_fc": ".mlp.fc",
                "vector_in_in_layer": "vector_in.fc1",
                "vector_in_out_layer": "vector_in.fc2",
                "final_layer.linear": "proj_out",
            }

    def t2v_text_encoder(self, cfg: ConvertConfig):
        """convert text encooder for t2v task"""
        if self.version != "t2v":
            raise RuntimeError(f"task: {self.version} do not need to convert text encoder")

        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration
        )
        processor = AutoProcessor.from_pretrained(cfg.source_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            cfg.source_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        model.language_model.save_pretrained(cfg.target_path)
        processor.tokenizer.save_pretrained(cfg.target_path)

    @check_method_support
    def source_to_mm(self, cfg: ConvertConfig):
        """convert ckpt from source code to megatron format"""
        if cfg.source_path.endswith("safetensors"):
            state_dict = load_file(cfg.source_path)
        else:
            state_dict = load_pt(cfg.source_path, module_name='module')
        state_dict = self._replace_state_dict(state_dict, self.convert_mapping, self.str_replace_mapping)
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)

    def _replace_state_dict(self, state_dict: dict, convert_mapping: dict = None, str_replace_mapping: dict = None):
        if self.version.endswith("lora"):
            names = list(state_dict.keys())
            for name in names:
                if ".alpha" in name:
                    state_dict.pop(name)
        state_dict = super()._replace_state_dict(state_dict, convert_mapping, str_replace_mapping)
        state_dict = self._split_qkv_mlp_fused_column_linear(state_dict)
        return state_dict

    def _split_qkv_mlp_fused_column_linear(self, state_dict: dict):
        """split qkv_mlp fused linear in single stream blocks int qkv part and mlp part"""
        hidden_size = self.num_heads * self.head_dim
        if self.version == "i2v-lora":
            names = list(state_dict.keys())
            for name in names:
                if "linear1" in name:
                    if ".lora_A" in name:
                        lora_a = state_dict.pop(name)
                        state_dict[name.replace("linear1", "linear1_qkv")] = lora_a
                        state_dict[name.replace("linear1", "linear1_mlp")] = lora_a
                    else:
                        lora_b = state_dict.pop(name)
                        w_qkv = lora_b[:hidden_size * 3]
                        w_mlp = lora_b[hidden_size * 3:]
                        state_dict[name.replace("linear1", "linear1_qkv")] = w_qkv
                        state_dict[name.replace("linear1", "linear1_mlp")] = w_mlp

        elif "single_blocks.0.linear1.weight" in state_dict.keys():
            # source_to_mm
            for index in range(self.single_stream_layers):
                weight1 = state_dict.pop(f"single_blocks.{index}.linear1.weight")
                bias1 = state_dict.pop(f"single_blocks.{index}.linear1.bias")
                state_dict[f"single_blocks.{index}.linear1_qkv.weight"] = weight1[:hidden_size * 3]
                state_dict[f"single_blocks.{index}.linear1_mlp.weight"] = weight1[hidden_size * 3:]
                state_dict[f"single_blocks.{index}.linear1_qkv.bias"] = bias1[:hidden_size * 3]
                state_dict[f"single_blocks.{index}.linear1_mlp.bias"] = bias1[hidden_size * 3:]
        return state_dict
