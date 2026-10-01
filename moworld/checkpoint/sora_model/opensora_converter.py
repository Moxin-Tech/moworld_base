from safetensors.torch import load_file
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.utils import check_method_support
from checkpoint.sora_model.convert_utils.save_load_utils import save_as_mm


class OpenSoraConverter(SoraModelConverter):
    """Converter for OpenSora"""

    _supported_methods = ["hf_to_mm", "layerzero_to_mm"]
    _enable_tp = False
    _enable_pp = False
    _enable_vpp = False

    hf_to_mm_convert_mapping = {
        "time_in.in_layer.bias": "time_in.mlp.0.bias",
        "time_in.in_layer.weight": "time_in.mlp.0.weight",
        "time_in.out_layer.bias": "time_in.mlp.2.bias",
        "time_in.out_layer.weight": "time_in.mlp.2.weight",
        "vector_in.in_layer.bias": "vector_in.fc1.bias",
        "vector_in.in_layer.weight": "vector_in.fc1.weight",
        "vector_in.out_layer.bias": "vector_in.fc2.bias",
        "vector_in.out_layer.weight": "vector_in.fc2.weight",
    }

    def __init__(self) -> None:
        super().__init__()
        double_stream_layers = 19
        single_stream_layers = 38

        for i in range(double_stream_layers):
            self.hf_to_mm_convert_mapping.update({
                f"double_blocks.{i}.img_mod.lin.bias": f"double_blocks.{i}.img_mod.linear.bias",
                f"double_blocks.{i}.img_mod.lin.weight": f"double_blocks.{i}.img_mod.linear.weight",
                f"double_blocks.{i}.img_attn.q_proj.bias": f"double_blocks.{i}.img_attn.proj_q.bias",
                f"double_blocks.{i}.img_attn.q_proj.weight": f"double_blocks.{i}.img_attn.proj_q.weight",
                f"double_blocks.{i}.img_attn.k_proj.bias": f"double_blocks.{i}.img_attn.proj_k.bias",
                f"double_blocks.{i}.img_attn.k_proj.weight": f"double_blocks.{i}.img_attn.proj_k.weight",
                f"double_blocks.{i}.img_attn.v_proj.bias": f"double_blocks.{i}.img_attn.proj_v.bias",
                f"double_blocks.{i}.img_attn.v_proj.weight": f"double_blocks.{i}.img_attn.proj_v.weight",
                f"double_blocks.{i}.img_attn.proj.bias": f"double_blocks.{i}.img_attn.proj_out.bias",
                f"double_blocks.{i}.img_attn.proj.weight": f"double_blocks.{i}.img_attn.proj_out.weight",
                f"double_blocks.{i}.img_attn.norm.query_norm.scale": f"double_blocks.{i}.img_attn.q_norm.weight",
                f"double_blocks.{i}.img_attn.norm.key_norm.scale": f"double_blocks.{i}.img_attn.k_norm.weight",
                f"double_blocks.{i}.img_mlp.0.bias": f"double_blocks.{i}.img_mlp.fc1.bias",
                f"double_blocks.{i}.img_mlp.0.weight": f"double_blocks.{i}.img_mlp.fc1.weight",
                f"double_blocks.{i}.img_mlp.2.bias": f"double_blocks.{i}.img_mlp.fc2.bias",
                f"double_blocks.{i}.img_mlp.2.weight": f"double_blocks.{i}.img_mlp.fc2.weight",
                f"double_blocks.{i}.txt_mod.lin.bias": f"double_blocks.{i}.txt_mod.linear.bias",
                f"double_blocks.{i}.txt_mod.lin.weight": f"double_blocks.{i}.txt_mod.linear.weight",
                f"double_blocks.{i}.txt_attn.q_proj.bias": f"double_blocks.{i}.txt_attn.proj_q.bias",
                f"double_blocks.{i}.txt_attn.q_proj.weight": f"double_blocks.{i}.txt_attn.proj_q.weight",
                f"double_blocks.{i}.txt_attn.k_proj.bias": f"double_blocks.{i}.txt_attn.proj_k.bias",
                f"double_blocks.{i}.txt_attn.k_proj.weight": f"double_blocks.{i}.txt_attn.proj_k.weight",
                f"double_blocks.{i}.txt_attn.v_proj.bias": f"double_blocks.{i}.txt_attn.proj_v.bias",
                f"double_blocks.{i}.txt_attn.v_proj.weight": f"double_blocks.{i}.txt_attn.proj_v.weight",
                f"double_blocks.{i}.txt_attn.proj.bias": f"double_blocks.{i}.txt_attn.proj_out.bias",
                f"double_blocks.{i}.txt_attn.proj.weight": f"double_blocks.{i}.txt_attn.proj_out.weight",
                f"double_blocks.{i}.txt_attn.norm.query_norm.scale": f"double_blocks.{i}.txt_attn.q_norm.weight",
                f"double_blocks.{i}.txt_attn.norm.key_norm.scale": f"double_blocks.{i}.txt_attn.k_norm.weight",
                f"double_blocks.{i}.txt_mlp.0.bias": f"double_blocks.{i}.txt_mlp.fc1.bias",
                f"double_blocks.{i}.txt_mlp.0.weight": f"double_blocks.{i}.txt_mlp.fc1.weight",
                f"double_blocks.{i}.txt_mlp.2.bias": f"double_blocks.{i}.txt_mlp.fc2.bias",
                f"double_blocks.{i}.txt_mlp.2.weight": f"double_blocks.{i}.txt_mlp.fc2.weight"
            })

        for i in range(single_stream_layers):
            self.hf_to_mm_convert_mapping.update({
                f"single_blocks.{i}.norm.query_norm.scale": f"single_blocks.{i}.q_norm.weight",
                f"single_blocks.{i}.norm.key_norm.scale": f"single_blocks.{i}.k_norm.weight",
                f"single_blocks.{i}.modulation.lin.bias": f"single_blocks.{i}.modulation.linear.bias",
                f"single_blocks.{i}.modulation.lin.weight": f"single_blocks.{i}.modulation.linear.weight"
            })

    @check_method_support
    def hf_to_mm(self, cfg: ConvertConfig):
        state_dict = load_file(cfg.source_path)
        state_dict = self._replace_state_dict(
            state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping
        )
        state_dicts = self._mm_split(state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, state_dicts)
