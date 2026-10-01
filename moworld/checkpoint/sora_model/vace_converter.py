import torch
import safetensors
from safetensors.torch import load_file
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig, ParallelConfig
from checkpoint.sora_model.convert_utils.utils import check_method_support, flip_mapping
from checkpoint.sora_model.convert_utils.save_load_utils import save_as_mm, load_from_hf, load_from_mm, save_as_pt


class VACEConverter(SoraModelConverter):
    """Converter for VACE"""

    _supported_methods = ["hf_to_mm", "mm_to_hf", "hf_diffusers_to_mm", "mm_to_hf_diffusers"]
    _enable_tp = False
    _enable_pp = True
    _enable_vpp = True

    def __init__(self) -> None:
        super().__init__()
        self.hf_to_mm_convert_mapping = {
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.linear_1.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.linear_1.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.linear_2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.linear_2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
            "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
            "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
            "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
            "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
            "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
            "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
            "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
            "condition_embedder.image_embedder.pos_embed": "img_emb.emb_pos",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }

        self.hf_to_mm_str_replace_mapping_wan = {
            "attn1.norm_q": "self_attn.q_norm",
            "attn1.norm_k": "self_attn.k_norm",
            "attn2.norm_q": "cross_attn.q_norm",
            "attn2.norm_k": "cross_attn.k_norm",
            "attn1.to_q.": "self_attn.proj_q.",
            "attn1.to_k.": "self_attn.proj_k.",
            "attn1.to_v.": "self_attn.proj_v.",
            "attn1.to_out.0.": "self_attn.proj_out.",
            "attn2.to_q.": "cross_attn.proj_q.",
            "attn2.to_k.": "cross_attn.proj_k.",
            "attn2.to_v.": "cross_attn.proj_v.",
            "attn2.add_k_proj": "cross_attn.k_img",
            "attn2.add_v_proj": "cross_attn.v_img",
            "attn2.norm_added_k": "cross_attn.k_norm_img",
            "attn2.to_out.0.": "cross_attn.proj_out.",
            ".ffn.net.0.proj.": ".ffn.0.",
            ".ffn.net.2.": ".ffn.2.",
            "scale_shift_table": "modulation",
            ".norm2.": ".norm3."
        }

        self.hf_to_mm_str_replace_mapping_vace = {
            ".proj_in.": ".before_proj.",
            ".proj_out.": ".after_proj.",
            "attn1.norm_q": "wan_dit_block.self_attn.q_norm",
            "attn1.norm_k": "wan_dit_block.self_attn.k_norm",
            "attn2.norm_q": "wan_dit_block.cross_attn.q_norm",
            "attn2.norm_k": "wan_dit_block.cross_attn.k_norm",
            "attn1.to_q.": "wan_dit_block.self_attn.proj_q.",
            "attn1.to_k.": "wan_dit_block.self_attn.proj_k.",
            "attn1.to_v.": "wan_dit_block.self_attn.proj_v.",
            "attn1.to_out.0.": "wan_dit_block.self_attn.proj_out.",
            "attn2.to_q.": "wan_dit_block.cross_attn.proj_q.",
            "attn2.to_k.": "wan_dit_block.cross_attn.proj_k.",
            "attn2.to_v.": "wan_dit_block.cross_attn.proj_v.",
            "attn2.add_k_proj": "wan_dit_block.cross_attn.k_img",
            "attn2.add_v_proj": "wan_dit_block.cross_attn.v_img",
            "attn2.norm_added_k": "wan_dit_block.cross_attn.k_norm_img",
            "attn2.to_out.0.": "wan_dit_block.cross_attn.proj_out.",
            ".ffn.net.0.proj.": ".wan_dit_block.ffn.0.",
            ".ffn.net.2.": ".wan_dit_block.ffn.2.",
            "scale_shift_table": "wan_dit_block.modulation",
            ".norm2.": ".wan_dit_block.norm3."
        }

        self.hf_civitai_to_diffusers_convert_mapping = {
            "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
            "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
            "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
            "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
            "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
            "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
            "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
            "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
            "time_projection.1.bias": "condition_embedder.time_proj.bias",
            "time_projection.1.weight": "condition_embedder.time_proj.weight",
            "img_emb.proj.1.bias": "condition_embedder.image_embedder.ff.net.0.proj.bias",
            "img_emb.proj.1.weight": "condition_embedder.image_embedder.ff.net.0.proj.weight",
            "img_emb.proj.3.bias": "condition_embedder.image_embedder.ff.net.2.bias",
            "img_emb.proj.3.weight": "condition_embedder.image_embedder.ff.net.2.weight",
            "img_emb.proj.0.bias": "condition_embedder.image_embedder.norm1.bias",
            "img_emb.proj.0.weight": "condition_embedder.image_embedder.norm1.weight",
            "img_emb.proj.4.bias": "condition_embedder.image_embedder.norm2.bias",
            "img_emb.proj.4.weight": "condition_embedder.image_embedder.norm2.weight",
            "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
            "head.modulation": "scale_shift_table",
            "head.head.bias": "proj_out.bias",
            "head.head.weight": "proj_out.weight"
        }

        self.hf_civitai_to_diffusers_replace_mapping = {
            ".cross_attn.k.": ".attn2.to_k.",
            ".cross_attn.norm_k.weight": ".attn2.norm_k.weight",
            ".cross_attn.norm_q.weight": ".attn2.norm_q.weight",
            ".cross_attn.o.": ".attn2.to_out.0.",
            ".cross_attn.q.": ".attn2.to_q.",
            ".cross_attn.v.": ".attn2.to_v.",
            ".ffn.0.": ".ffn.net.0.proj.",
            ".ffn.2.": ".ffn.net.2.",
            ".modulation": ".scale_shift_table",
            ".norm3.": ".norm2.",
            ".self_attn.k.": ".attn1.to_k.",
            ".self_attn.norm_k.weight": ".attn1.norm_k.weight",
            ".self_attn.norm_q.weight": ".attn1.norm_q.weight",
            ".self_attn.o.": ".attn1.to_out.0.",
            ".self_attn.q.": ".attn1.to_q.",
            ".self_attn.v.": ".attn1.to_v.",
            ".after_proj.": ".proj_out.",
            ".before_proj.": ".proj_in.",
            ".cross_attn.k_img.": ".attn2.add_k_proj.",
            ".cross_attn.v_img.": ".attn2.add_v_proj.",
            ".cross_attn.norm_k_img.weight": ".attn2.norm_added_k.weight",
        }

    @check_method_support
    def hf_diffusers_to_mm(self, cfg: ConvertConfig):
        state_dict = load_from_hf(cfg.source_path)
        self._hf_to_mm_state(cfg, state_dict)

    @check_method_support
    def mm_to_hf_diffusers(self, cfg: ConvertConfig):
        state_dict = load_from_mm(cfg.source_path)
        state_dict = self._mm_merge(state_dict)
        state_dict = self._mm_to_hf_state(cfg, state_dict)
        save_as_pt(state_dict, cfg.target_path)

    @check_method_support
    def hf_to_mm(self, cfg: ConvertConfig):
        state_dict = load_from_hf(cfg.source_path)
        # convert hf(not diffusers) civitai to diffusers
        state_dict = self._replace_state_dict(
            state_dict,
            self.hf_civitai_to_diffusers_convert_mapping,
            self.hf_civitai_to_diffusers_replace_mapping
        )
        # convert diffusers to mm
        self._hf_to_mm_state(cfg, state_dict)

    @check_method_support
    def mm_to_hf(self, cfg: ConvertConfig):
        state_dict = load_from_mm(cfg.source_path)
        state_dict = self._mm_merge(state_dict)
        # convert mm to diffusers
        state_dict = self._mm_to_hf_state(cfg, state_dict)
        # convert diffusers to hf(not diffusers)
        state_dict = self._replace_state_dict(
            state_dict,
            flip_mapping(self.hf_civitai_to_diffusers_convert_mapping),
            flip_mapping(self.hf_civitai_to_diffusers_replace_mapping)
        )
        save_as_pt(state_dict, cfg.target_path)

    def _hf_to_mm_state(self, cfg: ConvertConfig, state_dict: None):
        vace_state_dict = {key: state_dict[key] for key in state_dict if "vace" in key}
        wan_state_dict = {key: state_dict[key] for key in state_dict if "vace" not in key}
        vace_state_dict = self._replace_state_dict(
            vace_state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping_vace
        )
        wan_state_dict = self._replace_state_dict(
            wan_state_dict,
            self.hf_to_mm_convert_mapping,
            self.hf_to_mm_str_replace_mapping_wan
        )
        vace_state_dict = {f"vace_dit.{key}": vace_state_dict[key] for key in vace_state_dict}
        wan_state_dict = {f"wan_dit.{key}": wan_state_dict[key] for key in wan_state_dict}
        new_state_dict = {**vace_state_dict, **wan_state_dict}
        new_state_dict = self._mm_split(new_state_dict, cfg.target_parallel_config)
        save_as_mm(cfg.target_path, new_state_dict)

    def _mm_to_hf_state(self, cfg: ConvertConfig, state_dict: None):
        vace_state_dict = {key[9:]: state_dict[key] for key in state_dict if "vace_dit." in key}
        wan_state_dict = {key[8:]: state_dict[key] for key in state_dict if "wan_dit." in key}
        vace_state_dict = self._replace_state_dict(
            vace_state_dict,
            flip_mapping(self.hf_to_mm_convert_mapping),
            flip_mapping(self.hf_to_mm_str_replace_mapping_vace)
        )
        wan_state_dict = self._replace_state_dict(
            wan_state_dict,
            flip_mapping(self.hf_to_mm_convert_mapping),
            flip_mapping(self.hf_to_mm_str_replace_mapping_wan)
        )
        state_dict = {**wan_state_dict, **vace_state_dict}
        return state_dict
