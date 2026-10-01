from checkpoint.sora_model.sora_model_converter import SoraModelConverter


class LayerIndexConverter:
    @staticmethod
    def get_layer_index(name):
        if name.startswith("blocks"):
            idx = int(name.split('.')[1])
            return idx
        return None

    @staticmethod
    def convert_layer_index(name, new_layer_index):
        if name.startswith("blocks"):
            parts = name.split('.')
            parts[1] = str(new_layer_index)
            return ".".join(parts)
        return name


class WanConverter(SoraModelConverter):
    """Converter for Wan2.1"""

    _supported_methods = ["hf_to_mm", "resplit", "mm_to_hf", "lora_hf_to_mm", "layerzero_to_mm", "merge_lora_to_base", "mm_to_dcp"]
    _enable_tp = False
    _enable_pp = True
    _enable_vpp = True

    hf_to_mm_convert_mapping = {
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

    hf_to_mm_str_replace_mapping = {
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

    lora_hf_to_mm_str_replace_mapping = {
        "self_attn.q.": "self_attn.proj_q.",
        "self_attn.k.": "self_attn.proj_k.",
        "self_attn.v.": "self_attn.proj_v.",
        "self_attn.o.": "self_attn.proj_out.",
        "cross_attn.q.": "cross_attn.proj_q.",
        "cross_attn.k.": "cross_attn.proj_k.",
        "cross_attn.v.": "cross_attn.proj_v.",
        "cross_attn.o.": "cross_attn.proj_out.",
        "self_attn.norm_q": "self_attn.q_norm",
        "self_attn.norm_k": "self_attn.k_norm",
        "cross_attn.norm_q": "cross_attn.q_norm",
        "cross_attn.norm_k": "cross_attn.k_norm",
        "text_embedding.0.": "text_embedding.linear_1.",
        "text_embedding.2.": "text_embedding.linear_2.",
    }

    lora_target_modules = ["proj_q", "proj_k", "proj_v", "proj_out", "ffn.0", "ffn.2"]

    pre_process_weight_names = [
        "patch_embedding.weight", "patch_embedding.bias",
        "text_embedding.linear_1.weight", "text_embedding.linear_1.bias",
        "text_embedding.linear_2.weight", "text_embedding.linear_2.bias",
        "time_embedding.0.weight", "time_embedding.0.bias",
        "time_embedding.2.weight", "time_embedding.2.bias",
        "time_projection.1.weight", "time_projection.1.bias",
        "img_emb.proj.1.weight", "img_emb.proj.1.bias",
        "img_emb.proj.3.weight", "img_emb.proj.3.bias",
        "img_emb.emb_pos"
    ] # pre_process layers for pp
    post_preprocess_weight_names = ['head.head.weight', 'head.head.bias', 'head.modulation'] # post_process layers for pp
    layer_index_converter = LayerIndexConverter()
