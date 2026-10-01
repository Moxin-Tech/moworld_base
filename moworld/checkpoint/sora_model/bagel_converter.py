from safetensors.torch import load_file
from checkpoint.sora_model.sora_model_converter import SoraModelConverter
from checkpoint.sora_model.convert_utils.cfg import ConvertConfig
from checkpoint.sora_model.convert_utils.utils import check_method_support
from checkpoint.sora_model.convert_utils.save_load_utils import save_as_mm


class BagelConverter(SoraModelConverter):
    """Converter for Bagel"""

    _supported_methods = ["hf_to_mm"]
    _enable_tp = False
    _enable_pp = False
    _enable_vpp = False

    hf_to_mm_convert_mapping = {
        "connector.fc1.bias": "image_encoder.projector.linear_fc1.bias",
        "connector.fc1.weight": "image_encoder.projector.linear_fc1.weight",
        "connector.fc2.bias": "image_encoder.projector.linear_fc2.bias",
        "connector.fc2.weight": "image_encoder.projector.linear_fc2.weight",
        "language_model.lm_head.weight": "mllm.lm_head.weight",
        "language_model.model.embed_tokens.weight": "mllm.decoder.embed_tokens.weight",
        "language_model.model.norm.weight": "mllm.decoder.norm.weight",
        "language_model.model.norm_moe_gen.weight": "mllm.decoder.norm_moe_gen.weight",
        "latent_pos_embed.pos_embed": "latent_pos_embed.pos_embed",
        "llm2vae.bias": "mllm.llm2vae.bias",
        "llm2vae.weight": "mllm.llm2vae.weight",
        "time_embedder.mlp.0.bias": "mllm.time_embedder.time_embed.0.bias",
        "time_embedder.mlp.0.weight": "mllm.time_embedder.time_embed.0.weight",
        "time_embedder.mlp.2.bias": "mllm.time_embedder.time_embed.2.bias",
        "time_embedder.mlp.2.weight": "mllm.time_embedder.time_embed.2.weight",
        "vae2llm.bias": "mllm.vae2llm.bias",
        "vae2llm.weight": "mllm.vae2llm.weight",
        "vit_model.vision_model.embeddings.patch_embedding.bias": "image_encoder.encoder.patch_embed.patch_embedding.bias",
        "vit_model.vision_model.embeddings.patch_embedding.weight": "image_encoder.encoder.patch_embed.patch_embedding.weight",
        "vit_model.vision_model.embeddings.position_embedding.weight": "image_encoder.encoder.patch_embed.position_embedding.weight",
        "vit_model.vision_model.post_layernorm.bias": "image_encoder.encoder.norm.bias",
        "vit_model.vision_model.post_layernorm.weight": "image_encoder.encoder.norm.weight",
        "vit_pos_embed.pos_embed": "mllm.vit_pos_embed.pos_embed"
    }

    def __init__(self) -> None:
        super().__init__()
        llm_layers = 28
        image_encoder_layers = 26

        for i in range(llm_layers):
            self.hf_to_mm_convert_mapping.update({
                f"language_model.model.layers.{i}.input_layernorm.weight": f"mllm.decoder.layers.{i}.input_layernorm.weight",
                f"language_model.model.layers.{i}.input_layernorm_moe_gen.weight": f"mllm.decoder.layers.{i}.input_layernorm_moe_gen.weight",
                f"language_model.model.layers.{i}.mlp.down_proj.weight": f"mllm.decoder.layers.{i}.mlp.down_proj.weight",
                f"language_model.model.layers.{i}.mlp.gate_proj.weight": f"mllm.decoder.layers.{i}.mlp.gate_proj.weight",
                f"language_model.model.layers.{i}.mlp.up_proj.weight": f"mllm.decoder.layers.{i}.mlp.up_proj.weight",
                f"language_model.model.layers.{i}.mlp_moe_gen.down_proj.weight": f"mllm.decoder.layers.{i}.mlp_moe_gen.down_proj.weight",
                f"language_model.model.layers.{i}.mlp_moe_gen.gate_proj.weight": f"mllm.decoder.layers.{i}.mlp_moe_gen.gate_proj.weight",
                f"language_model.model.layers.{i}.mlp_moe_gen.up_proj.weight": f"mllm.decoder.layers.{i}.mlp_moe_gen.up_proj.weight",
                f"language_model.model.layers.{i}.post_attention_layernorm.weight": f"mllm.decoder.layers.{i}.post_attention_layernorm.weight",
                f"language_model.model.layers.{i}.post_attention_layernorm_moe_gen.weight": f"mllm.decoder.layers.{i}.post_attention_layernorm_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.k_norm.weight": f"mllm.decoder.layers.{i}.self_attn.k_norm.weight",
                f"language_model.model.layers.{i}.self_attn.k_norm_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.k_norm_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.k_proj.bias": f"mllm.decoder.layers.{i}.self_attn.k_proj.bias",
                f"language_model.model.layers.{i}.self_attn.k_proj.weight": f"mllm.decoder.layers.{i}.self_attn.k_proj.weight",
                f"language_model.model.layers.{i}.self_attn.k_proj_moe_gen.bias": f"mllm.decoder.layers.{i}.self_attn.k_proj_moe_gen.bias",
                f"language_model.model.layers.{i}.self_attn.k_proj_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.k_proj_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.o_proj.weight": f"mllm.decoder.layers.{i}.self_attn.o_proj.weight",
                f"language_model.model.layers.{i}.self_attn.o_proj_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.o_proj_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.q_norm.weight": f"mllm.decoder.layers.{i}.self_attn.q_norm.weight",
                f"language_model.model.layers.{i}.self_attn.q_norm_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.q_norm_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.q_proj.bias": f"mllm.decoder.layers.{i}.self_attn.q_proj.bias",
                f"language_model.model.layers.{i}.self_attn.q_proj.weight": f"mllm.decoder.layers.{i}.self_attn.q_proj.weight",
                f"language_model.model.layers.{i}.self_attn.q_proj_moe_gen.bias": f"mllm.decoder.layers.{i}.self_attn.q_proj_moe_gen.bias",
                f"language_model.model.layers.{i}.self_attn.q_proj_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.q_proj_moe_gen.weight",
                f"language_model.model.layers.{i}.self_attn.v_proj.bias": f"mllm.decoder.layers.{i}.self_attn.v_proj.bias",
                f"language_model.model.layers.{i}.self_attn.v_proj.weight": f"mllm.decoder.layers.{i}.self_attn.v_proj.weight",
                f"language_model.model.layers.{i}.self_attn.v_proj_moe_gen.bias": f"mllm.decoder.layers.{i}.self_attn.v_proj_moe_gen.bias",
                f"language_model.model.layers.{i}.self_attn.v_proj_moe_gen.weight": f"mllm.decoder.layers.{i}.self_attn.v_proj_moe_gen.weight"
            })

        for i in range(image_encoder_layers):
            self.hf_to_mm_convert_mapping.update({
                f"vit_model.vision_model.encoder.layers.{i}.layer_norm1.bias": f"image_encoder.encoder.blocks.{i}.norm1.bias",
                f"vit_model.vision_model.encoder.layers.{i}.layer_norm1.weight": f"image_encoder.encoder.blocks.{i}.norm1.weight",
                f"vit_model.vision_model.encoder.layers.{i}.layer_norm2.bias": f"image_encoder.encoder.blocks.{i}.norm2.bias",
                f"vit_model.vision_model.encoder.layers.{i}.layer_norm2.weight": f"image_encoder.encoder.blocks.{i}.norm2.weight",
                f"vit_model.vision_model.encoder.layers.{i}.mlp.fc1.bias": f"image_encoder.encoder.blocks.{i}.mlp.fc1.bias",
                f"vit_model.vision_model.encoder.layers.{i}.mlp.fc1.weight": f"image_encoder.encoder.blocks.{i}.mlp.fc1.weight",
                f"vit_model.vision_model.encoder.layers.{i}.mlp.fc2.bias": f"image_encoder.encoder.blocks.{i}.mlp.fc2.bias",
                f"vit_model.vision_model.encoder.layers.{i}.mlp.fc2.weight": f"image_encoder.encoder.blocks.{i}.mlp.fc2.weight",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.k_proj.bias": f"image_encoder.encoder.blocks.{i}.attn.k_proj.bias",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.k_proj.weight": f"image_encoder.encoder.blocks.{i}.attn.k_proj.weight",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.out_proj.bias": f"image_encoder.encoder.blocks.{i}.attn.out_proj.bias",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.out_proj.weight": f"image_encoder.encoder.blocks.{i}.attn.out_proj.weight",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.q_proj.bias": f"image_encoder.encoder.blocks.{i}.attn.q_proj.bias",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.q_proj.weight": f"image_encoder.encoder.blocks.{i}.attn.q_proj.weight",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.v_proj.bias": f"image_encoder.encoder.blocks.{i}.attn.v_proj.bias",
                f"vit_model.vision_model.encoder.layers.{i}.self_attn.v_proj.weight": f"image_encoder.encoder.blocks.{i}.attn.v_proj.weight",
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
