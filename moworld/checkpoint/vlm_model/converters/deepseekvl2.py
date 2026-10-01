from typing import List, cast

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig
from checkpoint.vlm_model.hf_to_mm import PPStageSchema, text_schema, convert_hf_to_mm
from checkpoint.vlm_model.operator import (
    ExpertUpGateMergeOp, Operator, UpGateMergeOp, QKVDirectMergeOp, RenameOp, GLUSplit,
    ColSplit, RowSplit
)


def create_deepseek_vl_ops() -> List[Operator]:
    """deepseekvl权重转换逻辑"""
    ops = [
        RenameOp(
            (
                (r'vision.attn_pool.kv.bias', r'image_encoder.encoder.attn_pool.kv.bias'),
                (r'vision.attn_pool.kv.weight', r'image_encoder.encoder.attn_pool.kv.weight'),
                (r'vision.attn_pool.latent', r'image_encoder.encoder.attn_pool.latent'),
                (r'vision.attn_pool.mlp.fc1.bias', r'image_encoder.encoder.attn_pool.mlp.fc1.bias'),
                (r'vision.attn_pool.mlp.fc1.weight', r'image_encoder.encoder.attn_pool.mlp.fc1.weight'),
                (r'vision.attn_pool.mlp.fc2.bias', r'image_encoder.encoder.attn_pool.mlp.fc2.bias'),
                (r'vision.attn_pool.mlp.fc2.weight', r'image_encoder.encoder.attn_pool.mlp.fc2.weight'),
                (r'vision.attn_pool.norm.bias', r'image_encoder.encoder.attn_pool.norm.bias'),
                (r'vision.attn_pool.norm.weight', r'image_encoder.encoder.attn_pool.norm.weight'),
                (r'vision.attn_pool.proj.bias', r'image_encoder.encoder.attn_pool.proj.bias'),
                (r'vision.attn_pool.proj.weight', r'image_encoder.encoder.attn_pool.proj.weight'),
                (r'vision.attn_pool.q.bias', r'image_encoder.encoder.attn_pool.q.bias'),
                (r'vision.attn_pool.q.weight', r'image_encoder.encoder.attn_pool.q.weight'),
                (r'vision.blocks.(\d+).attn.proj.bias', r'image_encoder.encoder.blocks.(\d+).attn.proj.bias'),
                (r'vision.blocks.(\d+).attn.proj.weight', r'image_encoder.encoder.blocks.(\d+).attn.proj.weight'),
                (r'vision.blocks.(\d+).attn.qkv.bias', r'image_encoder.encoder.blocks.(\d+).attn.qkv.bias'),
                (r'vision.blocks.(\d+).attn.qkv.weight', r'image_encoder.encoder.blocks.(\d+).attn.qkv.weight'),
                (r'vision.blocks.(\d+).mlp.fc1.bias', r'image_encoder.encoder.blocks.(\d+).mlp.fc1.bias'),
                (r'vision.blocks.(\d+).mlp.fc1.weight', r'image_encoder.encoder.blocks.(\d+).mlp.fc1.weight'),
                (r'vision.blocks.(\d+).mlp.fc2.bias', r'image_encoder.encoder.blocks.(\d+).mlp.fc2.bias'),
                (r'vision.blocks.(\d+).mlp.fc2.weight', r'image_encoder.encoder.blocks.(\d+).mlp.fc2.weight'),
                (r'vision.blocks.(\d+).norm1.bias', r'image_encoder.encoder.blocks.(\d+).norm1.bias'),
                (r'vision.blocks.(\d+).norm1.weight', r'image_encoder.encoder.blocks.(\d+).norm1.weight'),
                (r'vision.blocks.(\d+).norm2.bias', r'image_encoder.encoder.blocks.(\d+).norm2.bias'),
                (r'vision.blocks.(\d+).norm2.weight', r'image_encoder.encoder.blocks.(\d+).norm2.weight'),
                (r'vision.norm.bias', r'image_encoder.encoder.norm.bias'),
                (r'vision.norm.weight', r'image_encoder.encoder.norm.weight'),
                (r'vision.patch_embed.proj.bias', r'image_encoder.encoder.patch_embed.proj.bias'),
                (r'vision.patch_embed.proj.weight', r'image_encoder.encoder.patch_embed.proj.weight'),
                (r'vision.pos_embed', r'image_encoder.encoder.pos_embed'),

                (r'projector.layers.(\d+).bias', r'image_encoder.projector.layers.(\d+).bias'),
                (r'projector.layers.(\d+).weight', r'image_encoder.projector.layers.(\d+).weight'),

                (r'language.lm_head.weight', r'text_decoder.output_layer.weight'),
                (r'language.model.embed_tokens.weight', r'text_decoder.embedding.word_embeddings.weight'),
                (r'language.model.layers.(\d+).input_layernorm.weight',
                 r'text_decoder.decoder.layers.(\d+).input_layernorm.weight'),
                (r'language.model.layers.(\d+).mlp.down_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight'),
                (r'language.model.layers.(\d+).mlp.experts.(\d+).down_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight'),
                (r'language.model.layers.(\d+).mlp.experts.(\d+).gate_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1_gate.weight'),
                (r'language.model.layers.(\d+).mlp.experts.(\d+).up_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1_up.weight'),
                (r'language.model.layers.(\d+).mlp.gate.e_score_correction_bias',
                 r'text_decoder.decoder.layers.(\d+).mlp.router.expert_bias'),
                (r'language.model.layers.(\d+).mlp.gate.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.router.weight'),
                (r'language.model.layers.(\d+).mlp.gate_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1_gate.weight'),
                (r'language.model.layers.(\d+).mlp.shared_experts.down_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc2.weight'),
                (r'language.model.layers.(\d+).mlp.shared_experts.gate_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1_gate.weight'),
                (r'language.model.layers.(\d+).mlp.shared_experts.up_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1_up.weight'),
                (r'language.model.layers.(\d+).mlp.up_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1_up.weight'),
                (r'language.model.layers.(\d+).post_attention_layernorm.weight',
                 r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm.weight'),
                (r'language.model.layers.(\d+).self_attn.kv_a_layernorm.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.k_layernorm.weight'),
                (r'language.model.layers.(\d+).self_attn.kv_a_proj_with_mqa.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.kv_a_proj_with_mqa.weight'),
                (r'language.model.layers.(\d+).self_attn.kv_b_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.weight'),
                (r'language.model.layers.(\d+).self_attn.o_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight'),
                (r'language.model.layers.(\d+).self_attn.q_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.weight'),
                (r'language.model.norm.weight', r'text_decoder.decoder.final_layernorm.weight'),
                # deepseekvl3
                (r'language.model.layers.(\d+).self_attn.q_a_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.weight'),
                (r'language.model.layers.(\d+).self_attn.q_b_proj.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_qb.weight'),
                (r'language.model.layers.(\d+).self_attn.q_a_layernorm.weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.q_layernorm.weight'),
            )
        ),

        UpGateMergeOp(raw_names=[r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1_gate.weight",
                                 r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1_up.weight"],
                      new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),

        UpGateMergeOp(raw_names=[r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1_gate.weight",
                                 r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1_up.weight"],
                      new_name=r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1.weight"),

        ExpertUpGateMergeOp(
            raw_names=[r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1_gate.weight",
                       r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1_up.weight"],
            new_name=r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight"
        ),

        QKVDirectMergeOp(raw_names=(r"text_decoder.decoder.layers.(\d+).self_attention.q_proj.weight",
                                    r"text_decoder.decoder.layers.(\d+).self_attention.kv_a_proj_with_mqa.weight"),
                         new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight")
    ]
    return ops


deepseek_vl_tp_patterns = {
    r"text_decoder.output_layer.weight": RowSplit,
    r"text_decoder.embedding.word_embeddings.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_qb.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.weight": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.bias": RowSplit,
    r"text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1.weight": GLUSplit,
    r"text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc2.weight": ColSplit
}

vision_schema = PPStageSchema(
    firsts=['image_encoder.encoder.patch_embed.', 'image_encoder.encoder.pos_embed'],
    lasts=['image_encoder.encoder.norm', 'image_encoder.encoder.attn_pool', 'image_encoder.projector.'],
    middle='image_encoder.encoder.blocks.',
    all_layer=['image_newline', 'view_seperator']
)


class ConvertVppMMConfigDeepseekVl2(ConvertVppMMConfig):

    def model_post_init(self, _context):
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2Config
        config = cast(DeepseekVLV2Config, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.language_config.num_key_value_heads
        self.common_model_config.vit_num_layers = config.vision_config.layers
        self.common_model_config.llm_num_layers = config.language_config.num_hidden_layers
        self.common_model_config.num_experts = config.language_config.n_routed_experts
        self.common_model_config.tie_word_embeddings = config.language_config.tie_word_embeddings


class DeepSeekVLConverter(Converter):
    """DeepSeekVL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops() -> List[Operator]:
        ops = create_deepseek_vl_ops()
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigDeepseekVl2):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = DeepSeekVLConverter._create_ops()

        convert_hf_to_mm(cfg, ops, deepseek_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass
