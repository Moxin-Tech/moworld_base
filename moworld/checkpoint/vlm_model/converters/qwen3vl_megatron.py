from typing import cast, List

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model import hf_to_mm, mm_to_hf
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig
from checkpoint.vlm_model.hf_to_mm import PPStageSchema, text_schema
from checkpoint.vlm_model.operator import Operator, RenameOp, ExpertUpGateMergeOp, GLUSplit, ColSplit, RowSplit, QKVMergeOp, RelocateOp, ExpertSplitOp, UpGateMergeOp


vision_schema = PPStageSchema(
    firsts=['image_encoder.encoder.patch_embed.', 'image_encoder.encoder.pos_embed.'],
    lasts=['image_encoder.projector.'],
    middle='image_encoder.encoder.blocks.layers.'
)


def create_qwen3_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int, llm_q_size: int,
                        llm_kv_size: int, num_hidden_layers: int, num_experts: int, deepstack_visual_indexes: int) -> List[Operator]:
    ops = [
              RenameOp(
                  (
                      (r'model.visual.blocks.(\d+).norm1.bias', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm.bias'),
                      (r'model.visual.blocks.(\d+).norm1.weight', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm.weight'),
                      (r'model.visual.blocks.(\d+).norm2.bias', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm.bias'),
                      (r'model.visual.blocks.(\d+).norm2.weight',
                       r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm.weight'),
                      (r'model.visual.blocks.(\d+).attn.qkv.weight',
                       r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight'),
                      (r'model.visual.blocks.(\d+).attn.qkv.bias',
                       r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias'),
                      (r'model.visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                      (r'model.visual.blocks.(\d+).attn.proj', r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                      (r'model.visual.blocks.(\d+).mlp.linear_fc', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc'),
                      (r'model.visual.merger.linear_fc', r'image_encoder.projector.encoder.linear_fc'),
                      (r'model.visual.merger.norm', r'image_encoder.projector.layernorm'),
                      (r'model.visual.pos_embed', r'image_encoder.encoder.pos_embed'),

                      (r'model.language_model.layers.(\d+).mlp.gate.weight', r'text_decoder.decoder.layers.(\d+).mlp.router.weight'),
                      (r'model.language_model.layers.(\d+).self_attn.q_norm.weight',
                       r'text_decoder.decoder.layers.(\d+).self_attention.q_layernorm.weight'),
                      (r'model.language_model.layers.(\d+).self_attn.k_norm.weight',
                       r'text_decoder.decoder.layers.(\d+).self_attention.k_layernorm.weight'),
                      (r'model.language_model.layers.(\d+).self_attn.o_proj.weight',
                       r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight'),
                      (r'model.language_model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                      (r'model.language_model.layers.(\d+).post_attention_layernorm',
                          r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                      (r'model.language_model.embed_tokens.weight',
                       r'text_decoder.embedding.word_embeddings.weight'),
                      (r'model.language_model.norm.weight',
                       r'text_decoder.decoder.final_layernorm.weight'),
                      (r'lm_head', r'text_decoder.output_layer')
                  )
              ),
              QKVMergeOp(raw_names=(r"model.language_model.layers.(\d+).self_attn.q_proj.weight",
                                    r"model.language_model.layers.(\d+).self_attn.k_proj.weight",
                                    r"model.language_model.layers.(\d+).self_attn.v_proj.weight"),
                         new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                         group=llm_num_query_groups,
                         q_size=llm_q_size,
                         k_size=llm_kv_size,
                         v_size=llm_kv_size,
                         ),
              RelocateOp(name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                         new_name=r"model.visual.blocks.(\d+).attn.qkv.weight",
                         group=vit_num_heads,
                         split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                         ),
              RelocateOp(name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                         new_name=r"model.visual.blocks.(\d+).attn.qkv.bias",
                         group=vit_num_heads,
                         split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                         ),
          ]
    expert_split_ops = [
                        ExpertSplitOp(raw_name=rf"model.language_model.layers.{idx}.mlp.experts.gate_up_proj",
                                      new_name=rf"text_decoder.decoder.layers.{idx}.mlp.experts.local_experts.(\d+).linear_fc1.weight",
                                      num_experts=num_experts) for idx in range(num_hidden_layers)
                       ] + \
                       [
                        ExpertSplitOp(raw_name=rf"model.language_model.layers.{idx}.mlp.experts.down_proj",
                                      new_name=rf"text_decoder.decoder.layers.{idx}.mlp.experts.local_experts.(\d+).linear_fc2.weight",
                                      num_experts=num_experts) for idx in range(num_hidden_layers)
                       ]
    deepstack_rename_op = [RenameOp(
                                (
                                    (rf'model.visual.deepstack_merger_list.{idx}.linear_fc',
                                    rf'image_encoder.encoder.blocks.layers.{deepstack_visual_indexes[idx]}.deepstack_layer.encoder.linear_fc'),
                                    (rf'model.visual.deepstack_merger_list.{idx}.norm',
                                    rf'image_encoder.encoder.blocks.layers.{deepstack_visual_indexes[idx]}.deepstack_layer.layernorm')
                                )
                            ) for idx in range(len(deepstack_visual_indexes))
                          ]
    dense_merge_op = [
                        UpGateMergeOp(
                            raw_names=[r"model.language_model.layers.(\d+).mlp.gate_proj.weight", r"model.language_model.layers.(\d+).mlp.up_proj.weight"],
                            new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight")
                     ]
    dense_rename_op = [
                        RenameOp(
                            (
                                (r"model.language_model.layers.(\d+).mlp.down_proj", r"text_decoder.decoder.layers.(\d+).mlp.linear_fc2"),

                            )
                        )
                      ]
    return ops + expert_split_ops + deepstack_rename_op + dense_merge_op + dense_rename_op


qwen3_vl_tp_patterns = {
    **{
        r"text_decoder.output_layer.weight": RowSplit,
        r"text_decoder.embedding.word_embeddings.weight": RowSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': ColSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': ColSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": ColSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
        r"image_encoder.projector.encoder.linear_fc1.bias": RowSplit,
        r"image_encoder.projector.encoder.linear_fc1.weight": RowSplit,
        r"image_encoder.projector.encoder.linear_fc2.weight": ColSplit,
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight": GLUSplit,
        r"text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": ColSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).deepstack_layer.encoder.linear_fc1.weight": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).deepstack_layer.encoder.linear_fc1.bias": RowSplit,
        r"image_encoder.encoder.blocks.layers.(\d+).deepstack_layer.encoder.linear_fc2.weight": ColSplit,
    }
}


class ConvertVppMMConfigQwen3(ConvertVppMMConfig):

    def model_post_init(self, _context):
        from transformers.models.qwen3_vl_moe import Qwen3VLMoeConfig
        config = cast(Qwen3VLMoeConfig, self.hf_config.config)

        self.common_model_config.num_key_value_heads = config.text_config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.text_config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.num_experts = config.text_config.num_experts if hasattr(config.text_config, 'num_experts') else 0
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class Qwen3VLMegatronConverter(Converter):
    """Qwen3VL模型转换工具"""

    @staticmethod
    def _create_ops(config) -> List[Operator]:
        from transformers.models.qwen3_vl import Qwen3VLConfig
        config = cast(Qwen3VLConfig, config)
        num_key_value_heads = config.text_config.num_key_value_heads
        llm_head_hidden_size = config.text_config.head_dim if config.text_config.head_dim is not None \
            else config.text_config.hidden_size // config.text_config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.text_config.num_attention_heads // config.text_config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        num_hidden_layers = config.text_config.num_hidden_layers
        num_experts = config.text_config.num_experts if hasattr(config.text_config, 'num_experts') else 0
        deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes if hasattr(config.vision_config, 'deepstack_visual_indexes') else []
        return create_qwen3_vl_ops(config.vision_config.hidden_size,
                                   config.vision_config.num_heads,
                                   num_key_value_heads,
                                   llm_q_size,
                                   llm_kv_size,
                                   num_hidden_layers,
                                   num_experts,
                                   deepstack_visual_indexes
                                   )

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigQwen3):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = Qwen3VLMegatronConverter._create_ops(cfg.hf_config.config)
        hf_to_mm.convert_hf_to_mm(cfg, ops, qwen3_vl_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        config = cfg.hf_config.config
        ops = Qwen3VLMegatronConverter._create_ops(cfg.hf_config.config)
        mm_to_hf.convert_mm_to_hf(cfg, ops, qwen3_vl_tp_patterns)
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def resplit():
        """mindspeed-mm模型权重重新切分"""
        pass
