from typing import Any, cast, List, Optional

from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig, CommonModelConfig
from checkpoint.vlm_model.hf_to_mm import vision_schema, text_schema, split_by_tp, convert_hf_to_mm, merge_vpp_index, \
    partition_state_dict_by_pp, save_by_vpp
from checkpoint.vlm_model.mm_to_hf import load_from_mm, convert_mm_to_hf, merge_by_tp
from checkpoint.vlm_model.operator import (
    Operator, UpGateMergeOp, QKVMergeOp, RelocateOp, RenameOp, RowSplit, GLUSplit, ColSplit
)


def create_qwen2vl_ops(enable_canonical_hf_struct: bool, vit_embed_dim: int, vit_num_heads: int,
                       llm_num_query_groups: int,
                       llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """qwen2vl权重转换逻辑"""
    ops = [
        RenameOp(
            (
                (r'visual.blocks.(\d+).attn.proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'visual.blocks.(\d+).attn.qkv',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv'),
                (r'visual.blocks.(\d+).mlp.fc', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc'),
                (r'visual.blocks.(\d+).norm1', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'visual.blocks.(\d+).norm2', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'visual.merger.ln_q', r'image_encoder.projector.layernorm'),
                (r'visual.merger.mlp.0', r'image_encoder.projector.encoder.linear_fc1'),
                (r'visual.merger.mlp.2', r'image_encoder.projector.encoder.linear_fc2'),
                (r'visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                (r'model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
                (r'model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (r'model.layers.(\d+).mlp.down_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (
                    r'model.layers.(\d+).post_attention_layernorm',
                    r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (r'model.layers.(\d+).self_attn.o_proj',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj'),
                (r'lm_head', r'text_decoder.output_layer'),
                (r'model.norm', r'text_decoder.decoder.final_layernorm')
            )
        ),
    ]

    if not enable_canonical_hf_struct:
        mega_ops = [
            UpGateMergeOp(
                raw_names=[r"model.layers.(\d+).mlp.gate_proj.weight", r"model.layers.(\d+).mlp.up_proj.weight"],
                new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
            QKVMergeOp(raw_names=(r"model.layers.(\d+).self_attn.q_proj.weight",
                                  r"model.layers.(\d+).self_attn.k_proj.weight",
                                  r"model.layers.(\d+).self_attn.v_proj.weight"),
                       new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                       group=llm_num_query_groups,
                       q_size=llm_q_size,
                       k_size=llm_kv_size,
                       v_size=llm_kv_size,
                       ),
            QKVMergeOp(raw_names=(r"model.layers.(\d+).self_attn.q_proj.bias",
                                  r"model.layers.(\d+).self_attn.k_proj.bias",
                                  r"model.layers.(\d+).self_attn.v_proj.bias"),
                       new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                       group=llm_num_query_groups,
                       q_size=llm_q_size,
                       k_size=llm_kv_size,
                       v_size=llm_kv_size,
                       ),
            RelocateOp(name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                       new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       ),
            RelocateOp(name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                       new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       ),
            RelocateOp(name=r"visual.blocks.(\d+).attn.qkv.weight",
                       new_name=r"visual.blocks.(\d+).attn.qkv.weight",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       ),
            RelocateOp(name=r"visual.blocks.(\d+).attn.qkv.bias",
                       new_name=r"visual.blocks.(\d+).attn.qkv.bias",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       ),
        ]
        ops.extend(mega_ops)

    else:
        canonical_ops = RenameOp(
            (
                (r"visual.blocks.(\d+).attn.qkv.weight",
                 r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight"),
                (r"visual.blocks.(\d+).attn.qkv.bias",
                 r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias"),
                (r'model.layers.(\d+).self_attn.q_proj', r'text_decoder.decoder.layers.(\d+).self_attention.q_proj'),
                (r'model.layers.(\d+).self_attn.k_proj', r'text_decoder.decoder.layers.(\d+).self_attention.k_proj'),
                (r'model.layers.(\d+).self_attn.v_proj', r'text_decoder.decoder.layers.(\d+).self_attention.v_proj'),
                (r'model.layers.(\d+).mlp.up_proj', r'text_decoder.decoder.layers.(\d+).mlp.up_proj'),
                (r'model.layers.(\d+).mlp.gate_proj', r'text_decoder.decoder.layers.(\d+).mlp.gate_proj')
            )
        )
        ops.append(canonical_ops)
    return ops


base_qwen2vl_tp_patterns = {
    r"text_decoder.output_layer.weight": RowSplit,
    r"text_decoder.embedding.word_embeddings.weight": RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': ColSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': ColSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": ColSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": RowSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": RowSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": RowSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": RowSplit,
    r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
    r"image_encoder.projector.encoder.linear_fc1.bias": RowSplit,
    r"image_encoder.projector.encoder.linear_fc1.weight": RowSplit,
    r"image_encoder.projector.encoder.linear_fc2.weight": ColSplit
}

qwen2vl_tp_patterns = {
    **base_qwen2vl_tp_patterns,
    **{
        r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': RowSplit,
    }
}

canonical_qwen2vl_tp_patterns = {
    **base_qwen2vl_tp_patterns,
    **{
        r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.bias': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.bias': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.bias': RowSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.gate_proj.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.up_proj.weight': RowSplit
    }
}


class ModelConfigQwen2(CommonModelConfig):
    enable_canonical_hf_struct: Optional[bool] = False
    """是否使用标准huggingface模型结构"""

    new_transformers_weight_key: Optional[bool] = False
    """是否使用新transformers版本下的模型权重名"""

    model_prefix: Optional[str] = None
    """模型权重名包含额外前缀"""


class ConvertVppMMConfigQwen2(ConvertVppMMConfig):
    common_model_config: ModelConfigQwen2 = ModelConfigQwen2()
    """权重转换框架的模型配置"""

    def model_post_init(self, _context):
        from transformers.models.qwen2_vl import Qwen2VLConfig
        config = cast(Qwen2VLConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class ConvertHFConfigQwen2(ConvertHFConfig):
    common_model_config: ModelConfigQwen2 = ModelConfigQwen2()
    """权重转换框架的模型配置"""


class Qwen2VLConverter(Converter):
    """Qwen2VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any, common_model_config: Any) -> List[Operator]:
        from transformers.models.qwen2_vl import Qwen2VLConfig
        config = cast(Qwen2VLConfig, config)
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2vl_ops(common_model_config.enable_canonical_hf_struct,
                                 config.vision_config.embed_dim,
                                 config.vision_config.num_heads,
                                 config.num_key_value_heads,
                                 llm_q_size,
                                 llm_kv_size
                                 )
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigQwen2):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = Qwen2VLConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if cfg.common_model_config.enable_canonical_hf_struct:
            qwen2vl_tp_patterns_indeed = canonical_qwen2vl_tp_patterns
        else:
            qwen2vl_tp_patterns_indeed = qwen2vl_tp_patterns
        convert_hf_to_mm(cfg, ops, qwen2vl_tp_patterns_indeed, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfigQwen2):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops = Qwen2VLConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if cfg.common_model_config.enable_canonical_hf_struct:
            qwen2vl_tp_patterns_indeed = canonical_qwen2vl_tp_patterns
        else:
            qwen2vl_tp_patterns_indeed = qwen2vl_tp_patterns
        convert_mm_to_hf(cfg, ops, qwen2vl_tp_patterns_indeed)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        source = cfg.source_parallel_config
        target = cfg.target_parallel_config
        tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
        state_dict = merge_by_tp(tp_state_dicts=tp_state_dicts, patterns=qwen2vl_tp_patterns)
        tp_state_dicts = split_by_tp(state_dict=state_dict, patterns=qwen2vl_tp_patterns, tp_size=target.tp_size)
        pp_ranges = merge_vpp_index([target.vit_pp_layers], [target.llm_pp_layers], [[]])
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = partition_state_dict_by_pp(tp_state_dict, pp_ranges, [vision_schema, text_schema])
            save_by_vpp(pp_state_dicts, cfg.target_dir,
                        pp_and_vpp_size=(target.pp_size, 1),
                        tp_rank=tp_rank)
        # 安全管控权限
        set_directory_permissions(cfg.target_dir)
