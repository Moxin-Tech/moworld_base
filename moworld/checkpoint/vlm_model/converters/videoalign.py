from pathlib import Path
from typing import Any, cast, List, Optional

from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model.config import ConvertHFConfig, ConvertResplitConfig, CommonModelConfig
from checkpoint.vlm_model.converters.qwen2vl import ConvertVppMMConfigQwen2
from checkpoint.vlm_model.hf_to_mm import vision_schema, PPStageSchema, split_by_tp, convert_hf_to_mm, merge_vpp_index, \
    partition_state_dict_by_pp, save_by_vpp
from checkpoint.vlm_model.mm_to_hf import load_from_mm, convert_mm_to_hf, merge_by_tp
from checkpoint.vlm_model.operator import (
    Operator, UpGateMergeOp, QKVMergeOp, RelocateOp, RenameOp, ResizeEmbedOp, RowSplit, GLUSplit, ColSplit
)

text_schema = PPStageSchema(
    firsts=['text_decoder.embedding.'],
    lasts=['text_decoder.decoder.final_layernorm.', 'rm_head'],
    middle='text_decoder.decoder.layers.'
)


def create_videoalign_ops(new_transformers_weight_key: bool, enable_canonical_hf_struct: bool, model_prefix: str, resize_vocab_size: int,
                          vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int, llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """videoalign权重转换逻辑"""
    model_prefix_name = model_prefix if model_prefix else ""
    if new_transformers_weight_key:
        transformers_text_model_name = 'model.language_model.'
        transformers_visual_model_name = 'model.visual.'
    else:
        transformers_text_model_name = 'model.'
        transformers_visual_model_name = 'visual.'

    ops = [
        RenameOp(
            (
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.proj',
                 fr'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv',
                 fr'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).mlp.fc',
                 fr'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).norm1',
                 fr'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).norm2',
                 fr'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (fr'{model_prefix_name}{transformers_visual_model_name}merger.ln_q',
                 fr'image_encoder.projector.layernorm'),
                (fr'{model_prefix_name}{transformers_visual_model_name}merger.mlp.0',
                 fr'image_encoder.projector.encoder.linear_fc1'),
                (fr'{model_prefix_name}{transformers_visual_model_name}merger.mlp.2',
                 fr'image_encoder.projector.encoder.linear_fc2'),
                (fr'{model_prefix_name}{transformers_visual_model_name}patch_embed.proj',
                 fr'image_encoder.encoder.patch_embed.proj'),
                (fr'{model_prefix_name}{transformers_text_model_name}embed_tokens',
                 fr'text_decoder.embedding.word_embeddings'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).input_layernorm',
                 fr'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.down_proj',
                 fr'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (
                    fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).post_attention_layernorm',
                    fr'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.o_proj',
                 fr'text_decoder.decoder.layers.(\d+).self_attention.linear_proj'),
                (fr'{model_prefix_name}lm_head', fr'text_decoder.output_layer'),
                (fr'{model_prefix_name}{transformers_text_model_name}norm', fr'text_decoder.decoder.final_layernorm'),
                (fr'{model_prefix_name}rm_head.weight', fr'rm_head.weight'),
            )
        ),
    ]

    if not enable_canonical_hf_struct:
        relocate_ops = [
            RelocateOp(name=fr"{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv.weight",
                       new_name=fr"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       ),
            RelocateOp(name=fr"{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv.bias",
                       new_name=fr"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                       group=vit_num_heads,
                       split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                       )
        ]
        merge_ops = [
            UpGateMergeOp(raw_names=[
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.gate_proj.base_layer.weight",
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.up_proj.base_layer.weight"],
                          new_name=fr"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.base_layer.weight"),
            QKVMergeOp(raw_names=(
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.q_proj.base_layer.weight",
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.k_proj.base_layer.weight",
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.v_proj.base_layer.weight"),
                       new_name=fr"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.base_layer.weight",
                       group=llm_num_query_groups,
                       q_size=llm_q_size,
                       k_size=llm_kv_size,
                       v_size=llm_kv_size,
                       ),
            QKVMergeOp(raw_names=(
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.q_proj.base_layer.bias",
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.k_proj.base_layer.bias",
                fr"{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.v_proj.base_layer.bias"),
                       new_name=fr"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.base_layer.bias",
                       group=llm_num_query_groups,
                       q_size=llm_q_size,
                       k_size=llm_kv_size,
                       v_size=llm_kv_size,
                       ),
        ]
        rename_op = RenameOp(
            (
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.linear_qkv',
                 fr'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.fc1',
                 fr'text_decoder.decoder.layers.(\d+).mlp.linear_fc1')
            )
        )
        ops.extend(relocate_ops)
        ops.extend(merge_ops)
        ops.append(rename_op)

    else:
        rename_ops = RenameOp(
            (
                (fr"{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv.weight",
                 fr"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight"),
                (fr"{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv.bias",
                 fr"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias"),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.q_proj',
                 fr'text_decoder.decoder.layers.(\d+).self_attention.q_proj'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.k_proj',
                 fr'text_decoder.decoder.layers.(\d+).self_attention.k_proj'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.v_proj',
                 fr'text_decoder.decoder.layers.(\d+).self_attention.v_proj'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.up_proj',
                 fr'text_decoder.decoder.layers.(\d+).mlp.up_proj'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.gate_proj',
                 fr'text_decoder.decoder.layers.(\d+).mlp.gate_proj')
            )
        )
        ops.append(rename_ops)

    if resize_vocab_size:
        ops.append(ResizeEmbedOp(fr'text_decoder.embedding.word_embeddings.weight', resize_vocab_size))
    return ops


videoalign_tp_patterns = {
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

megatron_videoalign_tp_patterns = {
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': RowSplit
}

canonical_hf_videoalign_tp_patterns = {
    r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.bias': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.bias': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.bias': RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.gate_proj.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.up_proj.weight': RowSplit
}


class ModelConfigVideoAlign(CommonModelConfig):
    new_transformers_weight_key: Optional[bool] = None
    """是否使用新transformers版本下的模型权重名"""

    enable_canonical_hf_struct: Optional[bool] = False
    """是否使用标准huggingface模型结构"""

    model_prefix: Optional[str] = None
    """模型权重名包含额外前缀"""

    resize_vocab_size: Optional[int] = None
    """需要更改的vocab_size并同步更改word_embeddings的shape"""


class ConvertVppMMConfigVideoAlign(ConvertVppMMConfigQwen2):
    pt_path: Optional[Path] = None
    """pt/pth权重文件路径"""

    save_lora_only: Optional[bool] = False
    """是否只保存lora部分权重,默认为False"""

    common_model_config: ModelConfigVideoAlign = ModelConfigVideoAlign()
    """权重转换框架的模型配置"""


class VideoAlignConverter(Converter):
    """VideoAlign模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(hf_config: Any, common_model_config: Any) -> List[Operator]:
        from transformers.models.qwen2_vl import Qwen2VLConfig
        hf_config = cast(Qwen2VLConfig, hf_config)
        llm_head_hidden_size = hf_config.hidden_size // hf_config.num_attention_heads
        llm_q_size = llm_head_hidden_size * hf_config.num_attention_heads // hf_config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_videoalign_ops(common_model_config.new_transformers_weight_key,
                                    common_model_config.enable_canonical_hf_struct,
                                    common_model_config.model_prefix,
                                    common_model_config.resize_vocab_size,
                                    hf_config.vision_config.embed_dim,
                                    hf_config.vision_config.num_heads,
                                    hf_config.num_key_value_heads,
                                    llm_q_size,
                                    llm_kv_size
                                    )
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigVideoAlign):
        """huggingface模型转换mindspeed mm模型权重"""
        ops = VideoAlignConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if not cfg.common_model_config.enable_canonical_hf_struct:
            videoalign_tp_patterns.update(megatron_videoalign_tp_patterns)
        else:
            videoalign_tp_patterns.update(canonical_hf_videoalign_tp_patterns)
        convert_hf_to_mm(cfg, ops, videoalign_tp_patterns, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed mm模型转换huggingface模型权重"""
        ops = VideoAlignConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if not cfg.common_model_config.enable_canonical_hf_struct:
            videoalign_tp_patterns.update(megatron_videoalign_tp_patterns)
        else:
            videoalign_tp_patterns.update(canonical_hf_videoalign_tp_patterns)
        convert_mm_to_hf(cfg, ops, videoalign_tp_patterns)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed mm模型权重重新切分"""
        source = cfg.source_parallel_config
        target = cfg.target_parallel_config
        tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
        state_dict = merge_by_tp(tp_state_dicts, source.tp_size)
        tp_state_dicts = split_by_tp(state_dict, target.tp_size)
        pp_ranges = merge_vpp_index([target.vit_pp_layers], [target.llm_pp_layers], [[]])
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = partition_state_dict_by_pp(tp_state_dict, pp_ranges, [vision_schema, text_schema])
            save_by_vpp(pp_state_dicts, cfg.target_dir,
                        pp_and_vpp_size=(target.pp_size, 1),
                        tp_rank=tp_rank)
        # 安全管控权限
        set_directory_permissions(cfg.target_dir)
