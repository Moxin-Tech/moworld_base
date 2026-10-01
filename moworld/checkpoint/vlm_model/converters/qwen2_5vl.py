from typing import Any, cast, List, Optional

from tqdm import tqdm

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model import hf_to_mm, mm_to_hf
from checkpoint.vlm_model.hf_to_mm_ldt import convert_hf_to_mm_ldt
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig, ConvertTorchDCPConfig, \
    HfConfig, ConvertHFLoRAConfig
from checkpoint.vlm_model.converters.qwen2vl import create_qwen2vl_ops, qwen2vl_tp_patterns, canonical_qwen2vl_tp_patterns, \
    ModelConfigQwen2
from checkpoint.vlm_model.hf_to_mm import vision_schema, text_schema, split_by_tp, merge_vpp_index, \
    partition_state_dict_by_pp, save_by_vpp
from checkpoint.vlm_model.mm_to_hf import load_from_mm, merge_by_tp
from checkpoint.vlm_model.operator import (
    Operator, UpGateMergeOp, RenameOp, GLUSplit, RowSplit, ColSplit
)


def create_qwen2_5_vl_ops(enable_canonical_hf_struct: bool, vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int,
                          llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """qwen2.5vl在qwen2vl的基础上vit的mlp变成了glu模式、需要增加合并处理逻辑"""
    if not enable_canonical_hf_struct:
        ops = [
                UpGateMergeOp(
                    raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.weight", r"visual.blocks.(\d+).mlp.up_proj.weight"],
                    new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
                UpGateMergeOp(
                    raw_names=[r"visual.blocks.(\d+).mlp.gate_proj.bias", r"visual.blocks.(\d+).mlp.up_proj.bias"],
                    new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias"),
                RenameOp(
                    patterns=((r'visual.blocks.(\d+).mlp.down_proj',
                                r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),))
            ]
    else:
        ops = [
                RenameOp(
                    (
                        (r"visual.blocks.(\d+).mlp.gate_proj.weight", r"image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.weight"),
                        (r"visual.blocks.(\d+).mlp.up_proj.weight", r"image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.weight"),
                        (r"visual.blocks.(\d+).mlp.gate_proj.bias", r"image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.bias"),
                        (r"visual.blocks.(\d+).mlp.up_proj.bias", r"image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.bias"),
                        (r'visual.blocks.(\d+).mlp.down_proj', r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                        ),
                    )
            ]
    ops += create_qwen2vl_ops(enable_canonical_hf_struct, vit_embed_dim, vit_num_heads, llm_num_query_groups, llm_q_size, llm_kv_size)
    return ops


def create_qwen2_5_vl_lora_ops(new_transformers_weight_key: bool, model_prefix: str,) -> List[Operator]:
    """mindspeed-mm模型LoRA权重转换逻辑"""
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
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.proj.lora_(A|B).weight',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).attn.qkv.lora_(A|B).weight',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).mlp.gate_proj.lora_(A|B).weight',
                 r'image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).mlp.up_proj.lora_(A|B).weight',
                 r'image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_visual_model_name}blocks.(\d+).mlp.down_proj.lora_(A|B).weight',
                 r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.q_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.k_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.v_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.gate_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.gate_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.down_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).mlp.up_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).mlp.up_proj.lora_(A|B).default.weight'),
                (fr'{model_prefix_name}{transformers_text_model_name}layers.(\d+).self_attn.o_proj.lora_(A|B).weight',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.lora_(A|B).default.weight')
            )
        )
    ]

    return ops


#  qwen2.5vl的tp切分在qwen2vl的tp切分基础上，修改了vit中mlp的tp切分逻辑，适应glu结构
qwen2_5_vl_tp_patterns = {**qwen2vl_tp_patterns,
                          **{r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": GLUSplit,
                             r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": GLUSplit}
                          }


canonical_qwen2_5_vl_tp_patterns = {
    **canonical_qwen2vl_tp_patterns,
    **{
        r'image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.weight': RowSplit,
        r'image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.weight': RowSplit,
        r'image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.bias': RowSplit,
        r'image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.bias': RowSplit,
    }
}


canonical_qwen2_5_vl_tp_lora_patterns = {
    r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.lora_A.default.weight': ColSplit,
    r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.lora_B.default.weight': RowSplit,
    r'image_encoder.encoder.blocks.layers.(\d+).mlp.gate_proj.lora_B.default.weight': RowSplit,
    r'image_encoder.encoder.blocks.layers.(\d+).mlp.up_proj.lora_B.default.weight': RowSplit,
    r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.lora_A.default.weight': ColSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.lora_A.default.weight': ColSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.q_proj.lora_B.default.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.k_proj.lora_B.default.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).self_attention.v_proj.lora_B.default.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.gate_proj.lora_B.default.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.up_proj.lora_B.default.weight': RowSplit,
    r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.lora_A.default.weight': ColSplit
}


class ConvertVppMMConfigQwen2_5(ConvertVppMMConfig):
    common_model_config: ModelConfigQwen2 = ModelConfigQwen2()
    """权重转换框架的模型配置"""

    def model_post_init(self, _context):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class ConvertTorchDCPMMConfigQwen2_5(ConvertTorchDCPConfig):
    def model_post_init(self, _context):
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.num_key_value_heads
        self.common_model_config.llm_num_layers = config.num_hidden_layers
        self.common_model_config.vit_num_layers = config.vision_config.depth
        self.common_model_config.tie_word_embeddings = config.tie_word_embeddings


class ConvertHFConfigQwen2_5(ConvertHFConfig):
    common_model_config: ModelConfigQwen2 = ModelConfigQwen2()


class ConvertVppMMLoRAConfigQwen2_5(ConvertHFLoRAConfig):
    common_model_config: ModelConfigQwen2 = ModelConfigQwen2()


class ConvertHFLoRAConfigQwen2_5(ConvertHFConfigQwen2_5):
    hf_config: Optional[HfConfig] = None


class Qwen2_5_VLConverter(Converter):
    """Qwen2.5VL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any, common_model_config: Any) -> List[Operator]:
        from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig
        config = cast(Qwen2_5_VLConfig, config)
        # qwen2.5vl和qwen2vl的差异主要在权重转换的算子以及tp转换时的模式
        llm_head_hidden_size = config.hidden_size // config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.num_attention_heads // config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2_5_vl_ops(common_model_config.enable_canonical_hf_struct,
                                    config.vision_config.hidden_size,
                                    config.vision_config.num_heads,
                                    config.num_key_value_heads,
                                    llm_q_size,
                                    llm_kv_size
                                    )
        return ops

    @staticmethod
    def _create_lora_ops(common_model_config: Any) -> List[Operator]:
        ops = create_qwen2_5_vl_lora_ops(common_model_config.new_transformers_weight_key,
                                         common_model_config.model_prefix
                                         )
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigQwen2_5):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if cfg.common_model_config.enable_canonical_hf_struct:
            qwen2_5_vl_tp_patterns_indeed = canonical_qwen2_5_vl_tp_patterns
        else:
            qwen2_5_vl_tp_patterns_indeed = qwen2_5_vl_tp_patterns
        hf_to_mm.convert_hf_to_mm(cfg, ops, qwen2_5_vl_tp_patterns_indeed, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfigQwen2_5):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if cfg.common_model_config.enable_canonical_hf_struct:
            qwen2_5_vl_tp_patterns_indeed = canonical_qwen2_5_vl_tp_patterns
        else:
            qwen2_5_vl_tp_patterns_indeed = qwen2_5_vl_tp_patterns
        mm_to_hf.convert_mm_to_hf(cfg, ops, qwen2_5_vl_tp_patterns_indeed)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def hf_to_mm_dcp(cfg: ConvertTorchDCPMMConfigQwen2_5):
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config)
        hf_to_mm.convert_hf_to_mm_dcp(cfg, ops)
        # set directory permission for security control
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def lora_hf_to_mm(cfg: ConvertVppMMLoRAConfigQwen2_5):
        """hugging_face模型LoRA权重转换mindspeed-mm模型LoRA权重"""
        if not cfg.common_model_config.enable_canonical_hf_struct:
            raise ValueError("LoRA weight conversion only supports when enable_canonical_hf_struct is set to true.")
        ops = Qwen2_5_VLConverter._create_lora_ops(cfg.common_model_config)
        hf_to_mm.convert_hf_to_mm(cfg, ops, canonical_qwen2_5_vl_tp_lora_patterns, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def lora_mm_to_hf(cfg: ConvertHFLoRAConfigQwen2_5):
        """mindspeed-mm模型LoRA权重转换hugging_face模型LoRA权重"""
        if not cfg.common_model_config.enable_canonical_hf_struct:
            raise ValueError("LoRA weight conversion only supports when enable_canonical_hf_struct is set to true.")
        ops = Qwen2_5_VLConverter._create_lora_ops(cfg.common_model_config)
        mm_to_hf.convert_lora_mm_to_hf(cfg, ops, canonical_qwen2_5_vl_tp_lora_patterns)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        source = cfg.source_parallel_config
        target = cfg.target_parallel_config
        tp_state_dicts = load_from_mm(cfg.source_dir, source.vit_pp_layers, source.llm_pp_layers, source.tp_size)
        state_dict = merge_by_tp(tp_state_dicts=tp_state_dicts, patterns=qwen2_5_vl_tp_patterns)
        tp_state_dicts = split_by_tp(state_dict=state_dict, patterns=qwen2_5_vl_tp_patterns, tp_size=target.tp_size)
        pp_ranges = merge_vpp_index([target.vit_pp_layers], [target.llm_pp_layers], [[]])
        for tp_rank, tp_state_dict in enumerate(tqdm(tp_state_dicts, desc="tp step")):
            pp_state_dicts = partition_state_dict_by_pp(tp_state_dict, pp_ranges, [vision_schema, text_schema])
            save_by_vpp(pp_state_dicts, cfg.target_dir,
                        pp_and_vpp_size=(target.pp_size, 1),
                        tp_rank=tp_rank)
        # 安全管控权限
        set_directory_permissions(cfg.target_dir)

    @staticmethod
    def hf_to_mm_ldt(cfg: ConvertVppMMConfigQwen2_5):
        """huggingface模型转换mindspeed-mm模型权重,配合特性`layerwise_disaggregated_training`使用,支持U形布局"""
        ops = Qwen2_5_VLConverter._create_ops(cfg.hf_config.config, cfg.common_model_config)
        if cfg.common_model_config.enable_canonical_hf_struct:
            qwen2_5_vl_tp_patterns_indeed = canonical_qwen2_5_vl_tp_patterns
        else:
            qwen2_5_vl_tp_patterns_indeed = qwen2_5_vl_tp_patterns
        convert_hf_to_mm_ldt(cfg, ops, qwen2_5_vl_tp_patterns_indeed, [vision_schema, text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)
