from typing import Any, List

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model import hf_to_mm, mm_to_hf
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig
from checkpoint.vlm_model.operator import (
    ColSplit, GLUSplit, Operator, RowSplit, UnalignedColSplit, UnalignedEmbeddingSplit, UnalignedRowSplit,
    UpGateMergeOp, QKVMergeOp, RenameOp
)


def create_intern_vl_ops(llm_arch: str, llm_num_query_groups: int, llm_q_size: int, llm_kv_size: int) -> List[Operator]:
    """intern_vl权重转换逻辑"""
    rename_op = (
        (r'vision_model.(.*).attn.qkv', r'image_encoder.encoder.(.*).self_attention.linear_qkv'),
        (r'vision_model.(.*).attn.q_norm', r'image_encoder.encoder.(.*).self_attention.q_layernorm'),
        (r'vision_model.(.*).attn.k_norm', r'image_encoder.encoder.(.*).self_attention.k_layernorm'),
        (r'vision_model.(.*).attn.proj', r'image_encoder.encoder.(.*).self_attention.linear_proj'),
        (r'vision_model.(.*).mlp.fc1', r'image_encoder.encoder.(.*).mlp.linear_fc1'),
        (r'vision_model.(.*).mlp.fc2', r'image_encoder.encoder.(.*).mlp.linear_fc2'),
        (r'vision_model.(.*).norm1', r'image_encoder.encoder.(.*).input_layernorm'),
        (r'vision_model.(.*).norm2', r'image_encoder.encoder.(.*).pre_mlp_layernorm'),
        (r'vision_model.encoder.layers', r'image_encoder.encoder.encoder.layers'),
        (r'vision_model.embeddings.', r'image_encoder.encoder.embeddings.'),

        (r'mlp1.0', r'image_encoder.projector.norm'),
        (r'mlp1.1', r'image_encoder.projector.linear_fc1'),
        (r'mlp1.3', r'image_encoder.projector.linear_fc2'),
    )
    if llm_arch == 'LlamaForCausalLM':
        rename_op += (
            (r'language_model.lm_head', r'text_decoder.output_layer'),
            (r'language_model.model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
            (r'language_model.model.layers.(.*).self_attn.q_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.wq'),
            (r'language_model.model.layers.(.*).self_attn.k_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.wk'),
            (r'language_model.model.layers.(.*).self_attn.v_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.wv'),
            (r'language_model.model.layers.(.*).self_attn.o_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_proj'),
            (r'language_model.model.layers.(.*).gate_proj', r'text_decoder.decoder.layers.(.*).linear_fc1_gate'),
            (r'language_model.model.layers.(.*).up_proj', r'text_decoder.decoder.layers.(.*).linear_fc1_up'),
            (r'language_model.model.layers.(.*).down_proj', r'text_decoder.decoder.layers.(.*).linear_fc2'),
            (r'language_model.model.layers.(.*).post_attention_layernorm',
             r'text_decoder.decoder.layers.(.*).pre_mlp_layernorm'),
            (r'language_model.model.norm', r'text_decoder.decoder.final_layernorm'),
            (r'language_model.model.layers', r'text_decoder.decoder.layers'),
        )
    elif llm_arch == 'InternLM2ForCausalLM':
        rename_op += (
            (r'language_model.model.layers.(.*).attention.wqkv',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_qkv'),
            (r'language_model.model.layers.(.*).attention.wo',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_proj'),
            (r'language_model.model.layers.(.*).feed_forward.w1',
             r'text_decoder.decoder.layers.(.*).mlp.linear_fc1_gate'),
            (r'language_model.model.layers.(.*).feed_forward.w3',
             r'text_decoder.decoder.layers.(.*).mlp.linear_fc1_up'),
            (r'language_model.model.layers.(.*).feed_forward.w2', r'text_decoder.decoder.layers.(.*).mlp.linear_fc2'),
            (r'language_model.model.layers.(.*).attention_norm', r'text_decoder.decoder.layers.(.*).input_layernorm'),
            (r'language_model.model.layers.(.*).ffn_norm', r'text_decoder.decoder.layers.(.*).pre_mlp_layernorm'),
            (r'language_model.model.norm', r'text_decoder.decoder.final_layernorm'),
            (r'language_model.model.tok_embeddings', r'text_decoder.embedding.word_embeddings'),
            (r'language_model.output', r'text_decoder.output_layer'),
            (r'language_model.', r'text_decoder.'),
        )
    elif llm_arch == 'Qwen2ForCausalLM':
        rename_op += (
            (r'language_model.lm_head', r'text_decoder.output_layer'),
            (r'language_model.model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
            (r'language_model.model.norm', r'text_decoder.decoder.final_layernorm'),
            (r'language_model.model.layers.(.*).self_attn.q_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_q'),
            (r'language_model.model.layers.(.*).self_attn.k_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_k'),
            (r'language_model.model.layers.(.*).self_attn.v_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_v'),
            (r'language_model.model.layers.(.*).self_attn.o_proj',
             r'text_decoder.decoder.layers.(.*).self_attention.linear_proj'),
            (r'language_model.model.layers.(.*).post_attention_layernorm',
             r'text_decoder.decoder.layers.(.*).pre_mlp_layernorm'),
            (r'language_model.model.layers.(.*).gate_proj', r'text_decoder.decoder.layers.(.*).linear_fc1_gate'),
            (r'language_model.model.layers.(.*).up_proj', r'text_decoder.decoder.layers.(.*).linear_fc1_up'),
            (r'language_model.model.layers.(.*).down_proj', r'text_decoder.decoder.layers.(.*).linear_fc2'),
            (r'language_model.model.layers', r'text_decoder.decoder.layers'),
        )
    qkv_merge_ops = []
    if llm_arch == 'LlamaForCausalLM':
        qkv_merge_ops = [QKVMergeOp(raw_names=(r"text_decoder.decoder.layers.(\d+).self_attention.wq.weight",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.wk.weight",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.wv.weight"),
                                    new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                                    group=llm_num_query_groups,
                                    q_size=llm_q_size,
                                    k_size=llm_kv_size,
                                    v_size=llm_kv_size,
                                    )]
    elif llm_arch == 'Qwen2ForCausalLM':
        qkv_merge_ops = [QKVMergeOp(raw_names=(r"text_decoder.decoder.layers.(\d+).self_attention.linear_q.weight",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.linear_k.weight",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.linear_v.weight"),
                                    new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                                    group=llm_num_query_groups,
                                    q_size=llm_q_size,
                                    k_size=llm_kv_size,
                                    v_size=llm_kv_size,
                                    ),

                         QKVMergeOp(raw_names=(r"text_decoder.decoder.layers.(\d+).self_attention.linear_q.bias",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.linear_k.bias",
                                               r"text_decoder.decoder.layers.(\d+).self_attention.linear_v.bias"),
                                    new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                                    group=llm_num_query_groups,
                                    q_size=llm_q_size,
                                    k_size=llm_kv_size,
                                    v_size=llm_kv_size,
                                    )]
    ops = [
              RenameOp(rename_op),
              UpGateMergeOp(raw_names=[r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1_gate.weight",
                                       r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1_up.weight"],
                            new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
          ] + qkv_merge_ops

    return ops


def create_internvl_tp_patterns(vision_attention_heads: int) -> dict:
    """创建InternVL的TP模式"""

    tp_patterns = {
        r"text_decoder.output_layer.weight": UnalignedEmbeddingSplit,
        r"text_decoder.embedding.word_embeddings.weight": UnalignedEmbeddingSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': GLUSplit,
        r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': ColSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': RowSplit,
        r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': ColSplit,
        r"image_encoder.encoder.encoder.layers.(\d+).self_attention.linear_proj.weight": UnalignedColSplit(
            vision_attention_heads),  # InternVL系列ViT的heads可能存在无法均匀切分情况
        r"image_encoder.encoder.encoder.layers.(\d+).self_attention.linear_qkv.bias": UnalignedRowSplit(
            vision_attention_heads),
        r"image_encoder.encoder.encoder.layers.(\d+).self_attention.linear_qkv.weight": UnalignedRowSplit(
            vision_attention_heads),
        r"image_encoder.encoder.encoder.layers.(\d+).mlp.linear_fc1.bias": RowSplit,
        r"image_encoder.encoder.encoder.layers.(\d+).mlp.linear_fc1.weight": RowSplit,
        r"image_encoder.encoder.encoder.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
    }
    return tp_patterns


vision_schema = hf_to_mm.PPStageSchema(
    firsts=['image_encoder.encoder.embeddings.'],
    lasts=['image_encoder.projector.'],
    middle='image_encoder.encoder.encoder.layers.'
)


class ConvertVppMMConfigInternVL(ConvertVppMMConfig):

    def model_post_init(self, _context):
        self.common_model_config.num_key_value_heads = self.hf_config.config.llm_config.num_key_value_heads
        self.common_model_config.vit_num_layers = self.hf_config.config.vision_config.num_hidden_layers
        self.common_model_config.llm_num_layers = self.hf_config.config.llm_config.num_hidden_layers
        self.common_model_config.tie_word_embeddings = self.hf_config.config.llm_config.tie_word_embeddings


class InternVLConverter(Converter):
    """InternVL模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any) -> List[Operator]:
        llm_head_hidden_size = config.llm_config.hidden_size // config.llm_config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.llm_config.num_attention_heads // config.llm_config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size

        ops = create_intern_vl_ops(config.llm_config.architectures[0], config.llm_config.num_key_value_heads,
                                   llm_q_size, llm_kv_size)
        return ops

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigInternVL):
        """huggingface模型转换mindspeed-mm模型权重"""
        config = cfg.hf_config.config
        ops = InternVLConverter._create_ops(config)

        intern_vl_tp_patterns = create_internvl_tp_patterns(config.vision_config.num_attention_heads)

        hf_to_mm.convert_hf_to_mm(cfg, ops, intern_vl_tp_patterns, [vision_schema, hf_to_mm.text_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops = InternVLConverter._create_ops(cfg.hf_config.config)
        # 处理流程需要反转
        ops.reverse()
        intern_vl_tp_patterns = create_internvl_tp_patterns(cfg.hf_config.config.vision_config.num_attention_heads)
        mm_to_hf.convert_mm_to_hf(cfg, ops, intern_vl_tp_patterns)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass
