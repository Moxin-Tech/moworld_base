from typing import Any, cast, Tuple, List

from checkpoint.common.converter import Converter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.vlm_model import hf_to_mm, mm_to_hf
from checkpoint.vlm_model.config import ConvertVppMMConfig, ConvertHFConfig, ConvertResplitConfig
from checkpoint.vlm_model.converters.qwen2_5vl import qwen2_5_vl_tp_patterns
from checkpoint.vlm_model.hf_to_mm import vision_schema, text_schema, audio_schema
from checkpoint.vlm_model.operator import Operator, UpGateMergeOp, QKVMergeOp, QVToQKVMergeOp, \
    RenameOp, RowSplit, ColSplit


def create_qwen2_5_omni_ops(vit_num_heads: int, llm_num_query_groups: int, audio_num_heads: int,
                            head_sizes: Tuple[int, int, int, int]) -> List[Operator]:
    """
    创建qwen2.5-omni的权重转换操作列表

    参数:
        vit_num_heads: ViT中的注意力头数量
        llm_num_query_groups: LLM中的查询组数
        audio_num_heads: 音频处理中的注意力头数量
        head_sizes: 包含四个整数的元组，依次为:
                   - vit_head_hidden_size (ViT中每个头的隐藏层大小)
                   - audio_head_hidden_size (音频处理中每个头的隐藏层大小)
                   - llm_q_size (LLM中查询向量的大小)
                   - llm_kv_size (LLM中键/值向量的大小)
    """
    vit_head_hidden_size, audio_head_hidden_size, llm_q_size, llm_kv_size = head_sizes
    """qwen2.5-omni 权重转换逻辑"""
    ops = [
        UpGateMergeOp(
            raw_names=[r"thinker.visual.blocks.(\d+).mlp.gate_proj.weight",
                       r"thinker.visual.blocks.(\d+).mlp.up_proj.weight"],
            new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
        UpGateMergeOp(
            raw_names=[r"thinker.visual.blocks.(\d+).mlp.gate_proj.bias",
                       r"thinker.visual.blocks.(\d+).mlp.up_proj.bias"],
            new_name=r"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias"),
        UpGateMergeOp(raw_names=[r"thinker.model.layers.(\d+).mlp.gate_proj.weight",
                                 r"thinker.model.layers.(\d+).mlp.up_proj.weight"],
                      new_name=r"text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
        QKVMergeOp(raw_names=(r"thinker.model.layers.(\d+).self_attn.q_proj.weight",
                              r"thinker.model.layers.(\d+).self_attn.k_proj.weight",
                              r"thinker.model.layers.(\d+).self_attn.v_proj.weight"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.model.layers.(\d+).self_attn.q_proj.bias",
                              r"thinker.model.layers.(\d+).self_attn.k_proj.bias",
                              r"thinker.model.layers.(\d+).self_attn.v_proj.bias"),
                   new_name=r"text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                   group=llm_num_query_groups,
                   q_size=llm_q_size,
                   k_size=llm_kv_size,
                   v_size=llm_kv_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.visual.blocks.(\d+).attn.q.weight",
                              r"thinker.visual.blocks.(\d+).attn.k.weight",
                              r"thinker.visual.blocks.(\d+).attn.v.weight"),
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=vit_num_heads,
                   q_size=vit_head_hidden_size,
                   k_size=vit_head_hidden_size,
                   v_size=vit_head_hidden_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.visual.blocks.(\d+).attn.q.bias",
                              r"thinker.visual.blocks.(\d+).attn.k.bias",
                              r"thinker.visual.blocks.(\d+).attn.v.bias"),
                   new_name=r"image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=vit_num_heads,
                   q_size=vit_head_hidden_size,
                   k_size=vit_head_hidden_size,
                   v_size=vit_head_hidden_size,
                   ),
        QKVMergeOp(raw_names=(r"thinker.audio_tower.layers.(\d+).self_attn.q_proj.weight",
                              r"thinker.audio_tower.layers.(\d+).self_attn.k_proj.weight",
                              r"thinker.audio_tower.layers.(\d+).self_attn.v_proj.weight"),
                   new_name=r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=audio_num_heads,
                   q_size=audio_head_hidden_size,
                   k_size=audio_head_hidden_size,
                   v_size=audio_head_hidden_size,
                   ),
        # 音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典，以便进行后续的qkv拼接
        QVToQKVMergeOp(raw_names=(r"thinker.audio_tower.layers.(\d+).self_attn.q_proj.bias",
                                  r"thinker.audio_tower.layers.(\d+).self_attn.v_proj.bias"),
                       new_name=r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                       group=audio_num_heads,
                       q_size=audio_head_hidden_size,
                       k_size=audio_head_hidden_size,
                       v_size=audio_head_hidden_size,
                       ),
        RenameOp(
            (
                (r'thinker.visual.blocks.(\d+).attn.proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.visual.blocks.(\d+).mlp.down_proj',
                 r'image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.visual.blocks.(\d+).norm1', r'image_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'thinker.visual.blocks.(\d+).norm2', r'image_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.visual.merger.ln_q', r'image_encoder.projector.layernorm'),
                (r'thinker.visual.merger.mlp.0', r'image_encoder.projector.encoder.linear_fc1'),
                (r'thinker.visual.merger.mlp.2', r'image_encoder.projector.encoder.linear_fc2'),
                (r'thinker.visual.patch_embed.proj', r'image_encoder.encoder.patch_embed.proj'),
                (r'thinker.model.embed_tokens', r'text_decoder.embedding.word_embeddings'),
                (r'thinker.model.layers.(\d+).input_layernorm', r'text_decoder.decoder.layers.(\d+).input_layernorm'),
                (r'thinker.model.layers.(\d+).mlp.down_proj', r'text_decoder.decoder.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.model.layers.(\d+).post_attention_layernorm',
                 r'text_decoder.decoder.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.model.layers.(\d+).self_attn.o_proj',
                 r'text_decoder.decoder.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.lm_head', r'text_decoder.output_layer'),
                (r'thinker.model.norm', r'text_decoder.decoder.final_layernorm'),
                (r'thinker.audio_tower.layers.(\d+).fc1', r'audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1'),
                (r'thinker.audio_tower.layers.(\d+).fc2', r'audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2'),
                (r'thinker.audio_tower.layers.(\d+).final_layer_norm',
                 r'audio_encoder.encoder.blocks.layers.(\d+).pre_mlp_layernorm'),
                (r'thinker.audio_tower.layers.(\d+).self_attn.out_proj',
                 r'audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj'),
                (r'thinker.audio_tower.layers.(\d+).self_attn_layer_norm',
                 r'audio_encoder.encoder.blocks.layers.(\d+).input_layernorm'),
                (r'thinker.audio_tower.ln_post', r'audio_encoder.encoder.ln_post'),
                (r'thinker.audio_tower.proj', r'audio_encoder.encoder.proj'),
                (r'thinker.audio_tower.audio_bos_eos_token', r'audio_encoder.encoder.audio_bos_eos_token'),
                (r'thinker.audio_tower.conv1', r'audio_encoder.encoder.conv1'),
                (r'thinker.audio_tower.conv2', r'audio_encoder.encoder.conv2')
            )
        ),
    ]
    return ops


qwen2_5_omni_tp_patterns = {
    **qwen2_5_vl_tp_patterns,
    **{r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": RowSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": RowSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": ColSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": RowSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": RowSplit,
       r"audio_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": ColSplit,
       }
}


class ConvertVppMMConfigQwenOmni(ConvertVppMMConfig):

    def model_post_init(self, _context):
        from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig
        config = cast(Qwen2_5OmniConfig, self.hf_config.config)
        self.common_model_config.num_key_value_heads = config.thinker_config.text_config.num_key_value_heads
        self.common_model_config.vit_num_layers = config.thinker_config.vision_config.depth
        self.common_model_config.llm_num_layers = config.thinker_config.text_config.num_hidden_layers
        self.common_model_config.audio_num_layers = config.thinker_config.audio_config.num_hidden_layers
        self.common_model_config.tie_word_embeddings = config.thinker_config.text_config.tie_word_embeddings


class Qwen2_5_OmniConverter(Converter):
    """Qwen2.5Omni模型转换工具"""

    @staticmethod
    # 创建转换操作,加下划线之后命令行会自动忽略这条子命令
    def _create_ops(config: Any):
        from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig
        config = cast(Qwen2_5OmniConfig, config)
        vit_head_hidden_size = config.thinker_config.vision_config.hidden_size // config.thinker_config.vision_config.num_heads
        audio_head_hidden_size = config.thinker_config.audio_config.d_model // config.thinker_config.audio_config.encoder_attention_heads
        llm_head_hidden_size = config.thinker_config.text_config.hidden_size // config.thinker_config.text_config.num_attention_heads
        llm_q_size = llm_head_hidden_size * config.thinker_config.text_config.num_attention_heads // config.thinker_config.text_config.num_key_value_heads
        llm_kv_size = llm_head_hidden_size
        ops = create_qwen2_5_omni_ops(config.thinker_config.vision_config.num_heads,
                                      config.thinker_config.text_config.num_key_value_heads,
                                      config.thinker_config.audio_config.encoder_attention_heads,
                                      (vit_head_hidden_size,
                                       audio_head_hidden_size,
                                       llm_q_size,
                                       llm_kv_size)
                                      )
        config.tie_word_embeddings = config.thinker_config.text_config.tie_word_embeddings
        return ops, config

    @staticmethod
    def hf_to_mm(cfg: ConvertVppMMConfigQwenOmni):
        """huggingface模型转换mindspeed-mm模型权重"""
        ops, config = Qwen2_5_OmniConverter._create_ops(cfg.hf_config.config)

        hf_to_mm.convert_hf_to_mm(cfg, ops, qwen2_5_omni_tp_patterns, [vision_schema, text_schema, audio_schema])
        # 安全管控权限
        set_directory_permissions(cfg.mm_dir)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        ops, config = Qwen2_5_OmniConverter._create_ops(cfg.hf_config.config)
        mm_to_hf.convert_mm_to_hf(cfg, ops, qwen2_5_omni_tp_patterns, merge_source=True)
        # 安全管控权限
        set_directory_permissions(cfg.save_hf_dir)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass
