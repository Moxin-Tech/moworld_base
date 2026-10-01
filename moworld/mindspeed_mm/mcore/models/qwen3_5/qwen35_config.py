from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Any, Dict, Union, Callable

import torch
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import print_rank_0


@dataclass
class MindSpeedArgsRequired:
    """Base configuration for MindSpeed Core, MindSpeed Core will check if all the required args are provided, and raise error if not."""
    # Flash attention
    pre_tockens: int = 2147483647
    next_tockens: int = 2147483647
    sparse_mode: int = 0
    use_flash_attn: bool = True
    use_fusion_attn_v2: bool = True
    gemm_gradient_accumulation_fusion: bool = False

    ema_decay: float = 0.9999
    noop_layers: bool = False
    moe_fb_overlap: bool = False
    transformer_impl: str = "local"  # local or transformer_engine
    use_gmm_fp8: bool = False
    context_parallel_kv_cache_policy: bool = False
    context_parallel_cache_interval: int = 0
    use_ulysses_allgather_kv: bool = False
    use_fused_rotary_pos_emb: bool = False
    use_fused_moe_token_permute_and_unpermute: bool = False
    op_cal_tflops: bool = False
    recompute_activation_function_num_layers: int = 0
    recompute_norm_num_layers: int = 0
    unaligned_linear: bool = False
    recompute_in_bubble: bool = False
    recompute_in_advance: bool = False
    pipeline_num_transformer_layers: Any = None
    schedules_method: str = "None"
    dist_train: bool = False
    tokenizer_name_or_path: str = None
    param_and_grad_buffer_pad: bool = False
    reuse_fp32_param: bool = False
    smart_swap: bool = False

    # moe
    moe_tp_extend_ep: bool = False
    n_shared_experts: Any = None
    moe_allgather_overlap_comm: bool = False
    moe_alltoall_overlap_comm: bool = False
    moe_unperm2_mem_optim_swap: bool = False
    moe_alltoall_mc2: bool = False
    enable_fine_grained_expert_placement: bool = False
    fix_router: bool = False

    # optimizer
    optimizer_selection: str = "fused_torch_adamw"
    virtual_optimizer: Any = None
    compress_dense: bool = False
    compress_activation: str = ""
    compress_optimizer: str = "disable"

    # alibi
    alibi_fusion_attn_type: Any = None
    multi_head_latent_attention: bool = False
    experimental_attention_variant: str = ""
    use_fused_lightning_indexer: bool = False
    use_fused_sparse_flash_attention: bool = False
    use_fused_lightning_indexer_kl_loss: bool = False
    auto_settings: bool = False
    moe_zero_memory_num_layers: int = None
    moe_zero_memory: str = "disable"

    qk_l2_norm: bool = False
    batch_invariant_mode: bool = False


@dataclass
class Qwen3_5MoEModelConfig(TransformerConfig, MindSpeedArgsRequired):
    """
    TransformerConfig for Qwen 3.5 VL (Vision-Language) Models.

    Qwen 3.5 combines a hybrid GDN (Gated DeltaNet) + Gated Attention language model
    architecture (like Qwen3-Next) with a vision encoder (similar to Qwen3-VL) and
    Mixture of Experts (MoE) with shared experts.

    Key Architecture Details (397B-A17B):
    - 60 layers: 15 groups × (3 GDN-MoE + 1 Attention-MoE)
    - Hidden dim: 4096, Token Embedding: 248320
    - GDN: 16 QK heads, 64 V heads, head_dim=128
    - Gated Attention: 32 Q heads, 2 KV heads, head_dim=256
    - MoE: 512 experts, 10 routed + 1 shared, expert dim=1024
    - mRoPE with sections [11, 11, 10], rope_theta=10,000,000
    - partial_rotary_factor=0.25

    Note: num_query_groups corresponds to num_key_value_heads in HF config (for
    standard Gated Attention layers). GDN layers have separate head counts.
    """

    use_te: bool = False
    fp16_lm_cross_entropy: bool = False
    pipeline_dtype: torch.dtype = torch.bfloat16

    # =========================================================================
    # Hybrid Architecture (Qwen3-Next style)
    # =========================================================================
    layernorm_zero_centered_gamma: bool = True
    linear_attention_freq: int | list[int] = 4  # 1 standard attention per 4 layers

    # --- Gated DeltaNet (GDN) parameters ---
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64  # 64 V heads for GDN in 397B model

    # =========================================================================
    # MoE parameters
    # =========================================================================
    num_moe_experts: int = 512
    moe_ffn_hidden_size: int = 1024
    moe_router_topk: int = 10  # 10 routed experts per token
    moe_shared_expert_gate: bool = True
    moe_router_load_balancing_type: str = "aux_loss"
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3
    use_fused_moe_token_permute_and_unpermute: bool = False

    # =========================================================================
    # GDN implementation choice (PyTorch vs. custom Triton/AscendC kernels)
    # =========================================================================
    causal_conv1d_implementation: str = "torch"
    gdn_implementation: str = "torch"

    # =========================================================================
    # Common LLM parameters
    # =========================================================================
    share_embeddings_and_output_weights: bool = False
    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True
    kv_channels: int | None = 256  # head_dim for standard Gated Attention
    num_query_groups: int = 2  # KV heads for standard Gated Attention
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    rotary_base: float = 10000000.0  # rope_theta from HF config
    rotary_percent: float = 0.25  # partial_rotary_factor from HF config
    vocab_size: int = 248320
    seq_length: int = 262144  # 262K native context length
    attention_output_gate: bool = True  # Whether to apply output gate to the attention layers
    pipeline_model_parallel_layout: Optional[Union[str, list]] = None

    # =========================================================================
    # VL-specific parameters
    # =========================================================================

    # Position embedding: Qwen3.5 uses multimodal rope (mRoPE)
    position_embedding_type: str = "mrope"
    # Qwen3.5 mRoPE section is [11, 11, 10] (different from Qwen3-VL's [24, 20, 20])
    # because partial_rotary_factor=0.25, so RoPE dim = 256*0.25 = 64, with sections [11,11,10]
    # for [temporal, height, width] summing to 32 (half of 64 rotary dim).
    max_position_embeddings: int = 262144
    mrope_section: List[int] = field(default_factory=lambda: [11, 11, 10])
    apply_rotary_pos_emb_in_fp32: bool = False

    # Vision-Language token IDs
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    bos_token_id: int = 248045
    eos_token_id: int = 248046

    # =========================================================================
    # Freeze options for fine-tuning
    # =========================================================================
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # =========================================================================
    # Performance
    # =========================================================================
    fine_grained_activation_offloading: bool = False
    """If True, offload the input of the specified modules to the CPU.
    Fine-grained activation offloading is a module-level offloading method
    instead of a layer-level offloading method like cpu_offloading."""
    bias_activation_fusion: bool = False
    vision_dp_when_cp: bool = False

    # Heterogeneous dist checkpoint (needed for hybrid architecture)
    hetereogenous_dist_checkpoint: bool = True  # Megatron spelling error, consistent with Megatron to avoid load failure
    mtp_num_layers: Optional[int] = None
    no_rope_freq: Optional[Union[int, List[int]]] = None


@dataclass
class Qwen3_5VisionConfig(TransformerConfig, MindSpeedArgsRequired):
    use_te: bool = False
    pipeline_dtype: torch.dtype = torch.bfloat16

    num_layers: int = 27
    hidden_size: int = 1152
    num_attention_heads: int = 16
    ffn_hidden_size: int = 4303
    kv_channels: int = 72
    num_query_groups: int = 16  # no GQA
    activation_func: partial(F.gelu, approximate="tanh")
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    apply_rotary_pos_emb_in_fp32: bool = True
    attention_output_gate: bool = False  # Whether to apply output gate to the attention layers

    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    add_qkv_bias: bool = True
    add_bias_linear: bool = True
    add_bias_conv: bool = True

    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    bias_activation_fusion: bool = False  # no swiglu, set false
    bias_dropout_fusion: bool = False  # no dropout, set false
    attention_softmax_in_fp32: bool = True
    normalization: str = "LayerNorm"
    apply_rope_fusion: bool = False

    moe_zero_memory: str = "disable"
    moe_zero_memory_num_layers: Optional[int] = None
    no_rope_freq: Optional[Union[int, List[int]]] = None

    fine_grained_activation_offloading: bool = False
    """If True, offload the input of the specified modules to the CPU.
    Fine-grained activation offloading is a module-level offloading method
    instead of a layer-level offloading method like cpu_offloading."""


def get_qwen3_5_vision_config(config: Dict):
    return Qwen3_5VisionConfig(**config["vision_encoder"])


def get_qwen3_5_llm_config(config: Dict):
    model_id = config.pop("model_id")

    if model_id == "qwen3_5_moe":
        print_rank_0(f"language config: {config}")
        llm_config = Qwen3_5MoEModelConfig(**config)

    return llm_config
