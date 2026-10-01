from transformers.models.qwen3 import modeling_qwen3


def apply_funasr_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for FunASR Model
    modeling_qwen3.Qwen3RMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_qwen3.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
    modeling_qwen3.Qwen3MLP.forward = npu_fused_operator.silu_forward_npu
