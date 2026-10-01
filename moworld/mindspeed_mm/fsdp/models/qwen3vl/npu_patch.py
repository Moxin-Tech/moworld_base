from . import modeling_qwen3_vl_moe


def apply_qwen3vl_moe_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for Qwen3VL Model
    modeling_qwen3_vl_moe.apply_rotary_pos_emb_vision = npu_fused_operator.apply_transformers_vision_rope_half_npu
    modeling_qwen3_vl_moe.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
    modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_qwen3_vl_moe.Qwen3VLMoeTextExperts.forward = npu_fused_operator.fused_moe_forward_npu
