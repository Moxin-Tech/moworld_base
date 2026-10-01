from transformers.models.qwen2 import modeling_qwen2


def apply_cosyvoice_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for CosyVoice3 Model
    modeling_qwen2.Qwen2RMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
    modeling_qwen2.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
