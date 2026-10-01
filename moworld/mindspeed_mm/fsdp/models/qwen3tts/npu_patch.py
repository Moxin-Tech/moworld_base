from .core.models import modeling_qwen3_tts


def apply_qwen3tts_npu_patch():
    from mindspeed_mm.fsdp.ops.npu_patch import npu_fused_operator
    # Patches for Qwen3TTS Model
    modeling_qwen3_tts.apply_rotary_pos_emb = npu_fused_operator.apply_transformers_rope_half_npu
    modeling_qwen3_tts.Qwen3TTSRMSNorm.forward = npu_fused_operator.rms_norm_forward_npu
