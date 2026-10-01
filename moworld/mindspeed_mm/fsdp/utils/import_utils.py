IS_FLA_NPU_AVAILABLE = False
try:
    import fla_npu

    IS_FLA_NPU_AVAILABLE = True
except Exception:
    IS_FLA_NPU_AVAILABLE = False

IS_TRITON_AVAILABLE = False
try:
    import triton

    IS_TRITON_AVAILABLE = True
except Exception:
    IS_TRITON_AVAILABLE = False
