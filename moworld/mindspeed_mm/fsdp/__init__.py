import os
from .ops import apply_ops_patch

if os.getenv("NON_MEGATRON", "") == "true":
    apply_ops_patch()
