import os
if os.getenv("NON_MEGATRON", "") != "true":
    from mindspeed_mm.tasks.finetune.lora import lora_patch
    from mindspeed_mm.patchs import validate_args_patch
