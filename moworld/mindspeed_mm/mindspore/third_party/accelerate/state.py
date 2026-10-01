# Copyright 2021 The HuggingFace Team. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import os
import torch
import mindspore

from accelerate.utils import (
    DistributedType,
)


def PartialState_prepare_backend_wrapper(func):
    """
    Wrapper to set the os.environ["LOCAL_RANK"] value to obtain the rank ID.
    """
    def wrapper(*args, **kwargs):
        os.environ["LOCAL_RANK"] = f"{mindspore.communication.get_local_rank()}"
        return func(*args, **kwargs)
    return wrapper


def PartialState_set_device(self):
    """
    Avoid the "set_device" error.
    """
    if self.device is not None:
        return
    if self.distributed_type == DistributedType.NO:
        self.device = torch.device("cpu") if self._cpu else self.default_device
        return
    device = str(self.distributed_type).split(".")[-1].replace("MULTI_", "").lower()
    if device not in ("cpu", "gpu", "mlu", "npu", "xpu", "xla"):
        raise ValueError(
            f"Can't set device for {self.distributed_type} ({device}), verify we should be calling `_set_device()` for it!"
        )
    if device == "xla":
        self.device = xm.xla_device()
    else:
        if device == "gpu":
            device = "cuda"
        self.device = torch.device(device, self.local_process_index)
    if self.device is not None:
        if device == "xpu":
            torch.xpu.set_device(self.device)
        elif device == "mlu":
            torch.mlu.set_device(self.device)
        # Avoid calling set_device() after initialization when using NPU or CUDA devices, as it may cause runtime errors.
