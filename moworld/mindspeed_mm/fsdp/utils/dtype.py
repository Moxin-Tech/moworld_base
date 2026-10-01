# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
import torch


def get_dtype(dtype: str):
    DTYPE_MAP = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
        'fp64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64
    }
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return DTYPE_MAP[dtype]
