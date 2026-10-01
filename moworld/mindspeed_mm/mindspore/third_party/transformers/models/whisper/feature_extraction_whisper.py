# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import warnings


def _torch_extract_fbank_features_wrapper(func):
    """
    Wrapper to replace _torch_extract_fbank_features with _np_extract_fbank_features,
    since torch.stft is not supported.
    """
    def wrapper(*args, **kwargs):
        self = args[0]
        warnings.warn(f"`_torch_extract_fbank_features` is not currently supported; using `_np_extract_fbank_features` here instead.")
        return self._np_extract_fbank_features(*args[1:], **kwargs)
    return wrapper
