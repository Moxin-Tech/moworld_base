# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""VAE global variables."""

_GLOBAL_AE_ARGS = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    if var is None:
        raise ValueError('{} is not initialized.'.format(name))


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    if var is not None:
        raise ValueError('{} is already initialized.'.format(name))


def set_ae_global_variables(args):
    global _GLOBAL_AE_ARGS

    _ensure_var_is_not_initialized(_GLOBAL_AE_ARGS, "args")
    _GLOBAL_AE_ARGS = args


def get_ae_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_AE_ARGS, "args")
    return _GLOBAL_AE_ARGS
