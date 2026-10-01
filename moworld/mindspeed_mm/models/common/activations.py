import torch
import torch.nn as nn


def get_activation_layer(act_type):
    """get activation layer

    Args:
        act_type (str): the activation type

    Returns:
        torch.nn.functional: the activation layer
    """
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    elif act_type == "swish":
        return Sigmoid
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


class Sigmoid:
    @staticmethod
    def __call__(x):
        # swish
        return x * torch.sigmoid(x)
