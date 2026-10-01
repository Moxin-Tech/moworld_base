import torch.nn as nn

from mindspeed_mm.models.common.activations import get_activation_layer, Sigmoid
from tests.ut.utils import judge_expression


class TestActivation:
    """
    Test activation basic function.
    """
    def test_activation_when_get_right_act_type(self):
        act_type = "relu"
        res = get_activation_layer(act_type)
        judge_expression(isinstance(res(), nn.ReLU))
        act_type = "gelu"
        res = get_activation_layer(act_type)
        judge_expression(isinstance(res(), nn.GELU))
        act_type = "swish"
        res = get_activation_layer(act_type)
        judge_expression(isinstance(res(), Sigmoid))

    def test_unknown_activation(self):
        try:
            get_activation_layer("invalid_act")
            judge_expression(False)
        except ValueError:
            judge_expression(True)
