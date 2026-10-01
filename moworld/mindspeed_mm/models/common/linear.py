import torch
import torch.nn as nn
from torch import Tensor


class MatmulAddLinear(nn.Linear):
    def forward(self, input_tensor: Tensor) -> Tensor:
        output = torch.matmul(input_tensor, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output
