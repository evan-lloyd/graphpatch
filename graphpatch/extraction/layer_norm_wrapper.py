from typing import Tuple

import torch
from torch import Tensor
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import LayerNorm, Module, Parameter
from torch.nn.functional import layer_norm as torch_layer_norm


@allow_in_graph
def layer_norm(
    input: Tensor, normalized_shape: Tuple[int, ...], weight: Parameter, bias: Parameter, eps: float
):
    if isinstance(torch.empty(0), FakeTensor):
        return torch.zeros(input.shape, device=input.device, dtype=input.dtype)
    return torch_layer_norm(input, normalized_shape, weight, bias, eps)


class LayerNormWrapper(Module):
    def __init__(self, original: LayerNorm):
        super().__init__()
        self.normalized_shape = original.normalized_shape
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps

    def forward(self, input: Tensor):
        return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
