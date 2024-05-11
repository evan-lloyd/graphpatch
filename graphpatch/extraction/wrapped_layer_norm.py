from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, Parameter
from torch.nn.functional import layer_norm as torch_layer_norm

from .. import hacks


@hacks.allow_in_graph  # type: ignore
def layer_norm(
    input: Tensor, normalized_shape: Tuple[int, ...], weight: Parameter, bias: Parameter, eps: float
) -> Tensor:
    if hacks.in_fake_mode():
        return torch.zeros(input.shape, device=input.device, dtype=input.dtype)
    return torch_layer_norm(input, normalized_shape, weight, bias, eps)


class WrappedLayerNorm(Module):
    def __init__(self, original: LayerNorm):
        super().__init__()
        self.normalized_shape = original.normalized_shape
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps

    def forward(self, input: Tensor) -> Any:
        return layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
