from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, float16
from torch.nn import Module, Parameter

from .. import hacks
from ..optional.bitsandbytes import (
    Linear8bitLt,
    MatmulLtState,
    get_tile_inds,
    matmul,
    undo_layout,
)


@hacks.allow_in_graph  # type: ignore
def matmul_8bit(x: Tensor, weight: Tensor, bias: Tensor, threshold: float) -> Any:
    # bitsandbytes matmul doesn't work with FakeTensors, so just return a tensor of the right shape.
    if hacks.in_fake_mode():
        return torch.zeros(*x.shape[:-1], weight.shape[0], device=x.device, dtype=float16)
    state = MatmulLtState()
    state.has_fp16_weights = True
    state.threshold = threshold
    return matmul(x, weight, bias=bias, state=state).to(float16)


class Wrapped8BitLinear(Module):
    def __init__(self, original: Linear8bitLt):
        super().__init__()
        # CB and SCB get deleted when running inference, so we may have to recompute them.
        if original.weight.CB is not None:
            CB = original.weight.CB
        else:
            CB = undo_layout(
                original.weight, get_tile_inds(original.state.formatB, original.weight.device)
            )[: original.out_features, : original.in_features]
        if original.weight.SCB is not None:
            SCB = original.weight.SCB
        else:
            SCB = original.state.SCB
        self.threshold = original.state.threshold
        # It doesn't make sense with the current logic to compute gradients for the quantization
        # parameters. The user is expected instead to patch the "weight" node within the forward
        # computation.
        self.CB = Parameter(CB, requires_grad=False)
        self.SCB = Parameter(SCB.to(float16).unsqueeze(1), requires_grad=False)
        if original.bias is not None:
            self.bias = Parameter(original.bias.to(float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Any:
        weight = (self.CB * self.SCB) / 127
        return matmul_8bit(x, weight, self.bias, self.threshold)

    def __deepcopy__(self, memo: Any) -> "Wrapped8BitLinear":
        """Prevents an error when torch attempts to fakify our 8-bit parameters, which fails because
        they are a Tensor subclass."""
        if hacks.in_fake_mode():
            return self
        new_instance = type(self).__new__(type(self))
        Module.__init__(new_instance)
        new_instance.CB = deepcopy(self.CB, memo)
        new_instance.SCB = deepcopy(self.SCB, memo)
        new_instance.threshold = deepcopy(self.threshold, memo)
        if self.bias is not None:
            new_instance.bias = Parameter(deepcopy(self.bias, memo))
        else:
            new_instance.register_parameter("bias", None)
        return new_instance
