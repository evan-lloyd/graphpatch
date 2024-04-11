from typing import Any, Optional

import torch
from torch import Tensor, float16
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import Module, Parameter

from ..optional.accelerate import add_hook_to_module
from ..optional.bitsandbytes import (
    Linear8bitLt,
    MatmulLtState,
    get_tile_inds,
    matmul,
    undo_layout,
)


@allow_in_graph
def matmul_8bit(x, weight, bias, threshold):
    # bitsandbytes matmul doesn't work with FakeTensors, so just return a tensor of the right shape.
    if isinstance(torch.empty(0), FakeTensor):
        return torch.zeros(*x.shape[:-1], weight.shape[0], device=x.device, dtype=float16)
    state = MatmulLtState()
    state.has_fp16_weights = True
    state.threshold = threshold
    return matmul(x, weight, bias=bias, state=state).to(float16)


class Wrapped8BitLinear(Module):
    def __init__(self, CB, SCB, bias: Optional[Tensor], threshold: float):
        super().__init__()
        self.threshold = threshold
        # It doesn't make sense with the current logic to compute gradients for the quantization
        # parameters. The user is expected instead to patch the "weight" node within the forward
        # computation.
        self.CB = Parameter(CB, requires_grad=False)
        self.SCB = Parameter(SCB.to(float16).unsqueeze(1), requires_grad=False)
        if bias is not None:
            self.bias = Parameter(bias.to(float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Any:
        weight = (self.CB * self.SCB) / 127
        return matmul_8bit(x, weight, self.bias, self.threshold)


def maybe_wrap(module: Module) -> Module:
    if not isinstance(module, Linear8bitLt):
        return module
    # CB and SCB get deleted when running inference, so we may have to recompute them.
    CB = (
        module.weight.CB
        or undo_layout(module.weight, get_tile_inds(module.state.formatB, module.weight.device))[
            : module.out_features, : module.in_features
        ]
    )
    SCB = module.weight.SCB or module.state.SCB
    wrapped = Wrapped8BitLinear(CB, SCB, module.bias, module.state.threshold)
    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        add_hook_to_module(wrapped, hook)
    return wrapped
