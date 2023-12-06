from typing import Any

import torch
from torch.nn.functional import linear


class Wrapped8BitLinear(torch.nn.Module):
    def __init__(self, CB: torch.Tensor, SCB: torch.Tensor, bias: torch.Tensor):
        super().__init__()

        if CB is None or SCB is None:
            raise ValueError("Must have CB/SCB values to wrap")

        self.CB = torch.nn.Parameter(CB, requires_grad=False)
        self.SCB = torch.nn.Parameter(SCB.to(torch.float16).unsqueeze(1))
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.to(torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> Any:
        upcast_weights = (self.CB * self.SCB) / 127
        return linear(x, upcast_weights, self.bias)
