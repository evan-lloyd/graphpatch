import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import PatchableGraph


class MinimalModule(Module):
    _shape = (2, 3)

    def __init__(self):
        super().__init__()
        self.linear = Linear(*MinimalModule._shape)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def minimal_module():
    return MinimalModule()


@pytest.fixture
def minimal_module_inputs():
    return ones(*MinimalModule._shape).t()


@pytest.fixture
def patchable_minimal_module(minimal_module, minimal_module_inputs):
    return PatchableGraph(minimal_module, minimal_module_inputs)
