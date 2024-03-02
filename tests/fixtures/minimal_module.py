import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


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
def patchable_minimal_module(request, minimal_module, minimal_module_inputs):
    return PatchableGraph(
        minimal_module,
        ExtractionOptions(skip_compilation=getattr(request, "param", None) == "opaque"),
        minimal_module_inputs,
    )
