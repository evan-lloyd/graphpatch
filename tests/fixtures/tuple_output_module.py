import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class TupleOutputModule(Module):
    _shape = (2, 3)

    def __init__(self):
        super().__init__()
        self.linear = Linear(*TupleOutputModule._shape)

    def forward(self, x):
        return (self.linear(x), self.linear(x + 1))


@pytest.fixture
def tuple_output_module():
    return TupleOutputModule()


@pytest.fixture
def tuple_output_module_inputs():
    return ones(*TupleOutputModule._shape).t()


@pytest.fixture
def patchable_tuple_output_module(request, tuple_output_module, tuple_output_module_inputs):
    return PatchableGraph(
        tuple_output_module,
        ExtractionOptions(skip_compilation=getattr(request, "param", None) == "opaque"),
        tuple_output_module_inputs,
    )
