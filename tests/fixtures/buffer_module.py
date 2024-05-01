import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class BufferModule(Module):
    _shape = (2, 3)

    def __init__(self):
        super().__init__()
        self.linear = Linear(*BufferModule._shape)
        self.register_buffer("buffer", ones(*([BufferModule._shape[1]] * 2)))

    def forward(self, x):
        return self.linear(x) + self.buffer


@pytest.fixture
def buffer_module():
    return BufferModule()


@pytest.fixture
def buffer_module_inputs():
    return ones(*BufferModule._shape).t()


@pytest.fixture
def patchable_buffer_module(request, buffer_module, buffer_module_inputs):
    return PatchableGraph(
        buffer_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        buffer_module_inputs,
    )
