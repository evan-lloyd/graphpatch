import pytest
from torch import ones
from torch.nn import Linear, Module, Sequential, ModuleList, ModuleDict

from graphpatch import ExtractionOptions, PatchableGraph


class ContainerModule(Module):
    _shape = (3, 3)

    def __init__(self):
        super().__init__()
        self.linear = Linear(*ContainerModule._shape)
        self.duped_linear = self.linear
        chained_linear = Linear(*ContainerModule._shape)
        self.sequential = Sequential(chained_linear, chained_linear, chained_linear, chained_linear)

    def forward(self, x):
        y = self.duped_linear(x)
        return self.sequential(y) + self.linear(y)


@pytest.fixture
def container_module():
    return ContainerModule()


@pytest.fixture
def container_module_inputs():
    return ones(*ContainerModule._shape).t()


@pytest.fixture
def patchable_container_module(request, container_module, container_module_inputs):
    return PatchableGraph(
        container_module,
        ExtractionOptions(skip_compilation=getattr(request, "param", None) == "opaque"),
        container_module_inputs,
    )
