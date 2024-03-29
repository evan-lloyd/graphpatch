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
        chained_sequential = Sequential(self.duped_linear, self.linear)
        self.sequential = Sequential(
            chained_linear, self.linear, chained_sequential, chained_linear
        )
        self.module_list = ModuleList(
            [Linear(*ContainerModule._shape), Linear(*ContainerModule._shape)]
        )
        self.module_dict = ModuleDict(
            {
                "foo": self.linear,
                "bar": ModuleDict({"baz": ModuleList([self.linear, self.linear, self.sequential])}),
            }
        )

    def forward(self, x):
        y = self.duped_linear(x) - self.module_dict["foo"](x)
        for i in range(3):
            y += self.module_dict["bar"]["baz"][i](x) * self.linear(x)
        y += self.module_list[0](x)
        y += self.module_list[1](x)
        return self.sequential(y) + self.linear(y) + self.module_list[0](y)


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
