import pytest
from torch import ones
from torch._dynamo import graph_break
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class GraphBreakModule(Module):
    _shape = (3, 3)
    bar = 5
    shadowed_class_var = 1

    def __init__(self):
        super().__init__()
        self.linear = Linear(*GraphBreakModule._shape)
        self.instance_value = 7
        self.shadowed_class_var = 8
        self.unused_submodule = Linear(*GraphBreakModule._shape)

    def member_function(self, n):
        return ones(self._shape) - n

    def forward(self, x, foo=3):
        x = self.linear(x)
        y = self.linear(x)
        z = self.linear(y)

        graph_break()

        y = x + y + z + self.bar + self.instance_value + self.shadowed_class_var

        graph_break()

        return y + 5 * self.member_function(foo)


@pytest.fixture
def graph_break_module():
    return GraphBreakModule()


@pytest.fixture
def graph_break_module_inputs():
    return ones(*GraphBreakModule._shape).t()


@pytest.fixture
def patchable_graph_break_module(request, graph_break_module, graph_break_module_inputs):
    return PatchableGraph(
        graph_break_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=False,
        ),
        graph_break_module_inputs,
    )
