import pytest
from torch import ones
from torch.nn import Linear, Module
from torch._dynamo import graph_break

from graphpatch import PatchableGraph


class GraphBreakModule(Module):
    _shape = (3, 3)
    bar = 5

    def __init__(self):
        super().__init__()
        self.linear = Linear(*GraphBreakModule._shape)

    def member_function(self, n):
        return ones(self._shape) - n

    def forward(self, x, foo=3):
        x = self.linear(x)
        y = self.linear(x)
        z = self.linear(y)

        graph_break()

        y = x + y + z + self.bar

        graph_break()

        return y + 5 * self.member_function(foo)


@pytest.fixture
def graph_break_module():
    return GraphBreakModule()


@pytest.fixture
def graph_break_module_inputs():
    return ones(*GraphBreakModule._shape).t()


@pytest.fixture
def patchable_graph_break_module(graph_break_module, graph_break_module_inputs):
    return PatchableGraph(graph_break_module, graph_break_module_inputs)
