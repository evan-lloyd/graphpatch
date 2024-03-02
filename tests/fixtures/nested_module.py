import pytest
import torch
from torch.nn import Linear, Module, ModuleList

from graphpatch.patchable_graph import ExtractionOptions, PatchableGraph


class C(Module):
    def __init__(self):
        super().__init__()
        self.c_linear = Linear(100, 100)

    def forward(self, c_inputs, inputs_2=None, inputs_3=None, inputs_4=None):
        return self.c_linear(c_inputs + inputs_2 + inputs_3 + inputs_4)


class B(Module):
    def __init__(self):
        super().__init__()
        self.b_linear = Linear(100, 100)
        self.c = C()

    def forward(self, b_inputs, a_inputs):
        b_outputs = self.c(
            b_inputs,
            inputs_3=torch.ones_like(b_inputs),
            inputs_2=b_inputs * 2,
            inputs_4=a_inputs,
        )
        b_outputs = self.b_linear(b_outputs)
        return b_outputs


class A(Module):
    def __init__(self):
        super().__init__()
        self.a_linear = Linear(100, 100)
        self.grandchildren_b = ModuleList([B() for _ in range(3)])

    def forward(self, a_inputs):
        b_outputs = a_inputs
        for b in self.grandchildren_b:
            b_outputs = b(b_outputs, a_inputs)
        b_outputs = self.a_linear(b_outputs)
        return b_outputs


class NestedModule(Module):
    def __init__(self):
        super().__init__()
        self.root_linear = Linear(100, 100)
        self.child_a = A()

    def forward(self, root_inputs):
        a_outputs = self.child_a(root_inputs)
        a_outputs = self.root_linear(a_outputs)
        return a_outputs


@pytest.fixture
def nested_module():
    return NestedModule()


@pytest.fixture
def nested_module_inputs():
    return torch.ones(1, 100)


@pytest.fixture
def patchable_nested_module(request, nested_module, nested_module_inputs):
    return PatchableGraph(
        nested_module,
        ExtractionOptions(skip_compilation=getattr(request, "param", None) == "opaque"),
        nested_module_inputs,
    )
