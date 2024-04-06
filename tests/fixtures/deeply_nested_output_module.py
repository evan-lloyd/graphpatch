import pytest
import torch
from torch.nn import Linear, Module, ModuleList

from graphpatch import ExtractionOptions, PatchableGraph


class C(Module):
    def __init__(self):
        super().__init__()
        self.c_linear = Linear(10, 10)

    def forward(self, c_inputs, inputs_2=None, inputs_3=None, inputs_4=None):
        return [
            self.c_linear(c_inputs),
            (self.c_linear(c_inputs + inputs_2), self.c_linear(inputs_3 + inputs_4)),
        ]


class B(Module):
    def __init__(self):
        super().__init__()
        self.b_linear = Linear(10, 10)
        self.c = C()

    def forward(self, b_inputs, a_inputs):
        b_outputs = self.c(
            b_inputs,
            inputs_3=torch.ones_like(b_inputs),
            inputs_2=b_inputs * 2,
            inputs_4=a_inputs,
        )
        b_outputs[0] = self.b_linear(b_outputs[1][0])
        return (((((b_outputs[1][0],),),),),)


class A(Module):
    def __init__(self):
        super().__init__()
        self.a_linear = Linear(10, 10)
        self.grandchildren_b = ModuleList([B() for _ in range(3)])

    def forward(self, a_inputs):
        b_outputs = a_inputs
        output_dict = []
        for i, b in enumerate(self.grandchildren_b):
            b_outputs = b(b_outputs, a_inputs)[0][0][0][0][0]
            output_dict.append(([i * a_inputs], b_outputs.clone()))
        b_outputs = [self.a_linear(b_outputs)]
        return (b_outputs, output_dict)  # {k: v for k, v in output_dict})


class DeeplyNestedOutputModule(Module):
    """Nonsensical module with arbitrary nesting of intermediate outputs. Utility lies in testing
    that our graph extraction maintains the original signature, since otherwise we would fail to
    run forward() at all.
    """

    _shape = (10, 10)

    def __init__(self):
        super().__init__()
        self.child_a = A()
        self.linear = Linear(*DeeplyNestedOutputModule._shape)

    def forward(self, x):
        a_outputs, a_dict = self.child_a(x)
        a_outputs = self.linear(a_outputs[0])
        return ((a_outputs,), a_dict, {"nested_dict": [(self.linear(a_outputs + 2),)]})


@pytest.fixture
def deeply_nested_output_module():
    return DeeplyNestedOutputModule()


@pytest.fixture
def deeply_nested_output_module_inputs():
    return torch.ones((10, 10))


@pytest.fixture
def patchable_deeply_nested_output_module(
    request, deeply_nested_output_module, deeply_nested_output_module_inputs
):
    return PatchableGraph(
        deeply_nested_output_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        deeply_nested_output_module_inputs,
    )
