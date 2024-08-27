import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class VarargsModule(Module):
    _shape = (3, 3)

    def __init__(self):
        super().__init__()
        self.linear = Linear(*VarargsModule._shape)

    def forward(self, x, *foos, blah=3, **bars):
        result = self.linear(x)
        for f in foos:
            result += f
        result = self.linear(result + blah)
        for v in bars.values():
            result += v
        return result


@pytest.fixture
def varargs_module():
    return VarargsModule()


@pytest.fixture
def varargs_module_inputs():
    return ones(*VarargsModule._shape).t()


@pytest.fixture
def varargs_module_varargs(varargs_module_inputs):
    return (
        varargs_module_inputs.clone(),
        varargs_module_inputs.clone(),
        varargs_module_inputs.clone(),
    )


@pytest.fixture
def varargs_module_varkwargs(varargs_module_inputs):
    return {"a": varargs_module_inputs.clone(), "b": varargs_module_inputs.clone()}


@pytest.fixture
def patchable_varargs_module(
    request, varargs_module, varargs_module_inputs, varargs_module_varargs, varargs_module_varkwargs
):
    return PatchableGraph(
        varargs_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        varargs_module_inputs,
        *varargs_module_varargs,
        **varargs_module_varkwargs,
    )
