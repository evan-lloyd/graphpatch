import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class AttributeModule(Module):
    """Minimal reproduction of a model containing non-state attributes, to mimic an edge case we
    have to handle with LLamaRMSNorm's variance_epsilon. Annoyingly, torch.compile() likes to
    convert this kind of value into a constant, but only sometimes, for values in certain ranges.
    ints and values like 1e-3 get converted to constant, but 1e-4 and smaller get retained as
    attributes.
    """

    _shape = (2, 3)

    def __init__(self, attribute_val=1e-6):
        super().__init__()
        self.linear = Linear(*AttributeModule._shape)
        self.attribute_to_serialize = attribute_val

    def forward(self, x):
        return self.linear(x + self.attribute_to_serialize)


@pytest.fixture
def attribute_module():
    return AttributeModule()


@pytest.fixture
def attribute_module_inputs():
    return ones(*AttributeModule._shape).t()


@pytest.fixture
def patchable_attribute_module(request, attribute_module, attribute_module_inputs):
    return PatchableGraph(
        attribute_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        attribute_module_inputs,
    )
