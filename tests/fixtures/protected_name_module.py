import pytest
from torch import ones
from torch.nn import Linear, Module

from graphpatch import ExtractionOptions, PatchableGraph


class ProtectedNameModule(Module):
    _shape = (2, 3)

    def __init__(self):
        super().__init__()
        self._code = Linear(*ProtectedNameModule._shape)
        self.bar = (5.0, (6.0, (7.09)))

    def forward(self, _shape, _code=5, sub_shape=7):
        return self._code(_shape + _code + self._shape[0] + self.bar[1][1] + sub_shape)


@pytest.fixture
def protected_name_module():
    return ProtectedNameModule()


@pytest.fixture
def protected_name_module_inputs():
    return ones(*ProtectedNameModule._shape).t()


@pytest.fixture
def patchable_protected_name_module(request, protected_name_module, protected_name_module_inputs):
    return PatchableGraph(
        protected_name_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        protected_name_module_inputs,
    )
