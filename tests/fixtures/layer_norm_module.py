import pytest
from torch import ones
from torch.nn import LayerNorm, Module

from graphpatch import ExtractionOptions, PatchableGraph


class LayerNormModule(Module):
    _shape = (2, 3)

    def __init__(self):
        super().__init__()
        self.ln = LayerNorm(2)

    def forward(self, x):
        return self.ln(x)


@pytest.fixture
def layer_norm_module():
    return LayerNormModule()


@pytest.fixture
def layer_norm_module_inputs():
    return ones(*LayerNormModule._shape).t()


@pytest.fixture
def patchable_layer_norm_module(request, layer_norm_module, layer_norm_module_inputs):
    return PatchableGraph(
        layer_norm_module,
        ExtractionOptions(skip_compilation=getattr(request, "param", None) == "opaque"),
        layer_norm_module_inputs,
    )
