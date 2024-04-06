import pytest
import torch
from torch.nn import Linear

from graphpatch import ExtractionOptions, PatchableGraph
from graphpatch.optional.bitsandbytes import AVAILABLE as BNB_AVAILABLE, Linear8bitLt

if BNB_AVAILABLE:

    class QuantizedModule(torch.nn.Module):
        _shape = (2, 3)

        def __init__(self):
            super().__init__()
            object.__setattr__(self, "original_linear", Linear(*QuantizedModule._shape))
            self.linear = Linear8bitLt(
                *QuantizedModule._shape, has_fp16_weights=False, threshold=6.0
            )
            self.linear.weight.data = self.original_linear.weight.data
            self.linear.bias.data = self.original_linear.bias.data
            self.linear.cuda()

        def forward(self, x):
            return self.linear(x)

    @pytest.fixture
    def quantized_module():
        return QuantizedModule()

    @pytest.fixture
    def quantized_module_inputs():
        return torch.ones(*QuantizedModule._shape, device="cuda", dtype=torch.float16).t()

    @pytest.fixture
    def patchable_quantized_module(request, quantized_module, quantized_module_inputs):
        return PatchableGraph(
            quantized_module,
            ExtractionOptions(
                skip_compilation=getattr(request, "param", None) == "opaque",
                error_on_compilation_failure=True,
            ),
            quantized_module_inputs,
        )
