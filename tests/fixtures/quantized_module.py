import pytest
import torch

from graphpatch import PatchableGraph
from graphpatch.optional.bitsandbytes import AVAILABLE as BNB_AVAILABLE, Linear8bitLt

if BNB_AVAILABLE:

    class QuantizedModule(torch.nn.Module):
        _shape = (2, 3)

        def __init__(self):
            super().__init__()
            self.linear = Linear8bitLt(
                *QuantizedModule._shape, has_fp16_weights=False, threshold=6.0
            ).cuda()

        def forward(self, x):
            return self.linear(x)

    @pytest.fixture
    def quantized_module():
        return QuantizedModule()

    @pytest.fixture
    def quantized_module_inputs():
        return torch.ones(*QuantizedModule._shape, device="cuda", dtype=torch.float16).t()

    @pytest.fixture
    def patchable_quantized_module(quantized_module, quantized_module_inputs):
        return PatchableGraph(quantized_module, quantized_module_inputs)
