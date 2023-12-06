from graphpatch.wrapped_8bit_linear import Wrapped8BitLinear

from .util import assert_outputs_identical, requires_bitsandbytes, requires_gpu


@requires_gpu
@requires_bitsandbytes
def test_quantization_wrapper(quantized_module, quantized_module_inputs):
    wrapped = Wrapped8BitLinear(
        quantized_module.linear.weight.CB,
        quantized_module.linear.weight.SCB,
        quantized_module.linear.bias,
    )
    assert_outputs_identical(
        quantized_module.linear, wrapped, quantized_module_inputs, tolerance=0.1
    )
