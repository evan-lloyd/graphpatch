import io

import pytest
import torch

from graphpatch import PatchableGraph
from graphpatch.extraction import ExtractionOptions
from graphpatch.optional.accelerate import ModelHook, add_hook_to_module

from .util import (
    assert_patchable_graphs_identical,
    assert_results_identical,
    requires_accelerate,
    requires_bitsandbytes,
    requires_gpu,
    requires_multi_gpu,
    requires_transformers,
)


def _roundtrip(module):
    buffer = io.BytesIO()
    module.save(buffer)
    buffer.seek(0)
    return torch.load(buffer)


def _serialization_asserts(original_module, deserialized_module, test_inputs):
    # After round trip...
    # Object should match
    assert_patchable_graphs_identical(original_module, deserialized_module)

    # ...forward() and backward() should work, and give the same result
    assert_results_identical(original_module, deserialized_module, test_inputs)


def test_torch_save_raises(patchable_minimal_module):
    with pytest.raises(ValueError):
        buffer = io.BytesIO()
        torch.save(patchable_minimal_module, buffer)


def test_minimal_module_serialization(patchable_minimal_module, minimal_module_inputs):
    deserialized = _roundtrip(patchable_minimal_module)
    _serialization_asserts(patchable_minimal_module, deserialized, minimal_module_inputs)


def test_opaque_module_serialization(minimal_module, minimal_module_inputs):
    patchable_minimal_module = PatchableGraph(
        minimal_module,
        ExtractionOptions(classes_to_skip_compiling={torch.nn.Linear}),
        minimal_module_inputs,
    )
    deserialized = _roundtrip(patchable_minimal_module)
    _serialization_asserts(patchable_minimal_module, deserialized, minimal_module_inputs)


@requires_accelerate
def test_layer_norm_module_serialization(patchable_layer_norm_module, layer_norm_module_inputs):
    # TODO: remove once we make layer norm compilable!

    # Simulate behavior we get trying to serialize GPT2-XL; accelerate's hook makes the module
    # unpicklable.
    add_hook_to_module(patchable_layer_norm_module._graph_module.ln, hook := ModelHook())
    deserialized = _roundtrip(patchable_layer_norm_module)
    assert patchable_layer_norm_module._graph_module.ln._hf_hook is hook
    _serialization_asserts(patchable_layer_norm_module, deserialized, layer_norm_module_inputs)


def test_attribute_module_serialization(patchable_attribute_module, attribute_module_inputs):
    # Tests handling of non-state attributes on module.
    deserialized = _roundtrip(patchable_attribute_module)
    _serialization_asserts(patchable_attribute_module, deserialized, attribute_module_inputs)


def test_nested_module_serialization(patchable_nested_module, nested_module_inputs):
    deserialized = _roundtrip(patchable_nested_module)
    _serialization_asserts(patchable_nested_module, deserialized, nested_module_inputs)


def test_tuple_output_module_serialization(
    patchable_tuple_output_module, tuple_output_module_inputs
):
    # Tests handling of cloned GraphModules, due to original model making multiple calls to a
    # submodule.
    deserialized = _roundtrip(patchable_tuple_output_module)
    _serialization_asserts(patchable_tuple_output_module, deserialized, tuple_output_module_inputs)


def test_deeply_nested_output_module_serialization(
    patchable_deeply_nested_output_module, deeply_nested_output_module_inputs
):
    deserialized = _roundtrip(patchable_deeply_nested_output_module)
    _serialization_asserts(
        patchable_deeply_nested_output_module, deserialized, deeply_nested_output_module_inputs
    )


@requires_transformers
def test_pretrained_module_serialization(patchable_pretrained_module, pretrained_module_inputs):
    deserialized = _roundtrip(patchable_pretrained_module)
    _serialization_asserts(patchable_pretrained_module, deserialized, pretrained_module_inputs)


@requires_multi_gpu
@requires_transformers
@requires_accelerate
def test_multiple_device_serialization(
    patchable_accelerate_pretrained_module, accelerate_pretrained_module_inputs, mocker
):
    # Make sure we're *actually* testing on multiple GPU's
    assert len({p.device for p in patchable_accelerate_pretrained_module.parameters()}) >= 2
    deserialized = _roundtrip(patchable_accelerate_pretrained_module)
    _serialization_asserts(
        patchable_accelerate_pretrained_module,
        deserialized,
        accelerate_pretrained_module_inputs,
    )


@requires_gpu
@requires_transformers
@requires_accelerate
@requires_bitsandbytes
def test_quantized_pretrained_module_serialization(
    patchable_quantized_pretrained_module, quantized_pretrained_module_inputs
):
    deserialized = _roundtrip(patchable_quantized_pretrained_module)
    _serialization_asserts(
        patchable_quantized_pretrained_module,
        deserialized,
        quantized_pretrained_module_inputs.to(torch.float16),
    )
