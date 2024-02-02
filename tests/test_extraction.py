from graphpatch.graph_extraction import extract

from .util import (
    assert_outputs_identical,
    requires_accelerate,
    requires_bitsandbytes,
    requires_gpu,
    requires_multi_gpu,
    requires_transformers,
    validate_node_meta,
)


def test_extract_minimal_module(minimal_module, minimal_module_inputs):
    graph_module, meta = extract(minimal_module, minimal_module_inputs)

    validate_node_meta(meta, graph_module)
    assert_outputs_identical(minimal_module, graph_module, minimal_module_inputs)


def test_extract_nested_module(nested_module, nested_module_inputs):
    graph_module, meta = extract(nested_module, nested_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(nested_module, graph_module, nested_module_inputs)


def test_extract_tuple_output_module(tuple_output_module, tuple_output_module_inputs):
    graph_module, meta = extract(tuple_output_module, tuple_output_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(tuple_output_module, graph_module, tuple_output_module_inputs)


def test_extract_deeply_nested_module(
    deeply_nested_output_module, deeply_nested_output_module_inputs
):
    graph_module, meta = extract(deeply_nested_output_module, deeply_nested_output_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(
        deeply_nested_output_module, graph_module, deeply_nested_output_module_inputs
    )


def test_extract_graph_break_module(graph_break_module, graph_break_module_inputs):
    graph_module, meta = extract(graph_break_module, graph_break_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(graph_break_module, graph_module, graph_break_module_inputs)


@requires_transformers
def test_extract_pretrained_module(pretrained_module, pretrained_module_inputs):
    graph_module, meta = extract(pretrained_module, pretrained_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(pretrained_module, graph_module, pretrained_module_inputs)


@requires_multi_gpu
@requires_transformers
@requires_accelerate
def test_extract_multiple_device_module(
    accelerate_pretrained_module, accelerate_pretrained_module_inputs
):
    graph_module, meta = extract(accelerate_pretrained_module, accelerate_pretrained_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(
        accelerate_pretrained_module, graph_module, accelerate_pretrained_module_inputs
    )


@requires_gpu
@requires_bitsandbytes
@requires_transformers
@requires_accelerate
def test_extract_quantized_pretrained_module(
    quantized_pretrained_module, quantized_pretrained_module_inputs
):
    graph_module, meta = extract(quantized_pretrained_module, quantized_pretrained_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(
        quantized_pretrained_module,
        graph_module,
        quantized_pretrained_module_inputs,
        # TODO: There's probably a better way to test this while accounting for slight differences
        # in quantization, probably involving looking at intermediate values rather than just the
        # final output, which compounds small differences.
        tolerance=1.0,
    )
