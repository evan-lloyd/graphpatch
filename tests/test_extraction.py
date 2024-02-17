from itertools import chain, combinations

from graphpatch.extraction import ExtractionOptions
from graphpatch.extraction.graph_extraction import extract
from tests.fixtures.nested_module import A, B, C, NestedModule

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
    graph_module, meta = extract(minimal_module, ExtractionOptions(), minimal_module_inputs)

    validate_node_meta(meta, graph_module)
    assert_outputs_identical(minimal_module, graph_module, minimal_module_inputs)


def test_extract_nested_module(nested_module, nested_module_inputs):
    graph_module, meta = extract(nested_module, ExtractionOptions(), nested_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(nested_module, graph_module, nested_module_inputs)


def test_extract_tuple_output_module(tuple_output_module, tuple_output_module_inputs):
    graph_module, meta = extract(
        tuple_output_module, ExtractionOptions(), tuple_output_module_inputs
    )
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(tuple_output_module, graph_module, tuple_output_module_inputs)


def test_extract_deeply_nested_module(
    deeply_nested_output_module, deeply_nested_output_module_inputs
):
    graph_module, meta = extract(
        deeply_nested_output_module, ExtractionOptions(), deeply_nested_output_module_inputs
    )
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(
        deeply_nested_output_module, graph_module, deeply_nested_output_module_inputs
    )


def test_extract_with_opaque_modules(nested_module, nested_module_inputs):
    # Try every possible combination of opaque/compiled modules--we must always get an equivalent
    # module!
    module_types = {NestedModule, A, B, C}
    for uncompiled_subset in chain.from_iterable(
        combinations(module_types, subset_len) for subset_len in range(len(module_types) + 1)
    ):
        graph_module, meta = extract(
            nested_module,
            ExtractionOptions(classes_to_skip_compiling=uncompiled_subset),
            nested_module_inputs,
        )
        validate_node_meta(meta, graph_module)
        assert_outputs_identical(nested_module, graph_module, nested_module_inputs)


def test_extraction_fallbacks(graph_break_module, graph_break_module_inputs):
    # With default settings, we should silently get an opaque graph module back.
    graph_module, meta = extract(graph_break_module, ExtractionOptions(), graph_break_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(graph_break_module, graph_module, graph_break_module_inputs)
    breakpoint()
    assert "opaque_module_call" in meta


@requires_transformers
def test_extract_pretrained_module(pretrained_module, pretrained_module_inputs):
    graph_module, meta = extract(pretrained_module, ExtractionOptions(), pretrained_module_inputs)
    validate_node_meta(meta, graph_module)
    assert_outputs_identical(pretrained_module, graph_module, pretrained_module_inputs)


@requires_multi_gpu
@requires_transformers
@requires_accelerate
def test_extract_multiple_device_module(
    accelerate_pretrained_module, accelerate_pretrained_module_inputs
):
    graph_module, meta = extract(
        accelerate_pretrained_module, ExtractionOptions(), accelerate_pretrained_module_inputs
    )
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
    graph_module, meta = extract(
        quantized_pretrained_module, ExtractionOptions(), quantized_pretrained_module_inputs
    )
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
