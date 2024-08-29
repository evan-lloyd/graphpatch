from itertools import chain, combinations

import pytest
from torch.fx import Graph
from torch.nn import Linear

from graphpatch.extraction import (
    CompiledGraphModule,
    ExtractionOptions,
    OpaqueGraphModule,
    UnusedModule,
)
from graphpatch.extraction.graph_extraction import CompilationWarning, extract
from graphpatch.extraction.multiply_invoked_module import MultiplyInvokedModule
from tests.fixtures.nested_module import A, B, C, NestedModule

from .util import (
    assert_outputs_identical,
    assert_results_identical,
    requires_accelerate,
    requires_bitsandbytes,
    requires_gpu,
    requires_multi_gpu,
    requires_transformers,
    validate_extraction,
)


def test_extract_singleton_module(minimal_module, minimal_module_inputs):
    graph_module, meta = extract(
        minimal_module.linear,
        ExtractionOptions(error_on_compilation_failure=True),
        minimal_module_inputs,
    )
    validate_extraction(graph_module, minimal_module.linear, meta)
    assert_results_identical(minimal_module.linear, graph_module, minimal_module_inputs)


def test_extract_minimal_module(minimal_module, minimal_module_inputs):
    graph_module, meta = extract(
        minimal_module, ExtractionOptions(error_on_compilation_failure=True), minimal_module_inputs
    )
    validate_extraction(graph_module, minimal_module, meta)
    assert_results_identical(minimal_module, graph_module, minimal_module_inputs)


def test_extract_buffer_module(buffer_module, buffer_module_inputs):
    graph_module, meta = extract(
        buffer_module,
        ExtractionOptions(error_on_compilation_failure=True, allow_unused_submodules=True),
        buffer_module_inputs,
    )
    validate_extraction(graph_module, buffer_module, meta)
    assert_results_identical(buffer_module, graph_module, buffer_module_inputs)
    assert "buffer" in graph_module._buffers


def test_extract_nested_module(nested_module, nested_module_inputs):
    graph_module, meta = extract(
        nested_module, ExtractionOptions(error_on_compilation_failure=True), nested_module_inputs
    )
    validate_extraction(graph_module, nested_module, meta)
    assert_results_identical(nested_module, graph_module, nested_module_inputs)


def test_unused_submodule_module(unused_submodule_module, unused_submodule_module_inputs):
    graph_module, meta = extract(
        unused_submodule_module,
        ExtractionOptions(error_on_compilation_failure=True, allow_unused_submodules=True),
        unused_submodule_module_inputs,
    )
    validate_extraction(graph_module, unused_submodule_module, meta, allow_unused=True)
    assert_results_identical(
        unused_submodule_module, graph_module, unused_submodule_module_inputs, allow_unused=True
    )
    assert isinstance(graph_module.get_submodule("child_a.grandchildren_b.2"), UnusedModule)
    assert isinstance(
        graph_module.get_submodule("child_a.grandchildren_b.1.c.c_unused"), UnusedModule
    )
    # Unused submodules shouldn't show up in meta
    assert not any("c_unused" in k for k in meta.keys())
    assert not any("b_2" in k for k in meta.keys())


def test_extract_tuple_output_module(tuple_output_module, tuple_output_module_inputs):
    graph_module, meta = extract(
        tuple_output_module,
        ExtractionOptions(error_on_compilation_failure=True),
        tuple_output_module_inputs,
    )
    validate_extraction(graph_module, tuple_output_module, meta)
    assert_results_identical(tuple_output_module, graph_module, tuple_output_module_inputs)


def test_extract_deeply_nested_module(
    deeply_nested_output_module, deeply_nested_output_module_inputs
):
    graph_module, meta = extract(
        deeply_nested_output_module,
        ExtractionOptions(error_on_compilation_failure=True),
        deeply_nested_output_module_inputs,
    )
    validate_extraction(graph_module, deeply_nested_output_module, meta)
    assert_results_identical(
        deeply_nested_output_module, graph_module, deeply_nested_output_module_inputs
    )


def test_extract_with_opaque_modules(nested_module, nested_module_inputs):
    # Try every possible combination of opaque/compiled modules--we must always get an equivalent
    # module!
    module_types = (NestedModule, A, B, C, Linear)
    for uncompiled_subset in chain.from_iterable(
        combinations(module_types, subset_len) for subset_len in range(len(module_types) + 1)
    ):
        graph_module, meta = extract(
            nested_module,
            ExtractionOptions(
                classes_to_skip_compiling=uncompiled_subset, error_on_compilation_failure=True
            ),
            nested_module_inputs,
        )
        validate_extraction(graph_module, nested_module, meta)
        assert_results_identical(nested_module, graph_module, nested_module_inputs)


def test_extract_container_module(container_module, container_module_inputs):
    compiled_graph_module, meta = extract(
        container_module,
        ExtractionOptions(error_on_compilation_failure=True),
        container_module_inputs,
    )
    validate_extraction(compiled_graph_module, container_module, meta)
    assert_results_identical(container_module, compiled_graph_module, container_module_inputs)

    opaque_graph_module, meta = extract(
        container_module, ExtractionOptions(skip_compilation=True), container_module_inputs
    )
    validate_extraction(opaque_graph_module, container_module, meta)
    assert_results_identical(container_module, opaque_graph_module, container_module_inputs)


def test_extract_varargs_module(varargs_module, varargs_module_inputs):
    varargs = (
        varargs_module_inputs,
        varargs_module_inputs,
    )
    varkwargs = {"hmm": varargs_module_inputs, "aha": varargs_module_inputs}
    compiled_graph_module, meta = extract(
        varargs_module,
        ExtractionOptions(error_on_compilation_failure=True),
        varargs_module_inputs,
        *varargs,
        **varkwargs,
    )
    validate_extraction(compiled_graph_module, varargs_module, meta)
    assert_results_identical(
        varargs_module,
        compiled_graph_module,
        varargs_module_inputs,
        *varargs,
        input_kwargs=varkwargs,
    )

    opaque_graph_module, meta = extract(
        varargs_module,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=True),
        varargs_module_inputs,
        *varargs,
        **varkwargs,
    )
    validate_extraction(opaque_graph_module, varargs_module, meta)
    assert_results_identical(
        varargs_module, opaque_graph_module, varargs_module_inputs, *varargs, input_kwargs=varkwargs
    )


def test_extraction_fallbacks(graph_break_module, graph_break_module_inputs):
    # With default settings, we should silently get an opaque graph module back.
    graph_module, meta = extract(
        graph_break_module,
        ExtractionOptions(),
        graph_break_module_inputs,
    )
    validate_extraction(graph_module, graph_break_module, meta)
    assert_results_identical(graph_break_module, graph_module, graph_break_module_inputs)
    assert isinstance(graph_module, OpaqueGraphModule)
    # Child module should have been compiled despite failure at root.
    assert isinstance(graph_module.linear, MultiplyInvokedModule)
    assert [isinstance(m, CompiledGraphModule) for m in graph_module.linear] == [True] * 3

    # User should see the original raised exception with error_on_compilation_failure
    with pytest.raises(Exception) as exc:
        extract(
            graph_break_module,
            ExtractionOptions(
                error_on_compilation_failure=True,
            ),
            graph_break_module_inputs,
        )
    assert "graph_break" in str(exc.value)

    # User should get a warning referencing the original exception.
    with pytest.warns(CompilationWarning) as warnings:
        graph_module, meta = extract(
            graph_break_module,
            ExtractionOptions(
                warn_on_compilation_failure=True,
            ),
            graph_break_module_inputs,
        )
        assert len(warnings) == 2
        assert "graph_break" in warnings[0].message.args[0]
        assert (
            "Unable to compile unused_submodule; it was never called" in warnings[1].message.args[0]
        )


def test_custom_extraction_function(minimal_module, minimal_module_inputs):
    # Dummy graph that just returns its input.
    def generate_dummy_graph(module):
        graph = Graph()
        placeholder = graph.placeholder("input")

        def foo():
            pass

        graph.call_function(foo)
        graph.output((placeholder,))
        return graph

    graph_module, meta = extract(
        minimal_module,
        ExtractionOptions(custom_extraction_functions={Linear: generate_dummy_graph}),
        minimal_module_inputs,
    )
    assert graph_module(minimal_module_inputs).equal(minimal_module_inputs)
    assert "linear.foo" in meta
    assert isinstance(graph_module.linear, CompiledGraphModule)


def test_custom_extraction_fallback(minimal_module, minimal_module_inputs):
    def simulate_extraction_failure(module):
        raise Exception("simulated failure")

    # Should fall back to opaque.
    graph_module, meta = extract(
        minimal_module,
        ExtractionOptions(custom_extraction_functions={Linear: simulate_extraction_failure}),
        minimal_module_inputs,
    )
    assert isinstance(graph_module.linear, OpaqueGraphModule)
    assert_results_identical(minimal_module, graph_module, minimal_module_inputs)

    # User should see the original raised exception with error_on_compilation_failure
    with pytest.raises(Exception) as exc:
        extract(
            minimal_module,
            ExtractionOptions(
                custom_extraction_functions={Linear: simulate_extraction_failure},
                error_on_compilation_failure=True,
            ),
            minimal_module_inputs,
        )
    assert "simulated failure" in str(exc.value)

    # User should get a warning referencing the original exception.
    with pytest.warns(CompilationWarning, match=r"simulated failure"):
        graph_module, meta = extract(
            minimal_module,
            ExtractionOptions(
                custom_extraction_functions={Linear: simulate_extraction_failure},
                warn_on_compilation_failure=True,
            ),
            minimal_module_inputs,
        )


@requires_transformers
def test_extract_pretrained_module(pretrained_module, pretrained_module_inputs):
    graph_module, meta = extract(
        pretrained_module,
        ExtractionOptions(error_on_compilation_failure=True),
        pretrained_module_inputs,
    )
    validate_extraction(graph_module, pretrained_module, meta)
    assert_results_identical(pretrained_module, graph_module, pretrained_module_inputs)


@requires_gpu
@requires_transformers
@requires_accelerate
def test_extract_mixed_cpu_module(mixed_cpu_pretrained_module, mixed_cpu_pretrained_module_inputs):
    graph_module, meta = extract(
        mixed_cpu_pretrained_module,
        ExtractionOptions(error_on_compilation_failure=True),
        mixed_cpu_pretrained_module_inputs,
    )
    validate_extraction(graph_module, mixed_cpu_pretrained_module, meta)
    assert_results_identical(
        mixed_cpu_pretrained_module, graph_module, mixed_cpu_pretrained_module_inputs
    )


@requires_transformers
@requires_accelerate
def test_extract_disk_offload_module(
    disk_offload_pretrained_module, disk_offload_pretrained_module_inputs
):
    graph_module, meta = extract(
        disk_offload_pretrained_module,
        ExtractionOptions(error_on_compilation_failure=True),
        disk_offload_pretrained_module_inputs,
    )
    validate_extraction(graph_module, disk_offload_pretrained_module, meta)
    assert_results_identical(
        disk_offload_pretrained_module, graph_module, disk_offload_pretrained_module_inputs
    )


@requires_multi_gpu
@requires_transformers
@requires_accelerate
def test_extract_multiple_device_module(
    accelerate_pretrained_module, accelerate_pretrained_module_inputs
):
    graph_module, meta = extract(
        accelerate_pretrained_module,
        ExtractionOptions(error_on_compilation_failure=True),
        accelerate_pretrained_module_inputs,
    )
    validate_extraction(graph_module, accelerate_pretrained_module, meta)
    assert_results_identical(
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
        quantized_pretrained_module,
        ExtractionOptions(error_on_compilation_failure=True),
        quantized_pretrained_module_inputs,
    )
    validate_extraction(graph_module, quantized_pretrained_module, meta)
    # Only asserting on outputs since with default quantization we get no gradient.
    assert_outputs_identical(
        quantized_pretrained_module,
        graph_module,
        quantized_pretrained_module_inputs,
        tolerance=0.001,
    )
