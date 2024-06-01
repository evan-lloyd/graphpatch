import pytest
import torch

from graphpatch import (
    AddPatch,
    CustomPatch,
    ProbePatch,
    RecordPatch,
    ReplacePatch,
    ZeroPatch,
)

from .util import (
    opaque_and_compiled,
    requires_accelerate,
    requires_bitsandbytes,
    requires_gpu,
    requires_multi_gpu,
    requires_transformers,
)


@opaque_and_compiled("patchable_minimal_module")
def test_patch_module_input(pg, minimal_module_inputs):
    with pg.patch({"linear.input": ZeroPatch()}):
        patched_output = pg(minimal_module_inputs)

    assert patched_output.equal(pg._graph_module.linear(torch.zeros_like(minimal_module_inputs)))


@opaque_and_compiled("patchable_minimal_module")
def test_patch_module_output(pg, minimal_module_inputs):
    with pg.patch({"output": ReplacePatch(value=3, slice=(slice(None), slice(None)))}):
        patched_output = pg(minimal_module_inputs)

    assert (patched_output - 3).count_nonzero() == 0


@opaque_and_compiled("patchable_minimal_module")
def test_patch_probe(pg, minimal_module_inputs):
    with pg.patch(
        {"linear.weight": [weight_probe := ProbePatch()], "output": [output_probe := ProbePatch()]}
    ):
        patched_output = pg(minimal_module_inputs)

    assert output_probe.activation.equal(patched_output)
    assert weight_probe.activation.equal(pg._graph_module.linear.weight)


@opaque_and_compiled("patchable_container_module")
def test_patch_duplicate_modules(pg, container_module_inputs):
    with pg.patch(
        {
            "linear_0.weight": [weight_probe := ProbePatch()],
            "output": [output_probe := ProbePatch()],
        }
    ):
        patched_output = pg(container_module_inputs)

    # TODO: test differential patches to each invocation of linear

    assert output_probe.activation.equal(patched_output)
    assert weight_probe.activation.equal(pg._graph_module.linear[0].weight)


@opaque_and_compiled("patchable_minimal_module")
def test_update_patch_context(pg, minimal_module_inputs):
    original_output = pg(minimal_module_inputs)
    with pg.patch({"output": [pre_activation := RecordPatch()]}):
        patched_output_1 = pg(minimal_module_inputs)
        with pg.patch({"output": ZeroPatch()}):
            patched_output_2 = pg(minimal_module_inputs)
            with pg.patch({"output": [post_activation := RecordPatch()]}):
                patched_output_3 = pg(minimal_module_inputs)
            patched_output_4 = pg(minimal_module_inputs)
        patched_output_5 = pg(minimal_module_inputs)

    # Context should have persisted for all 5 calls
    assert len(pre_activation.activations) == 5
    # Probe should be applied before the zero patch in all cases
    assert all(a.count_nonzero() == a.numel() for a in pre_activation.activations)

    # Should only have been active for the innermost context
    assert len(post_activation.activations) == 1
    assert post_activation.activations[0].count_nonzero() == 0
    assert post_activation.activations[0].equal(patched_output_3)

    # Zero should have been active for middle 3 runs
    assert (
        patched_output_2.count_nonzero()
        == patched_output_3.count_nonzero()
        == patched_output_4.count_nonzero()
        == 0
    )

    # With no zero patch, we should match the original output
    assert original_output.equal(patched_output_1)
    assert original_output.equal(patched_output_5)


@opaque_and_compiled("patchable_minimal_module")
def test_patch_with_slice(pg, minimal_module_inputs):
    with pg.patch(
        {
            "linear.weight": [
                AddPatch(slice=[1, ...], value=torch.ones((2,))),
                weight_probe := ProbePatch(),
            ],
        }
    ):
        patched_output_1 = pg(minimal_module_inputs)
        with pg.patch({"output": [ZeroPatch(slice=[(0, 2), slice(1, 3)])]}):
            patched_output_2 = pg(minimal_module_inputs)

    # Should have modified the last two columns in the first and third row
    assert patched_output_2.equal(
        torch.vstack(
            (
                torch.Tensor([patched_output_1[0, 0], 0, 0]),
                patched_output_1[1, :],
                torch.Tensor([patched_output_1[2, 0], 0, 0]),
            )
        )
    )

    # Should have modified only the middle row
    assert weight_probe.activation.equal(
        torch.vstack(
            (
                pg._graph_module.linear.weight[0, :],
                pg._graph_module.linear.weight[1, :] + 1,
                pg._graph_module.linear.weight[2, :],
            )
        )
    )


@opaque_and_compiled("patchable_minimal_module")
def test_custom_patch(pg, minimal_module_inputs):
    original_weights = pg._graph_module.linear.weight.detach().clone()

    def custom_op(tensor):
        tensor += 1
        return tensor

    with pg.patch(
        {
            "linear.weight": [
                CustomPatch(custom_op=custom_op),
                probe := ProbePatch(),
            ]
        }
    ):
        pg(minimal_module_inputs)
    assert probe.activation.equal(pg._graph_module.linear.weight + 1)
    # Weight must not have been overwritten.
    assert pg._graph_module.linear.weight.equal(original_weights)


@opaque_and_compiled("patchable_tuple_output_module")
def test_patch_output_tuple(pg, tuple_output_module_inputs):
    with pg.patch(
        {
            "linear_0.output": [AddPatch(value=torch.ones((1,)))],
            "linear_1.output": [ZeroPatch()],
            "output|sub_0": [probe_0 := ProbePatch()],
            "output|sub_1": [probe_1 := ProbePatch()],
        }
    ):
        patched_output = pg(tuple_output_module_inputs)

    assert isinstance(patched_output, tuple)
    assert len(patched_output) == 2
    # Zero patch should have been effective
    assert patched_output[1].count_nonzero() == 0
    assert probe_1.activation.equal(patched_output[1])
    # Add patch should have been effective (and independent of zero patch, despite sharing modules
    # in the base model)
    assert patched_output[0].equal(pg(tuple_output_module_inputs)[0] + 1)
    assert probe_0.activation.equal(patched_output[0])


@opaque_and_compiled("patchable_tuple_output_module")
def test_patch_with_node_paths(pg, tuple_output_module_inputs):
    with pg.patch(
        {
            "output|sub_0": [AddPatch(value=torch.ones((1,)))],
            pg.graph.output.sub_1: [ZeroPatch()],
        }
    ):
        patched_output = pg(tuple_output_module_inputs)

    assert isinstance(patched_output, tuple)
    assert len(patched_output) == 2
    # Zero patch should have been effective
    assert patched_output[1].count_nonzero() == 0
    # Add patch should have been effective
    assert patched_output[0].equal(pg(tuple_output_module_inputs)[0] + 1)


@opaque_and_compiled("patchable_tuple_output_module")
def test_patch_with_node_path_exceptions(pg, tuple_output_module_inputs, mocker):
    node_path = pg.graph
    spy = mocker.spy(pg, "forward")

    # We need to fail early when the given node paths are invalid

    # Not a dict
    with pytest.raises(ValueError):
        with pg.patch({"foo"}):
            pg(tuple_output_module_inputs)
    assert spy.call_count == 0

    # Invalid path, but valid value
    with pytest.raises(KeyError):
        with pg.patch({"foo": [ZeroPatch()]}):
            pg(tuple_output_module_inputs)
    assert spy.call_count == 0

    # Invalid value, but valid path
    with pytest.raises(ValueError):
        with pg.patch({"output|sub_0": [3]}):
            pg(tuple_output_module_inputs)
    assert spy.call_count == 0

    # Invalid path (too many digs)
    with pytest.raises(KeyError):
        with pg.patch({"output|sub_0|sub_0": [ZeroPatch()]}):
            pg(tuple_output_module_inputs)
    assert spy.call_count == 0

    # Invalid path (incomplete node_path)
    with pytest.raises(KeyError):
        with pg.patch({node_path.output: [ZeroPatch()]}):
            pg(tuple_output_module_inputs)
    assert spy.call_count == 0


@opaque_and_compiled("patchable_deeply_nested_output_module")
def test_patch_deeply_nested_output_model(pg, deeply_nested_output_module_inputs):
    with pg.patch(
        {
            pg.graph.child_a.grandchildren_b_1.output.sub_0.sub_0.sub_0.sub_0.sub_0: [
                pre_activation := ProbePatch(),
                ZeroPatch(),
                post_activation := ProbePatch(),
            ]
        }
    ):
        pg(deeply_nested_output_module_inputs)

    assert pre_activation.activation.count_nonzero() == pre_activation.activation.numel()
    assert post_activation.activation.count_nonzero() == 0


@opaque_and_compiled("patchable_graph_break_module")
def test_patch_after_fallback(pg, graph_break_module_inputs):
    # Cloned graph weights should be patchable without affecting each other.
    original_weight_0 = pg._graph_module.linear[0].weight.clone()
    original_weight_1 = pg._graph_module.linear[1].weight.clone()
    original_weight_2 = pg._graph_module.linear[2].weight.clone()
    original_output = pg(graph_break_module_inputs)

    with pg.patch(
        {
            pg.graph.linear_0.weight: [probe_weight_0 := ProbePatch(), AddPatch(value=1)],
            pg.graph.linear_1.weight: [probe_weight_1 := ProbePatch(), AddPatch(value=1)],
            pg.graph.linear_2.weight: [probe_weight_2 := ProbePatch(), AddPatch(value=1)],
        }
    ):
        pg(graph_break_module_inputs)
    assert probe_weight_0.activation.equal(original_weight_0)
    assert probe_weight_1.activation.equal(original_weight_1)
    assert probe_weight_2.activation.equal(original_weight_2)

    # We should be able to patch attributes.
    with pg.patch({pg.graph.shadowed_class_var: AddPatch(value=5)}):
        patched_output = pg(graph_break_module_inputs)

    assert patched_output.equal(original_output + 5)


@opaque_and_compiled("patchable_container_module")
def test_patch_container_module(pg, container_module_inputs):
    with pg.patch(
        {
            pg.graph.module_dict_bar_baz_1.weight: ZeroPatch(),
            pg.graph.sequential.output: [sum_probe_0 := ProbePatch()],
            pg.graph.linear_3.output: [sum_probe_1 := ProbePatch()],
            pg.graph.module_list_0_1.output: [sum_probe_2 := ProbePatch(), AddPatch(value=1)],
        }
    ):
        output = pg(container_module_inputs)
    assert output.allclose(
        sum_probe_0.activation + sum_probe_1.activation + sum_probe_2.activation + 1
    )


@opaque_and_compiled("patchable_varargs_module")
def test_patch_varargs_module(
    pg, varargs_module_inputs, varargs_module_varargs, varargs_module_varkwargs
):
    with pg.patch(
        {
            pg.graph.linear_1.input: (sum_probe_1_orig := ProbePatch()),
        }
    ):
        pg(varargs_module_inputs, *varargs_module_varargs, **varargs_module_varkwargs)
    with pg.patch(
        {
            pg.graph.foos.sub_0: AddPatch(value=1),
            pg.graph.bars.a: AddPatch(value=2),
            pg.graph.linear_1.input: (sum_probe_1_patched := ProbePatch()),
        }
    ):
        patched_output = pg(
            varargs_module_inputs, *varargs_module_varargs, **varargs_module_varkwargs
        )
    assert sum_probe_1_patched.activation.allclose(sum_probe_1_orig.activation + 1)
    assert patched_output.allclose(
        pg._graph_module.linear[1](sum_probe_1_patched.activation)
        + 2
        + sum(varargs_module_varkwargs.values())
    )


@opaque_and_compiled("patchable_buffer_module")
def test_patch_buffer_module(pg, buffer_module_inputs):
    original_output = pg(buffer_module_inputs)
    with pg.patch({"buffer": ZeroPatch()}):
        patched_output = pg(buffer_module_inputs)
    # Buffer is full of ones and added to linear
    assert original_output.equal(patched_output + 1)


def _pretrained_module_patch_asserts(pg, inputs):
    # Basic run
    with pg.patch(
        {
            "model.root_linear.weight": ZeroPatch(),
            "model.root_linear.bias": ZeroPatch(),
            "output|logits": [output_probe := ProbePatch()],
        }
    ):
        output = pg(inputs)
    assert output.logits.count_nonzero() == 0
    assert output_probe.activation.count_nonzero() == 0

    # Generate
    generate_outputs = pg.generate(inputs)
    assert generate_outputs.shape[-1] == pg.generation_config.max_length

    logits_for_target_token = torch.zeros((1, 100))
    logits_for_target_token[0, 42] = 1
    with pg.patch({"output|logits": ReplacePatch(value=logits_for_target_token)}):
        patched_generate_outputs = pg.generate(inputs)
    # Should only have generated the target token since we patched the output logits
    assert (patched_generate_outputs[100:] - 42).count_nonzero() == 0


@requires_transformers
@opaque_and_compiled("patchable_pretrained_module")
def test_patch_pretrained_module(pg, pretrained_module_inputs):
    _pretrained_module_patch_asserts(pg, pretrained_module_inputs)


@requires_multi_gpu
@requires_transformers
@requires_accelerate
@opaque_and_compiled("patchable_accelerate_pretrained_module")
def test_patch_accelerate_pretrained_module(pg, pretrained_module_inputs):
    _pretrained_module_patch_asserts(pg, pretrained_module_inputs)


@requires_gpu
@requires_transformers
@requires_accelerate
@opaque_and_compiled("patchable_mixed_cpu_pretrained_module")
def test_patch_mixed_cpu_pretrained_module(pg, pretrained_module_inputs):
    _pretrained_module_patch_asserts(pg, pretrained_module_inputs)


@requires_transformers
@requires_accelerate
@opaque_and_compiled("patchable_disk_offload_pretrained_module")
def test_patch_disk_offload_pretrained_module(pg, pretrained_module_inputs):
    _pretrained_module_patch_asserts(pg, pretrained_module_inputs)


@requires_gpu
@requires_transformers
@requires_accelerate
@requires_bitsandbytes
@opaque_and_compiled("patchable_quantized_pretrained_module")
def test_patch_quantized_pretrained_module(pg, pretrained_module_inputs):
    _pretrained_module_patch_asserts(pg, pretrained_module_inputs)
