import inspect

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


def _opaque_and_compiled(pg_fixture):
    def _decorator(test_fn):
        # Assume the PatchableGraph fixture comes first.
        extra_fixtures = list(inspect.signature(test_fn).parameters.keys())[1:]

        fn_args = ", ".join([pg_fixture] + extra_fixtures)

        # Super hacky, but this causes pytest to see the wrapped test as having the name of the
        # desired fixture, as well as any other fixtures used by the original function.
        eval_dict = {"pytest": pytest, "test_fn": test_fn}
        eval(
            compile(
                f"""
@pytest.mark.parametrize("{pg_fixture}", ["compiled", "opaque"], indirect=True)
def _inner({fn_args}):
    return test_fn({fn_args})
""",
                "test_patch.py",
                "exec",
            ),
            eval_dict,
        )
        return eval_dict["_inner"]

    return _decorator


@_opaque_and_compiled("patchable_minimal_module")
def test_patch_module_input(pg, minimal_module_inputs):
    with pg.patch({"linear.input": ZeroPatch()}):
        patched_output = pg(minimal_module_inputs)

    assert patched_output.equal(pg._graph_module.linear(torch.zeros_like(minimal_module_inputs)))


@_opaque_and_compiled("patchable_minimal_module")
def test_patch_module_output(pg, minimal_module_inputs):
    with pg.patch({"output": ReplacePatch(value=3, slice=(slice(None), slice(None)))}):
        patched_output = pg(minimal_module_inputs)

    assert (patched_output - 3).count_nonzero() == 0


@_opaque_and_compiled("patchable_minimal_module")
def test_patch_probe(pg, minimal_module_inputs):
    with pg.patch(
        {"linear.weight": [weight_probe := ProbePatch()], "output": [output_probe := ProbePatch()]}
    ):
        patched_output = pg(minimal_module_inputs)

    assert output_probe.activation.equal(patched_output)
    assert weight_probe.activation.equal(pg._graph_module.linear.weight)


@_opaque_and_compiled("patchable_minimal_module")
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


@_opaque_and_compiled("patchable_minimal_module")
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


@_opaque_and_compiled("patchable_minimal_module")
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


@_opaque_and_compiled("patchable_tuple_output_module")
def test_patch_output_tuple(pg, tuple_output_module_inputs):
    with pg.patch(
        {
            "linear.output": [AddPatch(value=torch.ones((1,)))],
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


@_opaque_and_compiled("patchable_tuple_output_module")
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


@_opaque_and_compiled("patchable_tuple_output_module")
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


@_opaque_and_compiled("patchable_deeply_nested_output_module")
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
