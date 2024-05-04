import inspect
import os

import pytest
import torch
from torch.fx.graph_module import GraphModule

from graphpatch import PatchableGraph
from graphpatch.meta import GraphMeta


def opaque_and_compiled(pg_fixture):
    # TODO: seems like pytest-cases might offer a better solution here.
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
                inspect.getfile(test_fn),
                "exec",
            ),
            eval_dict,
        )
        return eval_dict["_inner"]

    return _decorator


def assert_patchable_graphs_identical(graph_1: PatchableGraph, graph_2: PatchableGraph):
    # Module hierarchy and code must be identical.
    submodules_1 = dict(graph_1.named_modules(remove_duplicate=False))
    submodules_2 = dict(graph_2.named_modules(remove_duplicate=False))
    assert set(submodules_1.keys()) == set(submodules_2.keys()), "Module hierarchies differ"
    # Comparing classes by name because each GraphModule is assigned a bespoke class.
    assert {k: v.__class__.__name__ for k, v in submodules_1.items()} == {
        k: v.__class__.__name__ for k, v in submodules_2.items()
    }, "Module classes differ"
    for k in submodules_1:
        if not isinstance(submodules_1[k], GraphModule):
            continue
        assert submodules_1[k].code == submodules_2[k].code, f"Graph code differs for {k}"

    # Parameters must be identical.
    parameters_1 = dict(graph_1.named_parameters(remove_duplicate=False))
    parameters_2 = dict(graph_2.named_parameters(remove_duplicate=False))
    assert set(parameters_1.keys()) == set(parameters_2.keys()), "Parameter sets differ"
    for k in parameters_1:
        [*submodule_name, weight_name] = k.split(".")
        submodule_1 = graph_1.get_submodule(".".join(submodule_name))
        submodule_2 = graph_2.get_submodule(".".join(submodule_name))
        # TODO: assert hooks equal
        if getattr(submodule_1._graphpatch_accelerate_hook, "offload", False):
            assert submodule_1._graphpatch_accelerate_hook.weights_map[weight_name].equal(
                submodule_2._graphpatch_accelerate_hook.weights_map[weight_name]
            ), f"Parameter mismatch for {k}"
        else:
            assert parameters_1[k].equal(parameters_2[k]), f"Parameter mismatch for {k}"

    buffers_1 = dict(graph_1.named_buffers())
    buffers_2 = dict(graph_2.named_buffers())

    assert set(buffers_1.keys()) == set(buffers_2.keys()), "Buffer sets differ"
    for k in buffers_1:
        assert buffers_1[k].equal(buffers_2[k]), f"Buffer mismatch for {k}"

    # Meta-info (node names, shapes, etc) must have transfered.
    assert list(graph_1._meta.keys()) == list(graph_2._meta.keys()), "Meta structure differs"
    for k in graph_1._meta.keys():
        assert graph_1._meta[k].name == graph_2._meta[k].name
        assert graph_1._meta[k].local_name == graph_2._meta[k].local_name
        assert graph_1._meta[k].shape == graph_2._meta[k].shape
        assert graph_1._meta[k].parent == graph_2._meta[k].parent


def assert_topk_tokens_identical(module1, module2, test_inputs, k, input_kwargs=None):
    output1 = module1(test_inputs, **(input_kwargs or {}))
    output2 = module2(test_inputs, **(input_kwargs or {}))
    if isinstance(output1, tuple):
        output1 = output1[0]
        output2 = output2[0]
    assert output1.shape == output2.shape, "Model output shape differs"
    top1 = torch.topk(output1[:, -1, :], k, sorted=True)
    top2 = torch.topk(output2[:, -1, :], k, sorted=True)
    assert top1.indices.equal(top2.indices)
    return output1, output2


def assert_on_nested_tensors(output_1, output_2):
    stack_1 = [("", output_1)]
    stack_2 = [("", output_2)]
    while stack_1 or stack_2:
        assert stack_1 and stack_2, "Model output nesting differs"
        (prefix_1, cur_1), (prefix_2, cur_2) = stack_1.pop(), stack_2.pop()
        assert (
            prefix_1 == prefix_2
        ), f"Model output key differs: {prefix_1 or '<root>'} != {prefix_2 or '<root>'}"
        assert (
            cur_1.__class__ is cur_2.__class__
        ), f"Model output type differs at {prefix_1 or '<root>'}"
        if isinstance(cur_1, (tuple, list)):
            assert len(cur_1) == len(
                cur_2
            ), f"Model output length differs at {prefix_1 or '<root>'}"
            stack_1.extend(
                (f"{prefix_1 + '.' if prefix_1 else ''}{i}", v) for i, v in enumerate(cur_1)
            )
            stack_2.extend(
                (f"{prefix_2 + '.' if prefix_2 else ''}{i}", v) for i, v in enumerate(cur_2)
            )
        elif isinstance(cur_1, dict):
            assert len(cur_1) == len(
                cur_2
            ), f"Model output length differs at {prefix_1 or '<root>'}"
            stack_1.extend(
                (f"{prefix_1 + '.' if prefix_1 else ''}{k}", v) for k, v in cur_1.items()
            )
            stack_2.extend(
                (f"{prefix_2 + '.' if prefix_2 else ''}{k}", v) for k, v in cur_2.items()
            )
        elif isinstance(cur_1, torch.Tensor) and isinstance(cur_2, torch.Tensor):
            assert (
                cur_1.shape == cur_2.shape
            ), f"Model output shape differs at {prefix_1 or '<root>'}"
            yield (prefix_1, cur_1), (prefix_2, cur_2)


def assert_outputs_identical(module_1, module_2, *test_inputs, tolerance=None, input_kwargs=None):
    output_1 = module_1(*test_inputs, **(input_kwargs or {}))
    output_2 = module_2(*test_inputs, **(input_kwargs or {}))

    for (prefix_1, cur_1), (prefix_2, cur_2) in assert_on_nested_tensors(output_1, output_2):
        if tolerance is not None:
            assert (norm := torch.linalg.norm(cur_1 - cur_2)) <= tolerance, (
                f"Model output difference greater than tolerance at {prefix_1 or '<root>'};"
                f" norm: {norm}"
            )
        else:
            assert cur_1.equal(cur_2), (
                f"Model output differs at {prefix_1 or '<root>'};"
                f" norm: {torch.linalg.norm(cur_1 - cur_2)}"
            )

    return output_1, output_2


def assert_gradients_identical(module_1, module_2, output_1, output_2, tolerance=None):
    for (prefix_1, cur_1), (prefix_2, cur_2) in assert_on_nested_tensors(output_1, output_2):
        assert (
            cur_1.requires_grad == cur_2.requires_grad
        ), f"requires_grad mismatch at {prefix_1 or '<root>'}"
        if not cur_1.requires_grad:
            continue
        module_1.zero_grad()
        module_2.zero_grad()
        loss_1 = cur_1.sum()
        loss_2 = cur_2.sum()
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)

        params_1 = dict(module_1.named_parameters(remove_duplicate=False))
        params_2 = dict(module_2.named_parameters(remove_duplicate=False))

        for name in params_1.keys():
            # Can happen if module makes multiple calls so that our compiled version clones it. In
            # that case we can just check the first instance.
            if name not in params_2:
                [*path, param_name] = name.split(".")
                param_2_name = f"{'.'.join(path)}.0.{param_name}"
            else:
                param_2_name = name
            assert (
                params_1[name].requires_grad == params_2[param_2_name].requires_grad
            ), f"requires_grad mismatch for {name} at {prefix_1 or '<root>'}"
            if not params_1[name].requires_grad:
                continue
            assert (params_1[name].grad is None) == (
                params_2[param_2_name].grad is None
            ), f"Gradient exists/doesn't exist for {name} at {prefix_1 or '<root>'}"
            if params_1[name].grad is None:
                continue
            if tolerance is not None:
                assert (
                    norm := torch.linalg.norm(params_1[name].grad - params_2[param_2_name].grad)
                ) <= tolerance, (
                    f"Gradient difference for {name} greater than tolerance at {prefix_1 or '<root>'};"
                    f" norm: {norm}"
                )
            else:
                assert params_1[name].grad.equal(
                    params_2[param_2_name].grad
                ), f"Gradient mismatch for {name} at {prefix_1 or '<root>'}"


def assert_results_identical(module_1, module_2, *test_inputs, tolerance=None, input_kwargs=None):
    output_1, output_2 = assert_outputs_identical(
        module_1, module_2, *test_inputs, tolerance=tolerance, input_kwargs=input_kwargs
    )
    assert_gradients_identical(module_1, module_2, output_1, output_2, tolerance=tolerance)


def validate_node_meta(graph_module, meta):
    # Name attribute must match key
    assert [n.name for n in meta.values()] == list(meta.keys())
    for name, module in graph_module.named_modules():
        if name in meta:
            assert isinstance(meta[name], GraphMeta), f"Expected {name} to be a Graph"
            assert meta[name].parent == ".".join(name.split(".")[:-1])
    # TODO: more validation. probably don't want to use named_modules
    # ModuleList added to handle duplicate instances of submodule in original code


def validate_module_hierarchy(graph_module):
    # This can happen when the hierarchy has containers and something went wrong when we extracted
    # the graph! For example, the root could have a submodule named "foo.0", but actually we were
    # supposed to place that in a container *named* foo. PyTorch, for some reason, lets you have
    # submodules with dots in their name, but this leads to ambiguity so we disallow it.
    submodule_stack = [("", graph_module)]
    while submodule_stack:
        path, cur_module = submodule_stack.pop()
        if path == "":
            prefix = ""
        else:
            prefix = f"{path}."
        for name, submodule in cur_module._modules.items():
            assert (
                "." not in name
            ), f"Module hierarchy contains invalid name: {name} at {path or '<root>'}"
        submodule_stack.extend((f"{prefix}{n}", m) for n, m in cur_module._modules.items())


def validate_extraction(graph_module, meta):
    validate_node_meta(graph_module, meta)
    validate_module_hierarchy(graph_module)


def requires_multi_gpu(f):
    if torch.cuda.device_count() < 2:
        return pytest.mark.skip("Need multiple CUDA devices to run this test")(f)
    return f


def requires_gpu(f):
    if torch.cuda.device_count() < 1:
        return pytest.mark.skip("Need a CUDA device to run this test")(f)
    return f


def requires_accelerate(f):
    from graphpatch.optional import accelerate

    if not accelerate.AVAILABLE:
        return pytest.mark.skip("accelerate must be installed to run this test")(f)
    return f


def requires_bitsandbytes(f):
    from graphpatch.optional import bitsandbytes

    if not bitsandbytes.AVAILABLE:
        return pytest.mark.skip("bitsandbytes must be installed to test quantization")(f)
    return f


def requires_transformers(f):
    from graphpatch.optional import transformers

    if not transformers.AVAILABLE:
        return pytest.mark.skip("transformers must be installed to run this test")(f)
    return f


def requires_transformer_lens(f):
    from graphpatch.optional import transformer_lens

    if not transformer_lens.AVAILABLE:
        return pytest.mark.skip("transformer_lens must be installed to run this test")(f)
    return f


def long_running(f):
    if not os.getenv("RUN_LONG_TESTS"):
        return pytest.mark.skip(
            "Must opt in to long-running tests with RUN_LONG_TESTS in environment"
        )(f)
    return f
