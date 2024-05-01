import inspect
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Optional, Tuple, Union
from warnings import warn

from torch.fx import Node
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleList

from .. import hacks
from ..exceptions import GraphPatchWarning
from ..meta import (
    GraphMeta,
    NodeData,
    NodeDataWrapper,
    NodeMeta,
    OutputArgumentIndex,
    wrap_graph_module,
    wrap_output_argument_index,
)
from .compiled_graph_module import CompiledGraphModule, compile_module
from .extraction_context import (
    ExtractionState,
    ExtractionWrapper,
    compilation_context,
    is_container,
)
from .extraction_options import ExtractionOptions
from .graphpatch_module import GraphPatchModule
from .invocation_tracking_module_list import InvocationTrackingModuleList
from .opaque_graph_module import OpaqueGraphModule, SubmoduleWrapper


class CompilationWarning(GraphPatchWarning):
    pass


def match_shape(indexes: NodeData[OutputArgumentIndex], *args: Any) -> Any:
    """Unflatten the args to an output node to match the original output shape."""
    if not indexes._value.should_unwrap:
        return args[0]

    return indexes.map(
        lambda _, index: args[index.index] if isinstance(index.index, int) else NodeData._NO_VALUE
    ).unwrap()


def clone_graph_module(
    module: Union[CompiledGraphModule, OpaqueGraphModule]
) -> Union[CompiledGraphModule, OpaqueGraphModule]:
    if isinstance(module, OpaqueGraphModule):
        return OpaqueGraphModule(module)
    return CompiledGraphModule(module, deepcopy(module.graph), "CompiledGraphModule")


def _should_skip_compilation(options: ExtractionOptions, module: Module):
    return options.skip_compilation or module.__class__ in options.classes_to_skip_compiling


def retarget_submodule_calls(state: ExtractionState):
    """compile() unrolls all containers, modifying the module hierarchy. Our later code is much
    simplified if we instead retain the original hierarchy. This also looks nicer when the user
    inspects the returned GraphModule.
    """
    graph_module = state.extracted_module
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        target_module = graph_module.get_submodule(node.target)
        if isinstance(target_module, ExtractionWrapper):
            # Remove parent's prefix, since the graph will be relative to it.
            if state.name != "":
                node.target = target_module._graphpatch_extraction_state.name.replace(
                    state.name + ".", "", 1
                )
            else:
                node.target = target_module._graphpatch_extraction_state.name


def _repair_input_signature(state: ExtractionState):
    """Compilation lifts some module attributes into arguments to forward, and has inconsistent
    handling of keyword arguments. We need to restore the original input signature so that calls
    to subgraphs behave correctly.
    """
    graph_module = state.extracted_module
    insert_after = graph_module.graph._root
    existing_placeholders = {n.target: n for n in graph_module.graph.nodes if n.op == "placeholder"}
    forward_parameters = inspect.signature(
        state.wrapped_module._graphpatch_wrapped_module.forward
    ).parameters

    # Construct (possibly new) graph inputs in the correct order.
    for name, parameter in forward_parameters.items():
        if name in existing_placeholders:
            type_annotation = existing_placeholders[name].type
        elif parameter.annotation != inspect._empty:
            type_annotation = parameter.annotation
        else:
            type_annotation = None
        with graph_module.graph.inserting_after(insert_after):
            if parameter.kind is inspect._ParameterKind.VAR_KEYWORD:
                target = f"**{name}"
            elif parameter.kind is inspect._ParameterKind.VAR_POSITIONAL:
                target = "f*{name}"
            else:
                target = name
            new_placeholder = Node(
                graph_module.graph,
                name,
                "placeholder",
                target,
                # Placeholder args take the default value for any kwargs.
                () if parameter.default is inspect.Signature.empty else (parameter.default,),
                {},
                type_annotation,
            )
            if name in existing_placeholders:
                hacks.replace_node_keeping_original_name(
                    existing_placeholders[name], new_placeholder, name
                )
                del existing_placeholders[name]
            else:
                graph_module.graph._insert(new_placeholder)
            insert_after = new_placeholder

    # Any remaining existing placeholders do not appear in the function signature. Assume they are
    # lifted module attributes.
    for name, placeholder in existing_placeholders.items():
        # Strip initial "self_" from the attribute name. For torch >= 2.1 we have already done this
        # via some monkeypatching during compilation.
        if hacks.TORCH_VERSION < (2, 1):
            original_attribute_name = placeholder.target[5:]
        else:
            original_attribute_name = placeholder.target
        with graph_module.graph.inserting_after(insert_after):
            setattr(
                graph_module,
                placeholder.target,
                getattr(state.wrapped_module._graphpatch_wrapped_module, original_attribute_name),
            )
            get_attr_node = Node(graph_module.graph, name, "get_attr", placeholder.target, (), {})
            hacks.replace_node_keeping_original_name(placeholder, get_attr_node, name)
            insert_after = get_attr_node


def _repair_output_signature(state: ExtractionState):
    """Compilation flattens the arguments to the output node into a single tuple, which changes the
    signature of the function. This is problematic when we have nested modules that may assume
    a particular shape! We can hack around this by injecting a function that generates the correct
    shape and returning its output.
    NB: we add an attribute to the graph_module rather than passing it to our function via closure
    because the latter method would not be picklable.
    """
    graph_module = state.extracted_module
    child_output_ids = set()
    for child in state.children.values():
        child_output_ids.update(set(id(invocation.output) for invocation in child.invocations))
    setattr(
        graph_module,
        "_graphpatch_output_indexes",
        wrap_output_argument_index(
            state.invocations[-1].output if len(state.invocations) > 0 else None,
            child_output_ids,
            # Do not unwrap values at runtime if this is an opaque module, since we'll only have
            # one input node.
            should_unwrap=not isinstance(graph_module, OpaqueGraphModule),
        ),
    )

    output_node = next(n for n in graph_module.graph.nodes if n.op == "output")
    with graph_module.graph.inserting_before(output_node):
        # Graph complains when accessing non-buffer/parameter attributes, but we don't care.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_index_node = graph_module.graph.get_attr("_graphpatch_output_indexes")
            output_index_node.meta["_graphpatch_hidden"] = True
        match_shape_node = graph_module.graph.call_function(
            match_shape, (output_index_node,) + output_node.args[0]
        )
        match_shape_node.meta["_graphpatch_hidden"] = True
        output_node.args = (match_shape_node,)


def _clone_repeated_submodules(state: ExtractionState):
    """Submodules can be used more than once, but this will cause problems later when we manipulate
    the graph, since the user may want to independently patch activations in each instance. To
    simplify things, here we clone the graph modules (but not their parameters!), so we'll have
    independent graphs to work with.
    """
    graph_module = state.extracted_module
    duplicate_graph_modules = defaultdict(list)
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            duplicate_graph_modules[graph_module.get_submodule(node.target)].append(node)
    for submodule, calling_nodes in duplicate_graph_modules.items():
        if len(calling_nodes) < 2:
            continue
        # Replace the original module with a ModuleList so we don't have to worry about name
        # collisions.
        module_name = calling_nodes[0].target
        [*parent_path, local_name] = module_name.split(".")
        setattr(
            graph_module.get_submodule(".".join(parent_path)),
            local_name,
            ModuleList([submodule] + [clone_graph_module(submodule) for _ in calling_nodes[1:]]),
        )
        # Update the call_module nodes to refer to a specific instance in the list.
        for i, node in enumerate(calling_nodes):
            node.target = f"{module_name}.{i}"
        graph_module._graphpatch_submodules[module_name] = (
            ModuleList,
            tuple(str(i) for i in range(len(calling_nodes))),
        )

    # If this module is opaque we can't track multiple invocations by nodes; instead use the
    # recorded invocations on our children.
    if isinstance(graph_module, OpaqueGraphModule):
        for name, submodule in graph_module.named_children():
            # Handle nested containers within this graph.
            [*parent_path, local_name] = name.split(".")
            child_state = state
            for next_item in parent_path + [local_name]:
                child_state = child_state.children[next_item]

            if len(child_state.invocations) < 2:
                continue

            submodule_node = next(
                n
                for n in graph_module.graph.nodes
                if n.op == "call_function"
                and isinstance(n.target, SubmoduleWrapper)
                and n.target.module_name == name
            )
            parent = graph_module.get_submodule(".".join(parent_path))
            setattr(
                parent,
                local_name,
                InvocationTrackingModuleList(
                    [submodule]
                    + [
                        clone_graph_module(submodule)
                        for _ in range(len(child_state.invocations) - 1)
                    ]
                ),
            )
            submodule_node.target = SubmoduleWrapper(f"{name}.0")
            node_name = name.replace(".", "_")
            submodule_node.name = f"{node_name}_0"
            for i in range(1, len(child_state.invocations)):
                with graph_module.graph.inserting_after(submodule_node):
                    submodule_node = graph_module.graph.call_function(
                        SubmoduleWrapper(f"{name}.{i}")
                    )
                    submodule_node.name = f"{node_name}_{i}"
            graph_module._graphpatch_submodules[name] = (
                InvocationTrackingModuleList,
                tuple(str(i) for i in range(len(child_state.invocations))),
            )


def _standardize_submodule_nodes(state: ExtractionState):
    """compile() mangles the names of nodes calling submodules that live in containers. Here we
    standardize to something much more user-friendly.
    """
    graph_module = state.extracted_module
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        node.name = node.target.replace(".", "_")
        if not node.name.isidentifier():
            node.name = f"sub_{node.name}"


def _postprocess_graph(state: ExtractionState):
    """Clean up the extracted graph to match the original module's signature and standardize node
    names.
    """
    graph_module = state.extracted_module
    _repair_input_signature(state)
    _repair_output_signature(state)
    _clone_repeated_submodules(state)
    _standardize_submodule_nodes(state)

    # Hack for some weirdness around dynamic shape detection (currently only seen in GPT2-XL)
    for node in graph_module.graph.nodes:
        hacks.maybe_replace_dynamo_get_item_lambda(node)

    graph_module.recompile()


def _run_extraction(
    state: ExtractionState, compile: bool, *args, **kwargs
) -> Tuple[GraphPatchModule, Any]:
    should_trace = state.name == ""
    for submodule in state.wrapped_module.modules():
        if isinstance(submodule, ExtractionWrapper):
            submodule._graphpatch_record_invocations = should_trace
            # Clear invocations in case of fallback after partially compiling the root module.
            if should_trace:
                submodule._graphpatch_extraction_state.invocations = []

    if compile:
        with compilation_context(state):
            extracted_module = compile_module(state.wrapped_module, *args, **kwargs)
            extracted_module._graphpatch_accelerate_hook = state.accelerate_hook
    else:
        extracted_module = OpaqueGraphModule(
            state.wrapped_module._graphpatch_wrapped_module, accelerate_hook=state.accelerate_hook
        )
        state.wrapped_module(*args, **kwargs)

    return extracted_module


class ExtractionStateWrapper(NodeDataWrapper[ExtractionState]):
    def handle_wrap(self, data: Module, path: str) -> NodeData[ExtractionState]:
        if path == "":
            prefix = ""
        else:
            prefix = f"{path}."
        graphpatch_path = hacks.override_reserved_name(path)
        children = {
            hacks.override_reserved_name(name): self.wrap(submodule, f"{prefix}{name}")
            for name, submodule in data._modules.items()
        }
        return self.make_wrapper(
            _original_type=data.__class__.__name__,
            _children=children or NodeData._NO_VALUE,
            _path=graphpatch_path,
            _value=ExtractionState(
                graphpatch_path,
                path,
                data,
                {name: child._value for name, child in children.items()},
            ),
        )


def extract(
    root_module: Module,
    options: ExtractionOptions,
    *trace_args: Any,
    **trace_kwargs: Any,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    extraction_state = ExtractionStateWrapper().wrap(root_module)
    root_state = extraction_state[""]

    # Extract GraphModules from each module in the hierarchy.
    for state in extraction_state.values():
        if is_container(state.wrapped_module):
            state.extracted_module = state.wrapped_module
            continue

        should_compile = not _should_skip_compilation(options, state.original_module)

        if should_compile:
            try:
                if state is not root_state and len(state.invocations) == 0:
                    raise ValueError(
                        f"Unable to compile {state.torch_name}; it was never called when"
                        " evaluating the given example inputs."
                    )

                if state is root_state:
                    args = trace_args
                    kwargs = trace_kwargs
                else:
                    args = state.invocations[-1].args
                    kwargs = state.invocations[-1].kwargs
                state.extracted_module = _run_extraction(state, should_compile, *args, **kwargs)

            except Exception as exc:
                if options.error_on_compilation_failure:
                    raise
                if options.warn_on_compilation_failure:
                    warn(
                        (
                            f"Compilation of {state.torch_name or '<root>'} failed.\n\n"
                            "Torch-generated exception:\n"
                            "**************************\n"
                            f"{exc}"
                            "**************************\n\n"
                            "User code:\n"
                        ),
                        CompilationWarning,
                        # Shows the user code as the call to PatchableGraph, assuming that the
                        # user isn't trying to call extract directly.
                        stacklevel=3,
                    )
                should_compile = False

        # Either we wanted to skip compilation, or we fell back to doing so.
        if not should_compile:
            # We still need to run inference on the root to record module outputs.
            if state is root_state:
                state.extracted_module = _run_extraction(
                    state, should_compile, *trace_args, **trace_kwargs
                )
            else:
                state.extracted_module = OpaqueGraphModule(
                    state.wrapped_module._graphpatch_wrapped_module,
                    accelerate_hook=state.accelerate_hook,
                )

    # Undo the unrolling of containers performed by compile(), so we'll end up with the same
    # module hierarchy as originally. Reset _modules so we'll additionally restore the original
    # ordering (compile re-orders them to the order in which they are invoked).
    for torch_qual_name, state in extraction_state.items():
        if isinstance(state.extracted_module, CompiledGraphModule):
            retarget_submodule_calls(state)
            state.extracted_module._modules = OrderedDict()
        if torch_qual_name == "":
            continue
        [*parent_path, local_name] = torch_qual_name.split(".")
        parent = ".".join(parent_path)
        parent_module = root_state.extracted_module.get_submodule(parent)
        setattr(parent_module, local_name, state.extracted_module)

    # With the container hierarchy finalized, we can set up additional attributes needed for
    # eventual serialization.
    for state in extraction_state.values():
        if isinstance(state.extracted_module, GraphPatchModule):
            state.extracted_module._set_submodules_for_serialization()

    # Postprocess after all modules have been converted. Reverse order so children are postprocessed
    # before their parents, which matters for cloned graphs.
    for state in reversed(list(extraction_state.values())):
        if isinstance(state.extracted_module, GraphPatchModule):
            _postprocess_graph(state)

    # Escape hatch for modules that torch just refuses to compile correctly. Ideally as
    # compatibility improves we won't need this in the future!
    graph_module = root_state.extracted_module
    if options.postprocessing_function is not None:
        options.postprocessing_function(graph_module, root_module)
        graph_module.recompile()

    return graph_module, wrap_graph_module(graph_module)
