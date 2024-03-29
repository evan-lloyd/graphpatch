import inspect
import re
import warnings
from collections import OrderedDict, defaultdict, deque
from contextlib import ExitStack
from copy import deepcopy
from typing import (
    Any,
    Deque,
    Dict,
    Generic,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleDict, ModuleList, Sequential

from .. import hacks
from ..meta import (
    GraphMeta,
    NodeData,
    NodeMeta,
    OutputArgumentIndex,
    wrap_graph_module,
    wrap_output_argument_index,
)
from ..optional.accelerate import add_hook_to_module
from .compiled_graph_module import CompiledGraphModule, compile_module
from .extraction_context import (
    DeduplicationWrapper,
    ExtractionState,
    ModuleInvocation,
    compilation_context,
    deduplication_context,
    root_context,
)
from .extraction_options import ExtractionOptions
from .graphpatch_module import GraphPatchModule
from .invocation_tracking_module_list import InvocationTrackingModuleList
from .opaque_graph_module import OpaqueGraphModule, SubmoduleWrapper

CONTAINER_TYPES = (ModuleList, ModuleDict)


def init_container(container: Union[ModuleList, ModuleDict]):
    if isinstance(container, ModuleList):
        return container.__class__([Module() for _ in range(len(container))])
    return container.__class__()


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


def is_container(module: Union[Module, Type[Module]]) -> bool:
    if isinstance(module, type):
        model_class = module
    else:
        model_class = module.__class__
    return model_class in CONTAINER_TYPES


def retarget_submodule_calls(graph_module: GraphPatchModule):
    """compile() unrolls all containers, modifying the module hierarchy. Our later code is much
    simplified if we instead retain the original hierarchy. This also looks nicer when the user
    inspects the returned GraphModule.
    """
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        target_module = graph_module.get_submodule(node.target)
        if isinstance(target_module, DeduplicationWrapper):
            node.target = target_module._graphpatch_original_module_name


def _repair_input_signature(state: ExtractionState):
    """Compilation lifts some module attributes into arguments to forward, and has inconsistent
    handling of keyword arguments. We need to restore the original input signature so that calls
    to subgraphs behave correctly.
    """
    # kwargs aren't included in the GraphModule call signature by default; often they are either
    # ignored or converted into positional arguments. Inspect the original forward() method to
    # repair the correct signature.
    graph_module = state.extracted_module
    insert_after = None
    placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
    if placeholders:
        insert_after = placeholders[-1]
    forward_parameters = inspect.signature(state.original_module.forward).parameters
    for name, parameter in forward_parameters.items():
        existing_placeholder = next((p for p in placeholders if p.target == name), None)
        # Argument was created positionally; convert into a kwarg.
        if existing_placeholder:
            if parameter.default is not inspect._empty:
                existing_placeholder.args = (parameter.default,)
        # Argument wasn't created at all, so add it.
        else:
            # Put new placeholder nodes at the end of the placeholder section to maintain ordering.
            if insert_after:
                insertion_context = graph_module.graph.inserting_after(insert_after)
            # No previous placeholder nodes; make sure we start at the very beginning of the graph.
            else:
                insertion_context = graph_module.graph.inserting_before(None)
            with insertion_context:
                insert_after = graph_module.graph.placeholder(
                    name=name,
                    default_value=parameter.default,
                    type_expr=parameter.annotation,
                )

    # Compilation skips non-parameter attributes on models, converting them into placeholders.
    # We can detect this by finding placeholders with no corresponding entry in the inspect
    # signature, and then replace them with a get_attr.
    for placeholder in placeholders:
        if placeholder.target in forward_parameters:
            continue

        # Strip initial "self_" from the attribute name. For torch >= 2.1 we have already done this
        # via some monkeypatching during compilation.
        if hacks.TORCH_VERSION < (2, 1):
            original_attribute_name = placeholder.target[5:]
        else:
            original_attribute_name = placeholder.target

        setattr(
            graph_module,
            placeholder.target,
            getattr(state.original_module, original_attribute_name),
        )
        placeholder.op = "get_attr"


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
        setattr(
            graph_module,
            module_name,
            ModuleList([submodule] + [clone_graph_module(submodule) for _ in calling_nodes[1:]]),
        )
        # Update the call_module nodes to refer to a specific instance in the list.
        for i, node in enumerate(calling_nodes):
            node.target = f"{module_name}.{i}"
        graph_module._graphpatch_module_containers[module_name] = (
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
            submodule_node.name = f"{name}_0"
            for i in range(1, len(child_state.invocations)):
                with graph_module.graph.inserting_after(submodule_node):
                    submodule_node = graph_module.graph.call_function(
                        SubmoduleWrapper(f"{name}.{i}")
                    )
                    submodule_node.name = f"{name}_{i}"
            graph_module._graphpatch_module_containers[name] = (
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


def postprocess_graph(state: ExtractionState):
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


def extract(
    root_module: Module,
    options: ExtractionOptions,
    *trace_args: Any,
    **trace_kwargs: Any,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    extraction_state: Dict[str, ExtractionState] = {
        (graphpatch_name := hacks.override_reserved_name(name)): ExtractionState(
            graphpatch_name, name, submodule
        )
        for name, submodule in root_module.named_modules(remove_duplicate=False)
    }
    root_state = extraction_state[""]
    # Root doesn't get a tracer hook, so we need to record its invocation manually.
    root_state.invocations = [ModuleInvocation(trace_args, trace_kwargs, None)]

    # Set up parent/child relationship between state items.
    for name, state in extraction_state.items():
        if name == "":
            continue
        [*parent_path, local_name] = name.split(".")
        parent_name = ".".join(parent_path)
        extraction_state[parent_name].children[local_name] = state

    # Convert the module hierarchy into GraphModules.
    for state in extraction_state.values():
        if is_container(state.original_module):
            state.extracted_module = init_container(state.original_module)
            continue

        should_compile = not _should_skip_compilation(options, state.original_module)

        if should_compile:
            with ExitStack() as context_stack:
                try:
                    if len(state.invocations) == 0:
                        raise ValueError(
                            f"Unable to compile {state.torch_name}; it was never called when"
                            " evaluating the given example inputs."
                        )

                    # TODO: refactor so order is not important (we need to root_context *after*
                    # any wrapping that might happen elsewhere so our tracers are on correct modules)
                    if state.original_module is root_module:
                        context_stack.enter_context(root_context(extraction_state))
                    else:
                        context_stack.enter_context(deduplication_context(state))
                    context_stack.enter_context(compilation_context(state))
                    state.extracted_module, state.invocations[-1].output = compile_module(
                        state.wrapped_module,
                        *state.invocations[-1].args,
                        **state.invocations[-1].kwargs,
                    )

                except Exception as exc:
                    if options.error_on_compilation_failure:
                        raise
                    print(f"Warning: compilation failed due to {exc}")
                    should_compile = False

        # Either we wanted to skip compilation, or we fell back to doing so.
        if not should_compile:
            # We still need to run inference on the root to record module invocations.
            if state.original_module is root_module:
                with root_context(extraction_state):
                    state.invocations[-1].output = state.wrapped_module(
                        *state.invocations[-1].args, **state.invocations[-1].kwargs
                    )
            state.extracted_module = OpaqueGraphModule(state.wrapped_module)

    for torch_qual_name, state in extraction_state.items():
        # Undo the unrolling of containers performed by compile(), so we'll end up with the same
        # module hierarchy as originally. Reset _modules so we'll additionally restore the original
        # ordering.
        if isinstance(state.extracted_module, CompiledGraphModule):
            retarget_submodule_calls(state.extracted_module)
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
            state.extracted_module._set_containers_for_serialization()

    # Postprocess after all modules have been converted. Reverse order so children are postprocessed
    # before their parents, which matters for cloned graphs.
    for state in reversed(extraction_state.values()):
        if isinstance(state.extracted_module, GraphPatchModule):
            postprocess_graph(state)
        if state.accelerate_hook is not None and state.original_module is not root_module:
            add_hook_to_module(state.extracted_module, state.accelerate_hook)

    # Escape hatch for modules that torch just refuses to compile correctly. Ideally as
    # compatibility improves we won't need this in the future!
    graph_module = root_state.extracted_module
    if options.postprocessing_function is not None:
        options.postprocessing_function(graph_module, root_module)
        graph_module.recompile()

    # Must happen *after* recompile, since that changes forward()
    if root_state.accelerate_hook is not None:
        add_hook_to_module(graph_module, root_state.accelerate_hook)

    return graph_module, wrap_graph_module(graph_module)
