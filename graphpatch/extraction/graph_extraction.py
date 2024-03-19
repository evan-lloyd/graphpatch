import inspect
import re
import warnings
from collections import defaultdict, deque
from contextlib import ExitStack
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type, Union

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
    ExtractionState,
    ModuleInvocation,
    compilation_context,
    root_context,
)
from .extraction_options import ExtractionOptions
from .graphpatch_module import GraphPatchModule
from .invocation_tracking_module_list import InvocationTrackingModuleList
from .opaque_graph_module import OpaqueGraphModule, SubmoduleWrapper

CONTAINER_TYPES = (ModuleList, ModuleDict, Sequential)


def init_container(container: Union[ModuleList, ModuleDict, Sequential]):
    if isinstance(container, ModuleList):
        return container.__class__([Module() for _ in range(len(container))])
    # We treat Sequential as a container (even though it has its own forward) because compile()
    # is hard-coded to unroll it.
    elif isinstance(container, Sequential):
        return container.__class__(*[Module() for _ in range(len(container))])
    return container.__class__()


def match_shape(indexes: NodeData[OutputArgumentIndex], *args: Any) -> Any:
    if not indexes._value.should_unwrap:
        return args[0]

    """Unflatten the args to an output node to match the original output shape."""
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


def canonical_module_name(name: str):
    # Need to mirror torch.compile() behavior of renaming modules that start with "_foo" as
    # "sub_foo".
    return re.sub(
        r"(\.|^)_",
        lambda m: f"{m.group(1) or ''}sub_",
        name,
    )
    # return re.sub(
    #     r"(\.|^)([^a-zA-Z])",
    #     lambda m: f"{m.group(1) or ''}sub_{m.group(2) if m.group(2) != '_' else ''}",
    #     name,
    # )


def postprocess_graph(state: ExtractionState):
    graph_module = state.extracted_module
    # Compilation flattens the arguments to the output node into a single tuple, which changes the
    # signature of the function. This is problematic when we have nested modules that may assume
    # a particular shape! We can hack around this by injecting a function that generates the correct
    # shape and returning its output.
    # NB: we add an attribute to the graph_module rather than passing it to our function via closure
    # because the latter method would not be picklable
    setattr(
        graph_module,
        "_graphpatch_output_indexes",
        wrap_output_argument_index(
            state.invocations[-1].output,
            set(
                id(v.invocations[-1].output)
                for v in state.children.values()
                if len(v.invocations) > 0
            ),
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

    # kwargs aren't included in the GraphModule call signature by default; often they are either
    # ignored or converted into positional arguments. Inspect the original forward() method to
    # repair the correct signature.
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

    # Submodules can be used more than once, but this will cause problems later when we manipulate
    # the graph, since the user may want to independently patch activations in each instance. To
    # simplify things, here we clone the graph modules (but not their parameters!), so we'll have
    # independent graphs to work with.
    duplicate_graph_modules = defaultdict(list)
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            duplicate_graph_modules[getattr(graph_module, node.target)].append(node)
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
            submodule_node.target.module_name = f"{name}.0"
            for i in range(1, len(child_state.invocations)):
                with graph_module.graph.inserting_after(submodule_node):
                    submodule_node = graph_module.graph.call_function(
                        SubmoduleWrapper(f"{name}.{i}")
                    )
            graph_module._graphpatch_module_containers[name] = (
                InvocationTrackingModuleList,
                tuple(str(i) for i in range(len(child_state.invocations))),
            )

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
        (state_name := canonical_module_name(name)): ExtractionState(state_name, submodule)
        for name, submodule in root_module.named_modules(remove_duplicate=False)
    }
    # Root doesn't get a tracer hook, so we need to record its invocation manually.
    extraction_state[""].invocations = [ModuleInvocation(trace_args, trace_kwargs, None)]

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
                    # TODO: refactor so order is not important
                    if state.original_module is root_module:
                        context_stack.enter_context(root_context(extraction_state))
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

    # Set up the GraphModule hierarchy.
    for torch_qual_name, state in extraction_state.items():
        if torch_qual_name == "":
            continue
        [*parent_path, local_name] = torch_qual_name.split(".")
        parent = ".".join(parent_path)
        parent_module = extraction_state[""].extracted_module.get_submodule(parent)

        setattr(parent_module, local_name, state.extracted_module)

    # Unroll containers in compiled modules to match the compiled code.
    for state in extraction_state.values():
        if not isinstance(state.extracted_module, CompiledGraphModule):
            continue
        child_queue = deque(state.extracted_module.named_children())
        while child_queue:
            child_name, child = child_queue.popleft()
            if is_container(child):
                child_queue.extend((f"{child_name}.{n}", m) for n, m in child.named_children())
                # Only need to pop containers that are direct descendants; nested containers
                # will get popped when the root is popped.
                if "." not in child_name:
                    delattr(state.extracted_module, child_name)
            else:
                setattr(state.extracted_module, child_name.replace(".", "_"), child)

    # Postprocess after all modules have been converted. Reverse order so children are postprocessed
    # before their parents, which matters for cloned graphs.
    for state in reversed(extraction_state.values()):
        if isinstance(state.extracted_module, GraphPatchModule):
            postprocess_graph(state)
        if state.accelerate_hook is not None and state.original_module is not root_module:
            add_hook_to_module(state.extracted_module, state.accelerate_hook)

    # Escape hatch for modules that torch just refuses to compile correctly. Ideally as
    # compatibility improves we won't need this in the future!
    graph_module = extraction_state[""].extracted_module
    if options.postprocessing_function is not None:
        options.postprocessing_function(graph_module, root_module)
        graph_module.recompile()

    # Must happen *after* recompile, since that changes forward()
    if extraction_state[""].accelerate_hook is not None:
        add_hook_to_module(graph_module, extraction_state[""].accelerate_hook)

    return graph_module, wrap_graph_module(graph_module)
