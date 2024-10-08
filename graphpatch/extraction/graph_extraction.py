import inspect
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, Optional, Tuple, Union, cast
from warnings import warn

from torch.fx import Graph, Node
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleDict, ModuleList

from .. import hacks
from ..exceptions import GraphPatchException, GraphPatchWarning
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
    ExtractionMethod,
    ExtractionState,
    ExtractionWrapper,
    compilation_context,
    detach_accelerate_hooks,
    is_container,
)
from .extraction_options import ExtractionOptions
from .graphpatch_module import GraphPatchModule
from .multiply_invoked_module import MultiplyInvokedModule
from .opaque_graph_module import OpaqueGraphModule, SubmoduleWrapper


class UnusedModule(GraphPatchModule):
    def __init__(self) -> None:
        super().__init__(Module(), Graph(), "UnusedModule")
        self._graphpatch_output_indexes = OutputArgumentIndex(None, False)
        self._graphpatch_submodules = {}


class CompilationError(GraphPatchException):
    pass


class NoRecordedInvocations(CompilationError):
    pass


class CompilationWarning(GraphPatchWarning):
    pass


def match_shape(indexes: NodeData[OutputArgumentIndex], *args: Any) -> Any:
    """Unflatten the args to an output node to match the original output shape."""
    if indexes._value is NodeData.Sentinels._NO_VALUE or not indexes._value.should_unwrap:
        return args[0]

    return indexes.map(
        lambda _, index: args[index.index] if isinstance(index.index, int) else NodeData._NO_VALUE
    ).unwrap()  # type: ignore


def _clone_module(
    module: Union[GraphPatchModule, ModuleList, ModuleDict]
) -> Union[GraphPatchModule, ModuleList, ModuleDict]:
    if isinstance(module, OpaqueGraphModule):
        return OpaqueGraphModule(module)
    elif isinstance(module, CompiledGraphModule):
        return CompiledGraphModule(module, deepcopy(module.graph), "CompiledGraphModule")
    elif isinstance(module, ModuleList):
        return type(module)([Module() for _ in range(len(module))])
    elif isinstance(module, ModuleDict):
        return type(module)()

    # Should not happen, but things will be broken if it does.
    raise ValueError("Internal GraphPatch error: unexpected module class in _clone_module.")


def _clone_module_with_submodules(
    module: Union[GraphPatchModule, ModuleList, ModuleDict]
) -> Union[GraphPatchModule, ModuleList, ModuleDict]:
    clones = {name: _clone_module(submodule) for name, submodule in module.named_modules()}
    for name in reversed(clones.keys()):
        if name == "":
            continue
        [*parent_path, local_name] = name.split(".")
        parent = clones[""].get_submodule(".".join(parent_path))
        setattr(parent, local_name, clones[name])

    return clones[""]


def _should_skip_compilation(options: ExtractionOptions, module: Module) -> bool:
    return options.skip_compilation or type(module) in options.classes_to_skip_compiling


def _retarget_submodule_calls(state: ExtractionState) -> None:
    """compile() unrolls all containers, modifying the module hierarchy. Our later code is much
    simplified if we instead retain the original hierarchy. This also looks nicer when the user
    inspects the returned GraphModule.
    """
    graph_module = state.extracted_module
    assert graph_module is not None
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


def _repair_input_signature(state: ExtractionState) -> None:
    """Compilation lifts some module attributes into arguments to forward, and has inconsistent
    handling of keyword arguments. We need to restore the original input signature so that calls
    to subgraphs behave correctly.
    """
    graph_module = state.extracted_module
    assert graph_module is not None
    insert_after = graph_module.graph._root
    existing_placeholders = {n.target: n for n in graph_module.graph.nodes if n.op == "placeholder"}
    with detach_accelerate_hooks(state.wrapped_module._graphpatch_wrapped_module):
        forward_parameters = inspect.signature(
            state.wrapped_module._graphpatch_wrapped_module.forward
        ).parameters
    canonical_placeholders: Dict[str, Node] = {}
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
                target = f"*{name}"
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
            if target in existing_placeholders:
                hacks.replace_node_keeping_original_name(
                    existing_placeholders[target], new_placeholder, name
                )
                del existing_placeholders[target]
            else:
                hacks.insert_node(new_placeholder)
                # Edge case for varargs/kwargs nodes
                canonical_placeholders[target.replace("*", "")] = new_placeholder
            insert_after = new_placeholder

    # Any remaining placeholders do not appear in the function signature. There are two known causes
    # for this:
    # 1) Lifted attributes; generally, non-Tensor attributes on the module that get converted into
    #    arguments to the GraphModule.
    # 2) Weird handling of container typed arguments. For example, each element of a tuple will get
    #    split into separate arguments, which are attributed to a LocalSource. (My guess is this
    #    happens because compile() assumes that it is going to flatten everything into one module,
    #    but because we override that it ends up just making no sense.)
    attribute_nodes: Dict[str, Node] = {}
    for name, placeholder in existing_placeholders.items():
        root_source = placeholder.meta["_graphpatch_placeholder_source"]
        source = root_source
        # Not all lifted attributes end up having an AttrSource for some reason. In that case
        # fall back to handling by name.
        attribute_name = placeholder.target
        replacement_node = None
        while True:
            if isinstance(source, hacks.AttrSource):
                attribute_name = source.member
                break
            # Happens for container inputs that aren't used directly, but their members are.
            elif isinstance(source, hacks.LocalSource):
                attribute_name = source.local_name
                attribute_nodes[attribute_name] = canonical_placeholders[attribute_name]
                break

            if hasattr(source, "base"):
                source = source.base
            # torch 2.0
            elif hasattr(source, "inner"):
                source = source.inner
            else:
                break

        if attribute_name not in attribute_nodes:
            with graph_module.graph.inserting_after(insert_after):
                attribute_nodes[attribute_name] = Node(
                    graph_module.graph,
                    attribute_name,
                    "get_attr",
                    attribute_name,
                    (),
                    {},
                )
                replacement_node = attribute_nodes[attribute_name]
                hacks.insert_node(attribute_nodes[attribute_name])
                insert_after = attribute_nodes[attribute_name]
                setattr(
                    graph_module,
                    attribute_name,
                    getattr(state.wrapped_module._graphpatch_wrapped_module, attribute_name),
                )
        # TODO: we should be able to handle arbitrary nesting, but I'm not even sure compile()
        # can, so putting that off for now.
        if isinstance(root_source, hacks.GetItemSource):
            with graph_module.graph.inserting_after(insert_after):
                get_item_node = Node(
                    graph_module.graph,
                    "getitem",
                    "call_method",
                    "__getitem__",
                    (attribute_nodes[attribute_name], root_source.index),
                    {},
                )
                insert_after = get_item_node
                replacement_node = get_item_node
                hacks.insert_node(get_item_node)

        if replacement_node is not None:
            hacks.replace_node_keeping_original_name(
                placeholder, replacement_node, replacement_node.name
            )


def _repair_output_signature(state: ExtractionState) -> None:
    """Compilation flattens the arguments to the output node into a single tuple, which changes the
    signature of the function. This is problematic when we have nested modules that may assume
    a particular shape! We can hack around this by injecting a function that generates the correct
    shape and returning its output.
    NB: we add an attribute to the graph_module rather than passing it to our function via closure
    because the latter method would not be picklable.
    """
    graph_module = state.extracted_module
    assert graph_module is not None
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


def _clone_repeated_submodules(state: ExtractionState) -> None:
    """Submodules can be used more than once, but this will cause problems later when we manipulate
    the graph, since the user may want to independently patch activations in each instance. To
    simplify things, here we clone the graph modules (but not their parameters!), so we'll have
    independent graphs to work with.
    """
    graph_module = state.extracted_module
    assert isinstance(graph_module, GraphPatchModule)
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
            MultiplyInvokedModule(
                [submodule] + [_clone_module_with_submodules(submodule) for _ in calling_nodes[1:]]
            ),
        )
        # Update the call_module nodes to refer to a specific instance in the list.
        for i, node in enumerate(calling_nodes):
            node.target = f"{module_name}.{i}"
        graph_module._graphpatch_submodules[module_name] = (
            MultiplyInvokedModule,
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
                MultiplyInvokedModule(
                    [submodule]
                    + [
                        _clone_module_with_submodules(submodule)
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
                MultiplyInvokedModule,
                tuple(str(i) for i in range(len(child_state.invocations))),
            )


def _standardize_submodule_nodes(state: ExtractionState) -> None:
    """compile() mangles the names of nodes calling submodules that live in containers. Here we
    standardize to something much more user-friendly.
    """
    graph_module = state.extracted_module
    assert graph_module is not None
    for node in graph_module.graph.nodes:
        if node.op != "call_module":
            continue
        node.name = node.target.replace(".", "_")
        if not node.name.isidentifier():
            node.name = f"sub_{node.name}"


def _postprocess_graph(state: ExtractionState) -> None:
    """Clean up the extracted graph to match the original module's signature and standardize node
    names.
    """
    graph_module = state.extracted_module
    assert graph_module is not None
    _repair_output_signature(state)
    _clone_repeated_submodules(state)
    _standardize_submodule_nodes(state)

    # Hack for some weirdness around dynamic shape detection (currently only seen in GPT2-XL)
    for node in graph_module.graph.nodes:
        hacks.maybe_replace_dynamo_get_item_lambda(node)

    graph_module.recompile()


def _run_extraction(
    state: ExtractionState,
    *args: Any,
    **kwargs: Any,
) -> GraphPatchModule:
    should_trace = state.name == ""
    for submodule in state.wrapped_module.modules():
        if isinstance(submodule, ExtractionWrapper):
            submodule._graphpatch_record_invocations = should_trace
            # Clear invocations in case of fallback after partially compiling the root module.
            if should_trace:
                submodule._graphpatch_extraction_state.invocations = []

    if (
        state.extraction_method is ExtractionMethod.custom
        and state.custom_extraction_function is not None
    ):
        graph = state.custom_extraction_function(state.wrapped_module._graphpatch_wrapped_module)
        extracted_module: GraphPatchModule = CompiledGraphModule(
            state.wrapped_module._graphpatch_wrapped_module,
            graph,
            "CompiledGraphModule",
            state.accelerate_hook,
        )
    elif state.extraction_method is ExtractionMethod.compiled:
        with compilation_context(state):
            extracted_module = compile_module(state.wrapped_module, *args, **kwargs)
            extracted_module._graphpatch_accelerate_hook = state.accelerate_hook
        # compile_module will already have run inference.
        should_trace = False
    else:
        extracted_module = OpaqueGraphModule(
            state.wrapped_module._graphpatch_wrapped_module, accelerate_hook=state.accelerate_hook
        )

    if should_trace:
        state.wrapped_module(*args, **kwargs)

    return extracted_module


class ExtractionStateWrapper(NodeDataWrapper[ExtractionState]):
    def __init__(self, options: ExtractionOptions):
        super().__init__()
        self.options = options

    def handle_wrap(self, data: Module, path: str) -> NodeData[ExtractionState]:
        custom_extraction_function = self.options.custom_extraction_functions.get(type(data))
        if custom_extraction_function is not None:
            extraction_method = ExtractionMethod.custom
        elif not _should_skip_compilation(self.options, data):
            extraction_method = ExtractionMethod.compiled
        else:
            extraction_method = ExtractionMethod.opaque
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
            _original_type=type(data),
            _children=children or NodeData._NO_VALUE,
            _path=graphpatch_path,
            _value=ExtractionState(
                graphpatch_path,
                path,
                data,
                {name: cast(ExtractionState, child._value) for name, child in children.items()},
                extraction_method,
                custom_extraction_function,
            ),
        )


def _process_extraction_options(options: ExtractionOptions) -> ExtractionOptions:
    new_options = deepcopy(options)
    # TODO: we should be able to make weight patchable for OpaqueGraphModule without making it a
    # special case
    hacks.maybe_add_8_bit_linear_custom_compilation(new_options)
    return new_options


@contextmanager
def _handle_compilation_failure(
    state: ExtractionState, options: ExtractionOptions
) -> Iterator[None]:
    try:
        yield
    except Exception as exc:
        if isinstance(exc, NoRecordedInvocations) and options.allow_unused_submodules:
            # Allow this failure and fall back to OpaqueGraphModule?
            pass
        # No fallback for OpaqueGraphModule failure.
        elif (
            options.error_on_compilation_failure
            or state.extraction_method is ExtractionMethod.opaque
        ):
            raise

        if options.warn_on_compilation_failure:
            if state.extraction_method is ExtractionMethod.custom:
                method_description = "Custom extraction function"
            else:
                method_description = "Compilation"
            warn(
                (
                    f"{method_description} for {state.torch_name or '<root>'} failed.\n\n"
                    "Exception:\n"
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

        # Fall back to OpaqueGraphModule.
        state.extraction_method = ExtractionMethod.opaque
        state.extracted_module = None


def extract(
    root_module: Module,
    options: ExtractionOptions,
    *trace_args: Any,
    **trace_kwargs: Any,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    options = _process_extraction_options(options)
    extraction_state = ExtractionStateWrapper(options).wrap(root_module)
    root_state = cast(ExtractionState, extraction_state[""])

    # Extract GraphModules from each module in the hierarchy.
    for state in extraction_state.values():
        if is_container(state.wrapped_module):
            state.extracted_module = state.wrapped_module
            continue

        # Attempt graph extraction, with the following priority on extraction methods:
        # 1) CompiledGraphModule from custom extraction function. Fall back to 3.
        # 2) Extract CompiledGraphModule using torch.compile(). Fall back to 3.
        # 3) Wrap with OpaqueGraphModule.
        def do_extraction() -> None:
            if state.extraction_method is ExtractionMethod.custom:
                args: Any = []
                kwargs: Any = {}
            elif state is root_state or state.extraction_method is ExtractionMethod.opaque:
                args = trace_args
                kwargs = trace_kwargs
            else:
                if len(state.invocations) == 0:
                    raise NoRecordedInvocations(
                        f"Unable to compile {state.torch_name}; it was never called when"
                        " evaluating the given example inputs."
                    )
                args = state.invocations[-1].args
                kwargs = state.invocations[-1].kwargs
            state.extracted_module = _run_extraction(state, *args, **kwargs)

        with _handle_compilation_failure(state, options):
            # TODO: we should refactor so that all postprocessing can be done here, so we can
            # more reliably fall back to opaque.
            do_extraction()
            _repair_input_signature(state)

        # Fall back to opaque if we failed to compile.
        if state.extracted_module is None:
            do_extraction()
            _repair_input_signature(state)

    # For the type-checker's benefit; this is a cannot-happen.
    assert isinstance(root_state.extracted_module, GraphPatchModule)

    # Undo the unrolling of containers performed by compile(), so we'll end up with the same
    # module hierarchy as originally. Reset _modules so we'll additionally restore the original
    # ordering (compile re-orders them to the order in which they are invoked).
    for torch_qual_name, state in extraction_state.items():
        if isinstance(state.extracted_module, CompiledGraphModule):
            _retarget_submodule_calls(state)
            state.extracted_module._modules = OrderedDict()
        if torch_qual_name == "":
            continue
        [*parent_path, local_name] = torch_qual_name.split(".")

        parent_module: Module = root_state.extracted_module
        non_container_parent: Module = root_state.extracted_module

        # We should ignore the entire hierarchy under any UnusedModules.
        for child_name in parent_path:
            if isinstance(parent_module, UnusedModule):
                break
            child = cast(Module, parent_module._modules[child_name])
            parent_module = child
            if not is_container(child):
                non_container_parent = child
        if isinstance(parent_module, UnusedModule):
            continue

        assert state.extracted_module is not None
        # Replace unusued submodules with dummies, unless the parent is opaque. For compiled
        # modules, we know the submodule is definitely never going to be used, since we didn't write
        # any instructions that would actually use it. For opaque modules, it could be the case that
        # with different data the module would be used, so we should keep it around to be safe.
        if (
            len(state.invocations) == 0
            and not is_container(state.extracted_module)
            and isinstance(non_container_parent, CompiledGraphModule)
        ):
            setattr(parent_module, local_name, UnusedModule())
        else:
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
