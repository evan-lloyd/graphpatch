import inspect
import re
import warnings
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import torch
from torch._dynamo.allowed_functions import (
    _allowed_function_ids,
    _disallowed_function_ids,
)
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleDict, ModuleList, Sequential

from .. import hacks
from ..meta import (
    GraphMeta,
    NodeData,
    NodeMeta,
    wrap_graph_module,
    wrap_output_argument_index,
)
from ..optional.accelerate import ModelHook, add_hook_to_module
from ..optional.dataclasses import dataclass, field
from .accelerate import detach_accelerate_hooks
from .bitsandbytes import wrap_bits_and_bytes
from .extraction_options import ExtractionOptions
from .opaque_graph_module import OpaqueGraphModule

CONTAINER_TYPES = (Sequential, ModuleList, ModuleDict)


class CompiledGraphModule(GraphModule):
    pass


def match_shape(indexes: NodeData[int], *args: Any) -> Any:
    """Unflatten the args to an output node to match the original output shape."""
    return indexes.map(
        lambda _, index: args[index] if isinstance(index, int) else NodeData._NO_VALUE
    ).unwrap()


@contextmanager
def eval_mode(module: Module) -> Iterator[None]:
    """Set a module into eval mode, so we skip including training-only things like dropouts in
    our graph.
    """
    eval_state = module.training

    if eval_state:
        module.eval()

    yield

    if eval_state:
        module.train()


@contextmanager
def allow_modules(modules: List[type], module_class: type) -> Iterator[None]:
    """Use the undocumented _allowed_function_ids to prevent compile() from inlining the child
    modules, so we can independently compile them into separate GraphModules.
    """

    module_ids = list(map(id, modules))
    ids_to_remove = [id for id in module_ids if id not in _allowed_function_ids.function_ids]
    module_filename = getattr(inspect.getmodule(module_class), "__file__", None)
    remove_skipfile = None

    # Let us compile even torch builtins
    if module_filename and module_filename not in torch._dynamo.skipfiles.FILENAME_ALLOWLIST:
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.add(module_filename)
        remove_skipfile = module_filename
    remove_id = None
    if id(module_class) in _allowed_function_ids.function_ids:
        remove_id = id(module_class)
        _allowed_function_ids.remove(remove_id)
        _disallowed_function_ids.add(remove_id)

    for _id in module_ids:
        _allowed_function_ids.add(_id)
    try:
        yield
    finally:
        for _id in ids_to_remove:
            _allowed_function_ids.remove(_id)
        if remove_id is not None:
            _allowed_function_ids.add(remove_id)
            _disallowed_function_ids.remove(remove_id)
        if remove_skipfile is not None:
            torch._dynamo.skipfiles.FILENAME_ALLOWLIST.remove(remove_skipfile)


@dataclass
class ArgTracker:
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    seen_outputs: Set[int] = field(default_factory=set)


@dataclass(kw_only=True)
class CompilationState:
    self_args: ArgTracker = field(default_factory=ArgTracker)
    child_state: Dict[str, "CompilationState"] = field(default_factory=dict)
    accelerate_hook: Optional[ModelHook] = None
    original_module: Module
    local_name: Optional[str] = None
    parent_name: Optional[str] = None


@dataclass
class GraphModuleWrapper:
    graph_module: Optional[GraphModule] = None


@contextmanager
def tracer_hook(
    module: Module, arg_tracker: ArgTracker, accelerate_hook: Optional[ModelHook]
) -> Iterator[None]:
    # compile() calls each module twice, but the first pass has FakeTensors, which we don't want to
    # trace. We need real example inputs for the recursive compile() calls, which fortunately get
    # passed in the second call.

    def pre_hook(module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
        if accelerate_hook is not None:
            args, kwargs = accelerate_hook.pre_forward(module, *args, **kwargs)

        # Note that there is an edge case where the same module instance is called multiple times in
        # its parent's code. Here we are implicitly assuming that the arguments passed in the
        # *last* invocation can be used for all invocations, which may not be the case, though this
        # seems unlikely in practice. We could handle this by compiling different versions of the
        # module per invocation.
        arg_tracker.args = list(args)
        arg_tracker.kwargs = kwargs

        return (args, kwargs)

    def post_hook(
        module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> Any:
        if accelerate_hook is not None:
            output = accelerate_hook.post_forward(module, output)
        arg_tracker.output = output
        # Mark any containers in the output as "seen" so we can short-circuit output shape tracking
        # appropriately in the parent of this module.
        sub_output_stack = [output]
        while sub_output_stack:
            cur_output = sub_output_stack.pop()
            if isinstance(cur_output, (tuple, list)):
                arg_tracker.seen_outputs.add(id(cur_output))
                sub_output_stack.extend(cur_output)
            elif isinstance(cur_output, dict):
                arg_tracker.seen_outputs.add(id(cur_output))
                sub_output_stack.extend(cur_output.values())
        return output

    pre_handle = None
    post_handle = None
    try:
        pre_handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        post_handle = module.register_forward_hook(post_hook, with_kwargs=True)
        yield
    finally:
        if pre_handle is not None:
            pre_handle.remove()
        if post_handle is not None:
            post_handle.remove()


def is_container(module: Union[Module, Type[Module]]) -> bool:
    if isinstance(module, type):
        model_class = module
    else:
        model_class = module.__class__
    return model_class in CONTAINER_TYPES


def postprocess_graph(
    graph_module: GraphModule, original_module: Module, compilation_state: CompilationState
):
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
            compilation_state.self_args.output,
            set(id(v.self_args.output) for v in compilation_state.child_state.values()),
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
    forward_parameters = inspect.signature(original_module.forward).parameters
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

        # Strip initial "self_" from the attribute name.
        if hacks.TORCH_VERSION < (2, 1):
            original_attribute_name = placeholder.target[5:]
        else:
            original_attribute_name = placeholder.target

        setattr(graph_module, placeholder.target, getattr(original_module, original_attribute_name))
        placeholder.op = "get_attr"

    # Submodules can be used more than once, but this will cause problems later when we manipulate
    # the graph, since the user may want to independently patch activations in each instance. To
    # simplify things, here we clone the graph modules (but not their parameters!), so we'll have
    # independent graphs to work with.
    duplicate_graph_modules = defaultdict(list)
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            duplicate_graph_modules[getattr(graph_module, node.target)].append(node)
    for module, calling_nodes in duplicate_graph_modules.items():
        if len(calling_nodes) < 2:
            continue
        # Replace the original module with a ModuleList so we don't have to worry about name
        # collisions.
        module_name = calling_nodes[0].target
        setattr(
            graph_module,
            module_name,
            ModuleList(
                [module] + [GraphModule(module, deepcopy(module.graph)) for _ in calling_nodes[1:]]
            ),
        )
        # Update the call_module nodes to refer to a specific instance in the list.
        for i, node in enumerate(calling_nodes):
            node.target = f"{module_name}.{i}"

    # Hack for some weirdness around dynamic shape detection (currently only seen in GPT2-XL)
    for node in graph_module.graph.nodes:
        hacks.maybe_replace_dynamo_get_item_lambda(node)


def convert_to_graph_module(
    module: Module, compilation_state: CompilationState, skip_compilation: bool, is_root: bool
) -> GraphModule:
    graph_module: Optional[GraphModule] = None

    def callback(gm: GraphModule, *args: Any, **kwargs: Any) -> GraphModule:
        nonlocal graph_module
        graph_module = gm
        # There is no hook to choose a subclass of GraphModule to create during compilation, so
        # dynamically make it a subclass of CompiledGraphModule. GraphModules are always created by
        # torch as the sole instance of a dynamically generated class, so this is safe.
        gm.__class__.__bases__ = (CompiledGraphModule,) + gm.__class__.__bases__
        return gm

    arg_tracker = compilation_state.self_args

    with ExitStack() as tracer_stack, torch.inference_mode(), eval_mode(
        module
    ), wrap_bits_and_bytes(
        module
    ) as maybe_wrapped_module, hacks.dynamo_hacks_for_current_torch_version():
        # This needs to happen after wrapping bits and bytes, so the wrapper class will be added
        # to allow_modules
        tracer_stack.enter_context(
            allow_modules(
                [m.__class__ for m in maybe_wrapped_module.modules() if m != module],
                module_class=module.__class__,
            )
        )

        if is_root:
            submodule_iterator = maybe_wrapped_module.named_modules()
        else:
            submodule_iterator = maybe_wrapped_module.named_children()
        for name, submodule in submodule_iterator:
            hook = tracer_stack.enter_context(detach_accelerate_hooks(submodule))
            if submodule is maybe_wrapped_module:
                compilation_state.accelerate_hook = hook
                continue

            # Need to mirror torch.compile() behavior, which adds this prefix in this situation.
            if not name[0].isalpha():
                name = "sub" + name

            # TODO: handle nested container modules gracefully
            # Bit of a quirk with compile(); the named_modules() iterator returns names like
            # module_list.0.foo, but the resulting modules will be named like module_list_0.foo
            name = re.sub(r"\.(\d+)", lambda match: f"_{match.group(1)}", name)

            [*parent_name, local_name] = name.split(".")
            parent_name = "_".join(parent_name)
            name = name.replace(".", "_")

            if is_root:
                compilation_state.child_state[name] = CompilationState(
                    original_module=submodule, local_name=local_name, parent_name=parent_name
                )
                tracer_stack.enter_context(
                    tracer_hook(
                        submodule,
                        compilation_state.child_state[name].self_args,
                        hook,
                    )
                )

        if not skip_compilation:
            torch._dynamo.reset()
            try:
                compiled_module = torch.compile(backend=callback, dynamic=True, fullgraph=True)(
                    maybe_wrapped_module
                )
                arg_tracker.output = compiled_module(*arg_tracker.args, **arg_tracker.kwargs)
            except Exception as exc:
                print(f"Warning: compilation failed due to {exc}")
                skip_compilation = True
        if skip_compilation:
            # Fall back to running the original module
            arg_tracker.output = module(*arg_tracker.args, **arg_tracker.kwargs)
            graph_module = OpaqueGraphModule(module)

    return graph_module


def extract(
    root_module: Module,
    options: ExtractionOptions,
    *trace_args: Any,
    **trace_kwargs: Any,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    arg_tracker = ArgTracker(list(trace_args), dict(**trace_kwargs), None, set())
    root_compilation_state = CompilationState(self_args=arg_tracker, original_module=root_module)
    graph_modules_by_name = {
        "": convert_to_graph_module(
            root_module,
            root_compilation_state,
            root_module.__class__ in options.classes_to_skip_compiling or options.skip_compilation,
            is_root=True,
        )
    }

    # Turn all children into graph modules themselves.
    for name, state in root_compilation_state.child_state.items():
        module = state.original_module
        if module is root_module or is_container(module):
            continue
        sub_graph = convert_to_graph_module(
            module,
            state,
            module.__class__ in options.classes_to_skip_compiling or options.skip_compilation,
            is_root=False,
        )

        graph_modules_by_name[name] = sub_graph
        setattr(graph_modules_by_name[state.parent_name], state.local_name, sub_graph)

    # Postprocess after all modules have been converted. Reverse order so children are postprocessed
    # before their parents, which matters for cloned graphs.
    for name, state in reversed(root_compilation_state.child_state.items()):
        module = state.original_module
        if is_container(module):
            continue
        graph_module = graph_modules_by_name[name]
        postprocess_graph(graph_module, module, state)
        graph_module.recompile()
        if state.accelerate_hook is not None:
            add_hook_to_module(graph_module, state.accelerate_hook)

    graph_module = graph_modules_by_name[""]

    # Finally, handle root.
    postprocess_graph(graph_module, root_module, root_compilation_state)

    # Escape hatch for modules that torch just refuses to compile correctly. Ideally as
    # compatibility improves we won't need this in the future!
    if options.postprocessing_function is not None:
        options.postprocessing_function(graph_module, root_module)
    graph_module.recompile()

    # Must happen *after* recompile, since that changes forward()
    if root_compilation_state.accelerate_hook is not None:
        add_hook_to_module(graph_module, root_compilation_state.accelerate_hook)

    return graph_module, wrap_graph_module(graph_module)
