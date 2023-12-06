import inspect
import warnings
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import torch
from torch._dynamo.allowed_functions import (
    _allowed_function_ids,
    _disallowed_function_ids,
)
from torch.fx.graph_module import GraphModule
from torch.nn import LayerNorm, Module, ModuleDict, ModuleList, Sequential

from . import hacks
from .meta import (
    GraphMeta,
    NodeData,
    NodeMeta,
    wrap_graph_module,
    wrap_output_argument_index,
)
from .optional.accelerate import ModelHook, add_hook_to_module, remove_hook_from_module
from .optional.bitsandbytes import Linear8bitLt
from .optional.dataclasses import dataclass
from .wrapped_8bit_linear import Wrapped8BitLinear

CONTAINER_TYPES = (Sequential, ModuleList, ModuleDict)
UNCOMPILABLE_BUILTINS = {LayerNorm}


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


def _wrapped_8bit(module: Linear8bitLt) -> Wrapped8BitLinear:
    # TODO: there's a gotcha where if you happened to run any inference on a base module, CB and SCB
    # will have been deleted; any way around this?
    wrapped = Wrapped8BitLinear(module.weight.CB, module.weight.SCB, module.bias)
    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        add_hook_to_module(wrapped, hook)
    return wrapped


@contextmanager
def wrap_bits_and_bytes(module: Module) -> Iterator[Module]:
    """Wrap any bitsandbytes quantized linear modules, since they use Tensor subclasses which are
    incompatible with the current (2.1.0) torch.compile() implementation.
    """
    original_submodules = {}
    original_module = module
    for name, submodule in original_module.named_modules():
        if submodule is module and isinstance(module, Linear8bitLt):
            module = _wrapped_8bit(module)
            continue
        path = name.split(".")
        parent = module.get_submodule(".".join(path[:-1]))
        if isinstance(submodule, Linear8bitLt):
            original_submodules[name] = submodule
            setattr(parent, path[-1], _wrapped_8bit(submodule))
    try:
        yield module
    finally:
        for name, original in original_submodules.items():
            path = name.split(".")
            parent = module.get_submodule(".".join(path[:-1]))
            setattr(parent, path[-1], original)


@contextmanager
def detach_accelerate_hooks(module: Module) -> Iterator[Optional[ModelHook]]:
    """Temporarily detach accelerate's hooks from the module, since they don't play nice with
    torch.compile(). Return the hook object so we can apply it to the compiled graph.
    """

    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        remove_hook_from_module(module)
        # Instance-level forward function doesn't play nice with torch.compile
        del module.forward
    try:
        yield hook
    finally:
        if hook is not None:
            add_hook_to_module(module, hook)


@dataclass
class ArgTracker:
    args: List[Any]
    kwargs: Dict[str, Any]
    output: Any
    seen_outputs: Set[int]


@dataclass
class GraphModuleWraper:
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


def is_container(model: Union[Module, Type[Module]]) -> bool:
    if isinstance(model, type):
        model_class = model
    else:
        model_class = model.__class__
    return model_class in CONTAINER_TYPES


def is_compilable(model: Union[Module, Type[Module]]) -> bool:
    if isinstance(model, type):
        model_class = model
    else:
        model_class = model.__class__
    return model_class not in UNCOMPILABLE_BUILTINS and not is_container(model)


def _extract_impl(
    model: Module,
    arg_tracker: ArgTracker,
    _graphpatch_postprocessing_function: Optional[Callable[[GraphModule, Module], None]] = None,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    # Some pytorch builtins remain uncompilable. TODO: workaround!
    if not is_compilable(model):
        return None, None

    # Wrap this in a mutable object soley to help VSCode detect that graph_module is non-None
    # later in this function.
    graph_module_wrapper = GraphModuleWraper()

    def callback(gm: GraphModule, *args: Any, **kwargs: Any) -> GraphModule:
        graph_module_wrapper.graph_module = gm
        return gm

    module_args: Dict[str, ArgTracker] = {}
    accelerate_hook = None

    with ExitStack() as tracer_stack, torch.inference_mode(), eval_mode(model), wrap_bits_and_bytes(
        model
    ) as maybe_wrapped_model, hacks.dynamo_hacks_for_current_torch_version():
        # This needs to happen after wrapping bits and bytes, so the wrapper class will be added
        # to allow_modules
        tracer_stack.enter_context(
            allow_modules(
                [m.__class__ for m in maybe_wrapped_model.modules() if m != model],
                module_class=model.__class__,
            )
        )
        for name, module in maybe_wrapped_model.named_modules():
            hook = tracer_stack.enter_context(detach_accelerate_hooks(module))
            if module is maybe_wrapped_model:
                accelerate_hook = hook
                continue
            name = name.replace(".", "_")
            # Need to mirror torch.compile() behavior, which adds this prefix in this situation.
            if not name[0].isalpha():
                name = "sub" + name

            module_args[name] = ArgTracker([], {}, None, set())
            tracer_stack.enter_context(
                tracer_hook(
                    module,
                    module_args[name],
                    hook,
                )
            )
        torch._dynamo.reset()
        compiled_model = torch.compile(backend=callback, dynamic=True, fullgraph=True)(
            maybe_wrapped_model
        )
        output_value = compiled_model(*arg_tracker.args, **arg_tracker.kwargs)

    graph_module = graph_module_wrapper.graph_module
    if graph_module is None:
        return None, None

    # Recurse to turn all children into graph modules themselves.
    # TODO: do this iteratively instead of recursively so we can't hit recursion limits for deeply
    # nested modules
    for name, module in graph_module.named_children():
        if module is graph_module or not is_compilable(module):
            continue
        # NB: deliberately not forwarding postprocessing function, since we want to apply it only
        # to the root.
        sub_graph, _ = _extract_impl(module, module_args[name])
        if sub_graph is not None:
            setattr(graph_module, name, sub_graph)

    # Compilation flattens the arguments to the output node into a single tuple, which changes the
    # signature of the function. This is problematic when we have nested modules that may assume
    # a particular shape! We can hack around this by injecting a function that generates the correct
    # shape and returning its output.
    # NB: we add an attribute to the graph_module rather than passing it to our function via closure
    # because the latter method would not be picklable
    setattr(
        graph_module,
        "_graphpatch_output_indexes",
        wrap_output_argument_index(output_value, set(id(v.output) for v in module_args.values())),
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
    forward_parameters = inspect.signature(model.forward).parameters
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

        setattr(graph_module, placeholder.target, getattr(model, original_attribute_name))
        placeholder.op = "get_attr"

    # Submodules can be used more than once, but this will cause problems later when we manipulate
    # the graph, since the user may want to independently patch activations in each instance. To
    # simplify things, here we clone the graph modules (but not their parameters!), so we'll have
    # independent graphs to work with.
    duplicate_graph_modules = defaultdict(list)
    for node in graph_module.graph.nodes:
        if node.op == "call_module" and isinstance(
            target := getattr(graph_module, node.target), GraphModule
        ):
            duplicate_graph_modules[target].append(node)
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

    # Escape hatch for models that torch just refuses to compile correctly. Ideally as compatibility
    # improves we won't need this in the future!
    if _graphpatch_postprocessing_function is not None:
        _graphpatch_postprocessing_function(graph_module, model)

    graph_module.recompile()

    # Must happen *after* recompile, since that changes forward()
    if accelerate_hook is not None:
        add_hook_to_module(graph_module, accelerate_hook)

    return graph_module, wrap_graph_module(graph_module)


def extract(
    model: Module,
    *trace_args: Any,
    _graphpatch_postprocessing_function: Optional[Callable[[GraphModule, Module], None]] = None,
    **trace_kwargs: Any,
) -> Tuple[Optional[GraphModule], Optional[NodeData[Union[GraphMeta, NodeMeta]]]]:
    arg_tracker = ArgTracker(list(trace_args), dict(**trace_kwargs), None, set())
    return _extract_impl(model, arg_tracker, _graphpatch_postprocessing_function)
