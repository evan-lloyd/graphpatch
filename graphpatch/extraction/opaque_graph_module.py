import inspect
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
    Union,
)

from torch import Tensor
from torch.fx.graph import Graph
from torch.nn import Module, ModuleDict, ModuleList, Parameter, Sequential
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX

from .graphpatch_module import GraphPatchModule

if TYPE_CHECKING:
    from ..meta import OutputArgumentIndex

MethodBindingType = Enum("MethodBindingType", ("_none", "_instance", "_class"))

_UNPATCHABLE_MODULE_ATTRIBUTES = frozenset(
    {
        # These attributes need special handling.
        "_modules",
        "_parameters",
        "_buffers",
        "_graphpatch_self",
        # Don't include Module internals.
        *(k for k in Module.__dict__),
        *(k for k in Module.__annotations__),
        "___needs_generation_tag_patch",
        # Python internals not covered by the above.
        "__class__",
    }
)

_UNPATCHABLE_MODULE_METHODS = frozenset(
    {
        # These get monkeypatched in an unpicklable way by dynamo, which would block
        # serialization. Shouldn't actually need them at inference time.
        "__setstate__",
        "__init__",
    }
)


def _module_methods(module: Module) -> Iterator[Callable]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return t[0] not in _UNPATCHABLE_MODULE_METHODS and inspect.isroutine(t[1])

    return filter(_filter, inspect.getmembers(module))


def _module_attributes(module: Module) -> Iterator[Any]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return t[0] not in _UNPATCHABLE_MODULE_ATTRIBUTES and not inspect.isroutine(t[1])

    return filter(_filter, inspect.getmembers(module))


def _method_binding_type(function_or_method: Callable):
    if isinstance(getattr(function_or_method, "__self__", None), type):
        return MethodBindingType._class
    elif hasattr(function_or_method, "__self__"):
        return MethodBindingType._instance
    else:
        return MethodBindingType._none


def _unbound_method(function_or_method: Callable):
    return getattr(function_or_method, "__func__", function_or_method)


class MethodWrapper:
    """Save an unbound version of the given method so we can serialize instance methods without
    also serializing the instance itself.
    """

    function: Callable
    binding: MethodBindingType

    def __init__(self, function_or_method: Callable):
        self.function = _unbound_method(function_or_method)
        self.binding = _method_binding_type(function_or_method)


def _bind_method(module: Module, method: MethodWrapper) -> Callable:
    if method.binding is MethodBindingType._class:
        return MethodType(_unbound_method(method.function), module.__class__)
    elif method.binding is MethodBindingType._instance:
        return MethodType(_unbound_method(method.function), module)
    else:
        return method.function


@contextmanager
def _patched_attributes(module: Module, patches):
    try:
        original = {}
        for name, value in patches.items():
            original[name] = getattr(module, name)
            # Module will throw an exception if we try to overwrite a parameter with a Tensor.
            if isinstance(original[name], Parameter):
                value = Parameter(value)
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


@contextmanager
def _patched_children(module: Module, patch):
    try:
        original = {}
        for name, submodule in module.named_children():
            original[name] = submodule.forward
            submodule.forward = patch(submodule.forward)
        yield
    finally:
        for name, submodule in module.named_children():
            submodule.forward = original[name]


@contextmanager
def _patched_methods(module: Module, patches):
    try:
        nonexistent = object()
        original = {}
        for name, method in patches.items():
            original[name] = getattr(module, name, nonexistent)
            object.__setattr__(module, name, _bind_method(module, method))
        yield
    finally:
        for name in patches.keys():
            original_method = original.get(name, None)
            # Hit an exception before we could record this one.
            if original_method is None:
                continue
            if original_method is nonexistent and hasattr(module, name):
                object.__delattr__(module, name)
            elif original_method is not nonexistent:
                object.__setattr__(module, name, original_method)


class InvocationTrackingModuleList(ModuleList):
    _graphpatch_invocation_index: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graphpatch_invocation_index = 0

    def forward(self, *args, **kwargs) -> Any:
        # TODO: surely any sane module will never vary how many times it calls its submodules
        # and the modulo doesn't matter? But we may want to make this configurable between
        # round-robin or throwing an exception, possibly a global "strict" mode?
        index = self._graphpatch_invocation_index % len(self._modules)
        self._graphpatch_invocation_index = index + 1
        return self[index](*args, **kwargs)


class SubmoduleWrapper:
    """Dummy function to call to place submodules of opaque modules within the graph; the actual
    calls will happen within the opaque module.
    """

    module_name: str

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.__name__ = module_name
        self.__module__ = "opaque_module_submodule"

    def __call__(self, *args, **kwargs) -> None:
        return

    def __str__(self) -> str:
        return self.module_name


_OPAQUE_GRAPH_MODULE_SERIALIZATION_KEYS = (
    "_graphpatch_attribute_names",
    "_graphpatch_num_invocations",
    "_graphpatch_opaque_module_class",
    "_graphpatch_opaque_module_methods",
    "_graphpatch_output_indexes",
    # Not using _graphpatch_self, since we'll create a fresh instance when we deserialize.
)


class OpaqueGraphModule(GraphPatchModule):
    """OpaqueGraphModule constructs a GraphModule from a :class:`torch.nn.Module` without using
    :func:`torch.compile`. This results in a graph that can only be patched at submodule inputs,
    outputs, buffers, parameters, and attributes. An OpaqueGraphModule may have CompiledGraphModules
    as submodules, which can be patched normally.
    """

    _graphpatch_output_indexes: "OutputArgumentIndex"
    _graphpatch_attribute_names: Tuple[str]
    _graphpatch_num_invocations: int
    _graphpatch_opaque_module_class: Type[Module]
    _graphpatch_opaque_module_methods: Dict[str, Callable]
    _graphpatch_self: "OpaqueGraphModule"

    def get_extra_state(self) -> Any:
        return {k: getattr(self, k) for k in _OPAQUE_GRAPH_MODULE_SERIALIZATION_KEYS}

    def set_extra_state(self, state: Any) -> None:
        for k in _OPAQUE_GRAPH_MODULE_SERIALIZATION_KEYS:
            setattr(self, k, state[k])

    def _opaque_module_call(self, *args, **kwargs):
        _graphpatch_args = kwargs.pop("_graphpatch_args", None)

        # Pass the patching arguments down to submodules when in a patching context by
        # monkeypatching their forward() methods. Normally we handle this by manipulating the call
        # site in the parent graph module, but since this is an opaque module we don't have one.
        if _graphpatch_args is not None:

            def pass_patch_args(fn):
                def _inner(*args, **kwargs):
                    return fn(*args, **kwargs, _graphpatch_args=_graphpatch_args)

                return _inner

        else:

            def pass_patch_args(fn):
                return fn

        num_attributes = len(self._graphpatch_attribute_names)

        # TODO: "self" shouldn't be the proxy. maybe a dummy module? or regular object?

        # Set "self" up as a proxy for the original module. We copy over all methods from the
        # original instance (needed if, for example, the original forward calls other methods),
        # but maintain our own attributes, including buffers and parameters.
        with _patched_attributes(
            self, {name: args[i] for i, name in enumerate(self._graphpatch_attribute_names)}
        ), _patched_children(self, pass_patch_args), _patched_methods(
            self, self._graphpatch_opaque_module_methods
        ):
            return self(*args[num_attributes:], **kwargs)

    def __setattr__(self, name: str, value: Union[Tensor, Module]) -> None:
        # Need to bypass Module's behavior, which would try to add _graphpatch_self to submodules.
        if name == "_graphpatch_self":
            return object.__setattr__(self, name, value)
        super().__setattr__(name, value)

    def _construct_graph(self, module: Module) -> Graph:
        graph = Graph()

        # Set up placeholder nodes from module's forward(), skipping first argument (self).
        module_args = {}
        module_kwargs = {}
        for name, arg in list(
            inspect.signature(self._graphpatch_opaque_module_class.forward).parameters.items()
        )[1:]:
            if arg.default is inspect._empty:
                module_args[name] = graph.placeholder(name)
            else:
                module_kwargs[name] = graph.placeholder(name, default_value=arg.default)

        # Add get_attr nodes for buffers, parameters, and any other attributes on the Module class
        # or instance. This allows patching them, and also sets them up to be properly serialized.
        # Skip submodules, which we handle separately.
        self._graphpatch_attribute_names = tuple(
            name for name, value in _module_attributes(module) if not isinstance(value, Module)
        )
        attribute_nodes = [graph.get_attr(name) for name in self._graphpatch_attribute_names]

        # This is a bit silly, but Graph doesn't have any built-in way to refer to "self" when
        # calling methods!
        self_node = graph.get_attr("_graphpatch_self")
        self_node.meta["_graphpatch_hidden"] = True

        call_forward = graph.call_method(
            "_opaque_module_call",
            (self_node,) + tuple(attribute_nodes) + tuple(module_args.values()),
            module_kwargs,
        )
        call_forward.meta["_graphpatch_hidden"] = True

        # Insert submodules as siblings of the opaque module call (which is hidden); this gives
        # simpler canonical names for them. Note that we do not actually call these modules here
        # when executing the graph; they are implicitly called at the call_forward node by the
        # original module.
        for name, _ in self._children_with_container_passthrough(module.named_children()):
            graph.call_function(SubmoduleWrapper(name))

        graph.output((call_forward,))
        return graph

    @staticmethod
    def _children_with_container_passthrough(
        children: Iterator[Tuple[str, Module]]
    ) -> Iterator[Tuple[str, Module]]:
        child_modules = list(children)
        while child_modules:
            name, submodule = child_modules.pop()
            if isinstance(submodule, (ModuleList, Sequential, ModuleDict)):
                child_modules.extend(
                    reversed([(f"{name}.{n}", m) for n, m in submodule.named_children()])
                )
            else:
                yield name, submodule

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        # We want OpaqueGraphModule to behave as if its containers don't exist, instead directly
        # owning the contained submodules. This lets us keep the hierarchy that the original module
        # code was expecting, while looking to the rest of our own code as if we had unrolled the
        # containers as we do with CompiledGraphModule.
        return self._children_with_container_passthrough(super().named_children())

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Optional[Graph] = None,
        class_name: str = "OpaqueGraphModule",
        num_invocations: int = 1,
    ):
        # Deserializing from pickle.
        if isinstance(root, dict):
            # Need to set this, since GraphModule constructor will see that we get_attr on it and
            # attempt to copy it in.
            root["_graphpatch_self"] = self
            super().__init__(root, graph, class_name)
            self.set_extra_state(root[_EXTRA_STATE_KEY_SUFFIX])
            # Register submodules.
            for node in self.graph.nodes:
                if node.op == "call_function" and isinstance(node.target, SubmoduleWrapper):
                    setattr(self, node.target.module_name, root[node.target.module_name])
            return

        super().__init__(Module(), Graph(), class_name)
        self._graphpatch_self = self

        # Cloning an existing OpaqueGraphModule.
        if isinstance(root, OpaqueGraphModule):
            self.set_extra_state(root.get_extra_state())
            self.graph = deepcopy(root.graph)
        # Constructing from scratch.
        else:
            self._graphpatch_opaque_module_class = root.__class__
            self.graph = self._construct_graph(root)
            self._graphpatch_opaque_module_methods = {}
            self._graphpatch_num_invocations = num_invocations
            for name, method in _module_methods(root):
                self._graphpatch_opaque_module_methods[name] = MethodWrapper(method)

        # We need to copy attributes even if cloning an existing OpaqueGraphModule.
        for name in self._graphpatch_attribute_names:
            setattr(self, name, getattr(root, name))
