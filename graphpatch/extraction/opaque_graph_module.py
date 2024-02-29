import inspect
from contextlib import contextmanager
from copy import copy, deepcopy
from types import MethodType
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from enum import Enum
from torch import Tensor
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleList, Sequential
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX


MethodBindingType = Enum("MethodBindingType", ("_none", "_instance", "_class"))


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
            # Hit an exception before we could record this one
            if original_method is None:
                continue
            if original_method is nonexistent and hasattr(module, name):
                object.__delattr__(module, name)
            elif original_method is not nonexistent:
                object.__setattr__(module, name, original_method)


class ChildModuleWrapper:
    """Dummy function to call to place submodules of opaque modules within the graph; the actual
    calls will happen within the opaque module.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.node_name = None
        self.__name__ = module_name
        self.__module__ = "opaque_module_submodule"

    def __call__(self, *args, **kwargs) -> None:
        return

    def __str__(self) -> str:
        return self.module_name


class OpaqueGraphModule(GraphModule):
    """OpaqueGraphModule constructs a GraphModule from a :class:`torch.nn.Module` without using
    :func:`torch.compile`. This results in a graph that can only be patched at submodule inputs,
    outputs, buffers, and parameters. An OpaqueGraphModule may have CompiledGraphModules as
    submodules, which can be patched normally, but note that if this module makes multiple calls
    to these children, patches will be applied to each such invocation.
    """

    _graphpatch_opaque_module_class: Type[Module]
    _graphpatch_opaque_module_methods: Dict[str, Callable]
    _graphpatch_parameter_names: Tuple[str]
    _graphpatch_self: "OpaqueGraphModule"

    def get_extra_state(self) -> Any:
        # Not setting _graphpatch_self, since we'll create a fresh instance when we deserialize.
        return {
            "_graphpatch_parameter_names": self._graphpatch_parameter_names,
            "_graphpatch_opaque_module_class": self._graphpatch_opaque_module_class,
            "_graphpatch_opaque_module_methods": self._graphpatch_opaque_module_methods,
        }

    def set_extra_state(self, state: Any) -> None:
        self._graphpatch_parameter_names = state["_graphpatch_parameter_names"]
        self._graphpatch_opaque_module_class = state["_graphpatch_opaque_module_class"]
        self._graphpatch_opaque_module_methods = state["_graphpatch_opaque_module_methods"]

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

        num_parameters = len(self._graphpatch_parameter_names)

        # Set "self" up as a proxy for the original module. We copy over all methods from the
        # original instance (needed if, for example, the original forward calls other methods),
        # but maintain our own attributes. In particular, we substitute the possibly patched
        # parameter and buffer values in the first part of our args list.
        with _patched_attributes(
            self,
            dict((name, args[i]) for i, name in enumerate(self._graphpatch_parameter_names)),
        ), _patched_children(self, pass_patch_args), _patched_methods(
            self, self._graphpatch_opaque_module_methods
        ):
            return self(*args[num_parameters:], **kwargs)

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

        # TODO: also buffers
        self._graphpatch_parameter_names = tuple(
            name for name, _ in module.named_parameters(recurse=False)
        )
        parameter_nodes = [graph.get_attr(name) for name in self._graphpatch_parameter_names]

        # This is a bit silly, but Graph doesn't have any built-in way to refer to "self" when
        # calling methods!
        self_node = graph.get_attr("_graphpatch_self")
        call_forward = graph.call_method(
            "_opaque_module_call",
            (self_node,) + tuple(parameter_nodes) + tuple(module_args.values()),
            module_kwargs,
        )
        call_forward.meta["_graphpatch_hidden"] = True

        # Insert submodules as siblings of the opaque module call (which is hidden); this gives
        # simpler canonical names for them. Note that we do not actually call these modules here
        # when executing the graph; they are implicitly called at the call_forward node by the
        # original module.
        child_modules = list(module.named_children())
        while child_modules:
            name, submodule = child_modules.pop()
            # TODO: handle ModuleDict
            if isinstance(submodule, (ModuleList, Sequential)):
                child_modules.extend(
                    [(f"{name}.{i}", m) for i, m in enumerate(submodule.named_children())]
                )
            else:
                call_child = graph.call_function(child_wrapper := ChildModuleWrapper(name), (), {})
                # In some edge cases (eg, shadowing a previous operation), the name may get
                # transformed
                child_wrapper.node_name = call_child.name

        graph.output((call_forward,))
        return graph

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Optional[Graph] = None,
        class_name: str = "OpaqueGraphModule",
    ):
        # Deserializing from pickle.
        if isinstance(root, dict):
            # Need to set this, since GraphModule constructor will see that we get_attr on it and
            # attempt to copy it in.
            root["_graphpatch_self"] = self
            super().__init__(root, graph, class_name)
            self.set_extra_state(root[_EXTRA_STATE_KEY_SUFFIX])
            return

        super().__init__(Module(), Graph(), class_name)
        self._graphpatch_self = self

        # Cloning an existing OpaqueGraphModule.
        if isinstance(root, OpaqueGraphModule):
            self.graph = deepcopy(root.graph)
        else:
            self._graphpatch_opaque_module_class = root.__class__
            self.graph = self._construct_graph(root)

        # Clone attributes from the original module. We skip submodules, because those will be
        # replaced by GraphModules immediately afterwards. Note that the default GraphModule
        # constructor will automatically copy parameters and buffers, since we have getattr nodes
        # accessing them.
        self._graphpatch_opaque_module_methods = {}
        for k, v in inspect.getmembers(root):
            if k in (
                "_modules",
                "_parameters",
                "_buffers",
                # Might be cloning another OpaqueGraphModule
                "_graphpatch_self",
                # These get monkeypatched in an unpicklable way by dynamo, which would block
                # serialization. Shouldn't actually need them at inference time.
                "__setstate__",
                "__init__",
            ):
                continue
            if inspect.isroutine(v):
                self._graphpatch_opaque_module_methods[k] = MethodWrapper(v)
            else:
                self.__dict__[k] = deepcopy(v)

        # for name, parameter in root.named_parameters(recurse=False):
        #     self.register_parameter(name, parameter.detach())
        # self._parameters = copy(root._parameters)
        self._buffers = copy(root._buffers)
