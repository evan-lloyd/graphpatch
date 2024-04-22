import inspect
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from types import MethodType
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, Union

from torch.fx.graph import Graph
from torch.nn import Module

from ..optional.transformers import PreTrainedModel
from .graphpatch_module import GraphPatchModule

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
        # Needs to be skipped for torch 2.0
        "__slotnames__",
        # accelerate
        "_hf_hook",
        # Python internals not covered by the above.
        "__class__",
    }
)

# Only applicable if the Module is a PreTrainedModel.
_UNPATCHABLE_TRANSFORMERS_ATTRIBUTES = frozenset(
    {
        *(k for k in PreTrainedModel.__dict__),
        *(k for k in PreTrainedModel.__annotations__),
        # These aren't declared, but are still used.
        "_is_hf_initialized",
        "config",
        "device",
        "dtype",
        "generation_config",
        "hf_device_map",
        "is_loaded_in_8bit",
        "name_or_path",
        "warnings_issued",
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


def _is_routine(obj: Any) -> bool:
    # For torch < 2.1, some methods don't show up as routines to inspect.
    return inspect.isroutine(obj) or obj.__class__.__name__ == "method-wrapper"


def _module_methods(module: Module) -> Iterator[Callable]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return t[0] not in _UNPATCHABLE_MODULE_METHODS and _is_routine(t[1])

    return filter(_filter, inspect.getmembers(module))


def _module_attributes(module: Module) -> Iterator[Any]:
    def _filter(t: Tuple[str, Any]) -> bool:
        if isinstance(module, PreTrainedModel) and t[0] in _UNPATCHABLE_TRANSFORMERS_ATTRIBUTES:
            return False

        return t[0] not in _UNPATCHABLE_MODULE_ATTRIBUTES and not _is_routine(t[1])

    return filter(_filter, inspect.getmembers(module))


class MethodWrapper:
    """Save an unbound version of the given method so we can serialize instance methods without
    also serializing the instance itself.
    """

    function: Callable
    binding: MethodBindingType

    def __init__(self, function_or_method: Callable):
        self.function = self._unbound_method(function_or_method)
        self.binding = self._method_binding_type(function_or_method)

    @staticmethod
    def _unbound_method(function_or_method: Callable):
        return getattr(function_or_method, "__func__", function_or_method)

    @staticmethod
    def _method_binding_type(function_or_method: Callable):
        if isinstance(getattr(function_or_method, "__self__", None), type):
            return MethodBindingType._class
        elif hasattr(function_or_method, "__self__"):
            return MethodBindingType._instance
        else:
            return MethodBindingType._none

    def bound_method(self, module: Module) -> Callable:
        if self.binding is MethodBindingType._class:
            return MethodType(self._unbound_method(self.function), module.__class__)
        elif self.binding is MethodBindingType._instance:
            return MethodType(self._unbound_method(self.function), module)
        else:
            return self.function


@contextmanager
def _patched_forward(module_iterator: Iterator[Tuple[str, Module]], patch):
    try:
        original: Dict[str, Tuple[Callable, bool]] = {}
        module_list = list(module_iterator)
        for name, submodule in module_list:
            original[name] = (
                submodule.forward,
                # Was forward overridden elsewhere? accelerate likes to do this.
                "forward" in submodule.__dict__,
            )
            submodule.forward = patch(MethodWrapper(submodule.forward).bound_method(submodule))
        yield
    finally:
        for name, submodule in module_list:
            if original[name][1]:
                submodule.forward = original[name]
            else:
                del submodule.forward


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
    "_graphpatch_opaque_module_class",
    "_graphpatch_opaque_module_methods",
    # Not using _graphpatch_self or proxy, since we'll create a fresh instance when we deserialize.
)


class OpaqueGraphModule(GraphPatchModule):
    """OpaqueGraphModule constructs a GraphModule from a :class:`torch.nn.Module` without using
    :func:`torch.compile`. This results in a graph that can only be patched at submodule inputs,
    outputs, buffers, parameters, and attributes. An OpaqueGraphModule may have CompiledGraphModules
    as submodules, which can be patched normally.
    """

    _graphpatch_attribute_names: Tuple[str]
    _graphpatch_opaque_module_class: Type[Module]
    _graphpatch_opaque_module_methods: Dict[str, MethodWrapper]
    _graphpatch_opaque_module_proxy: Module
    _graphpatch_self: "OpaqueGraphModule"

    def get_extra_state(self) -> Any:
        state = super().get_extra_state()
        state.update({k: getattr(self, k) for k in _OPAQUE_GRAPH_MODULE_SERIALIZATION_KEYS})
        return deepcopy(state)

    def set_extra_state(self, state: Any) -> None:
        super().set_extra_state(state)
        for k in _OPAQUE_GRAPH_MODULE_SERIALIZATION_KEYS:
            setattr(self, k, deepcopy(state[k]))

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

        for i, name in enumerate(self._graphpatch_attribute_names):
            setattr(self._graphpatch_opaque_module_proxy, name, args[i])
        with _patched_forward(self.named_children(), pass_patch_args):
            try:
                return self._graphpatch_opaque_module_proxy(*args[num_attributes:], **kwargs)
            finally:
                # Don't hold on to references to the patched attributes.
                for i, name in enumerate(self._graphpatch_attribute_names):
                    delattr(self._graphpatch_opaque_module_proxy, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Need to bypass Module's behavior, which adds Module values to _modules.
        if name in ("_graphpatch_self", "_graphpatch_opaque_module_proxy"):
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
        # original module. NB: going to low-level _modules because named_children() unconfigurably
        # skips duplicates, which we don't want.
        for name, _ in self._child_modules(module._modules.items()):
            node = graph.call_function(SubmoduleWrapper(name))
            # Make sure all nodes are addressible by attribute access.
            if not name.split(".")[0].isidentifier():
                node.name = f"sub_{name}"

        graph.output((call_forward,))
        return graph

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        # We want OpaqueGraphModule to behave as if its containers don't exist, instead directly
        # owning the contained submodules. This lets us keep the hierarchy that the original module
        # code was expecting, while looking to the rest of our own code as if we had unrolled the
        # containers as we do with CompiledGraphModule.
        return self._child_modules(self._modules.items())

    def _initialize_proxy(self):
        # Set up proxy for the original module. We copy over all methods from the original instance
        # (needed if, for example, the original forward calls other methods), but maintain our own
        # attributes, including buffers and parameters.
        proxy = self._graphpatch_opaque_module_class.__new__(self._graphpatch_opaque_module_class)
        Module.__init__(proxy)
        # Deliberately not copying here; we want our proxy's submodules to always match ours.
        proxy._modules = self._modules

        for name, method in self._graphpatch_opaque_module_methods.items():
            object.__setattr__(proxy, name, method.bound_method(proxy))

        self._graphpatch_opaque_module_proxy = proxy

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

            self._initialize_proxy()

            # Register submodules.
            for node in self.graph.nodes:
                if node.op == "call_function" and isinstance(node.target, SubmoduleWrapper):
                    setattr(self, node.target.module_name, root[node.target.module_name])
            return

        super().__init__(Module(), Graph(), class_name)
        self._graphpatch_self = self

        # Cloning an existing OpaqueGraphModule.
        if isinstance(root, OpaqueGraphModule):
            self.set_extra_state(deepcopy(root.get_extra_state()))
            self._initialize_proxy()
            self.graph = deepcopy(root.graph)
        # Constructing from scratch.
        else:
            self._graphpatch_opaque_module_class = root.__class__
            self._graphpatch_opaque_module_methods = {
                name: MethodWrapper(method) for name, method in _module_methods(root)
            }
            self._initialize_proxy()
            self.graph = self._construct_graph(root)

        # We need to copy attributes even if cloning an existing OpaqueGraphModule.
        for name in self._graphpatch_attribute_names:
            setattr(self, name, getattr(root, name))
