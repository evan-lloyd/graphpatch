import inspect
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, Union

from torch.fx.graph import Graph
from torch.nn import Module

from ..optional.accelerate import ModelHook
from ..optional.transformers import PreTrainedModel
from .graphpatch_module import GraphPatchModule

MethodBindingType = Enum("MethodBindingType", ("_none", "_instance", "_class"))

_UNCOPYABLE_MODULE_ATTRIBUTES = frozenset(
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


def _is_routine(obj: Any) -> bool:
    # For torch < 2.1, some methods don't show up as routines to inspect.
    return inspect.isroutine(obj) or obj.__class__.__name__ == "method-wrapper"


def _is_property(cls: Type[Module], name: str) -> bool:
    return isinstance(getattr(cls, name, None), property)


def _module_attributes(module: Module) -> Iterator[Any]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return not _is_routine(t[1]) and not _is_property(type(module), t[0])

    return filter(_filter, inspect.getmembers(module))


def _patchable_module_attributes(module: Module) -> Iterator[Any]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return t[0] not in _UNCOPYABLE_MODULE_ATTRIBUTES and not (
            isinstance(module, PreTrainedModel) and t[0] in _UNPATCHABLE_TRANSFORMERS_ATTRIBUTES
        )

    return filter(_filter, _module_attributes(module))


def _static_module_attributes(module: Module) -> Iterator[Any]:
    def _filter(t: Tuple[str, Any]) -> bool:
        return t[0] not in _UNCOPYABLE_MODULE_ATTRIBUTES and (
            isinstance(module, PreTrainedModel) and t[0] in _UNPATCHABLE_TRANSFORMERS_ATTRIBUTES
        )

    return filter(_filter, _module_attributes(module))


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
            submodule.forward = patch(submodule.forward)
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
    "_graphpatch_patchable_attributes",
    "_graphpatch_static_attributes",
    "_graphpatch_opaque_module_class",
    # Not using _graphpatch_self or proxy, since we'll create a fresh instance when we deserialize.
)


class OpaqueGraphModule(GraphPatchModule):
    """OpaqueGraphModule constructs a GraphModule from a :class:`torch.nn.Module` without using
    :func:`torch.compile`. This results in a graph that can only be patched at submodule inputs,
    outputs, buffers, parameters, and attributes. An OpaqueGraphModule may have CompiledGraphModules
    as submodules, which can be patched normally.
    """

    _graphpatch_patchable_attributes: Tuple[str]
    _graphpatch_static_attributes: Dict[str, Any]
    _graphpatch_opaque_module_class: Type[Module]
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

        num_attributes = len(self._graphpatch_patchable_attributes)

        for i, name in enumerate(self._graphpatch_patchable_attributes):
            setattr(self._graphpatch_opaque_module_proxy, name, args[i])
        with _patched_forward(self.named_children(), pass_patch_args):
            try:
                return self._graphpatch_opaque_module_proxy(*args[num_attributes:], **kwargs)
            finally:
                # Don't hold on to references to the patched attributes.
                for i, name in enumerate(self._graphpatch_patchable_attributes):
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
        self._graphpatch_patchable_attributes = tuple(
            name
            for name, value in _patchable_module_attributes(module)
            if not isinstance(value, Module)
        )
        self._graphpatch_static_attributes = {
            name: value for name, value in _static_module_attributes(module)
        }
        attribute_nodes = [graph.get_attr(name) for name in self._graphpatch_patchable_attributes]

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

    def _initialize_proxy(self, root: Union[Module, Dict[str, Any]]):
        self._graphpatch_opaque_module_proxy = self._graphpatch_opaque_module_class.__new__(
            self._graphpatch_opaque_module_class
        )
        Module.__init__(self._graphpatch_opaque_module_proxy)
        # Deliberately not copying here; we want our proxy's submodules to always match ours.
        self._graphpatch_opaque_module_proxy._modules = self._modules

        # When unpickling, the GraphModule constructor will have handled our patchable attributes,
        # since we have getattr nodes referring to them. If we're constructing this Module for the
        # first time, we need to manually handle this copying. Note also that we set the attributes
        # on self, rather than the proxy, because they need to get copied at runtime to account for
        # the possibility that the user wants to patch them.
        if isinstance(root, Module):
            for name in self._graphpatch_patchable_attributes:
                setattr(self, name, getattr(root, name))
        for name, value in self._graphpatch_static_attributes.items():
            setattr(self._graphpatch_opaque_module_proxy, name, value)

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Optional[Graph] = None,
        class_name: str = "OpaqueGraphModule",
        accelerate_hook: Optional[ModelHook] = None,
    ):
        # Deserializing from pickle.
        if isinstance(root, dict):
            # Need to set this, since GraphModule constructor will see that we get_attr on it and
            # attempt to copy it in.
            root["_graphpatch_self"] = self
            super().__init__(root, graph, class_name, accelerate_hook)

            self._initialize_proxy(root)

            # Register submodules.
            for node in self.graph.nodes:
                if node.op == "call_function" and isinstance(node.target, SubmoduleWrapper):
                    setattr(self, node.target.module_name, root[node.target.module_name])
            return

        super().__init__(Module(), Graph(), class_name, accelerate_hook)
        self._graphpatch_self = self

        # Cloning an existing OpaqueGraphModule.
        if isinstance(root, OpaqueGraphModule):
            self.set_extra_state(deepcopy(root.get_extra_state()))
            self.graph = deepcopy(root.graph)
        # Constructing from scratch.
        else:
            self._graphpatch_opaque_module_class = root.__class__
            self.graph = self._construct_graph(root)

        self._initialize_proxy(root)
