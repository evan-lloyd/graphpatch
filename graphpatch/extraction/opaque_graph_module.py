import inspect
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from torch import Tensor
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleList, Sequential
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX


def _unbound_method(function_or_method: Callable):
    return getattr(function_or_method, "__func__", function_or_method)


@contextmanager
def _patched_attributes(module, patches):
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
def _patched_children(module, patch):
    try:
        original = {}
        for name, submodule in module.named_children():
            original[name] = submodule.forward
            submodule.forward = patch(submodule.forward)
        yield
    finally:
        for name, submodule in module.named_children():
            submodule.forward = original[name]


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
    _graphpatch_parameter_names: Tuple[str]
    _graphpatch_self: "OpaqueGraphModule"

    def __reduce__(self, *args, **kwargs):
        breakpoint()

    def get_extra_state(self) -> Any:
        # Not setting _graphpatch_self, since we'll create a fresh instance when we deserialize.
        return {
            "_graphpatch_parameter_names": self._graphpatch_parameter_names,
            "_graphpatch_opaque_module_class": self._graphpatch_opaque_module_class,
        }

    def set_extra_state(self, state: Any) -> None:
        self._graphpatch_parameter_names = state["_graphpatch_parameter_names"]
        self._graphpatch_opaque_module_class = state["_graphpatch_opaque_module_class"]

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

        original_forward = _unbound_method(self._graphpatch_opaque_module_class.forward)
        with _patched_attributes(
            self,
            dict((name, args[i]) for i, name in enumerate(self._graphpatch_parameter_names)),
        ), _patched_children(self, pass_patch_args):
            # TODO: does this work if original_forward calls other methods? probably need something
            # like an attribute-only proxy
            return original_forward(self, *args[num_parameters:], **kwargs)

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
            # if isinstance(original_module, dict):
            #     super().__init__(original_module, graph, "OpaqueGraphModule")
            self._graphpatch_opaque_module_class = root.__class__
            self.graph = self._construct_graph(root)

        # Clone attributes from the original module. We skip submodules, because those will be
        # replaced by GraphModules immediately afterwards. Make shallow copies of parameters/buffers
        # out of memory concerns (TODO: make configurable?)
        for k, v in root.__dict__.items():
            if k in ("_modules", "_parameters", "_buffers", "_graphpatch_self"):
                continue
            self.__dict__[k] = deepcopy(v)

        self._parameters = copy(root._parameters)
        self._buffers = copy(root._buffers)
