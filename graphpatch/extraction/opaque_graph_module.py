import inspect
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Any, Callable, Tuple

from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module, ModuleList, Sequential

from ..optional.dataclasses import dataclass


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


@dataclass(frozen=True)
class _OpaqueModuleWrapper:
    graph_module: "OpaqueGraphModule"
    module: Module
    parameter_names: Tuple[str]


class OpaqueModuleWrapper(_OpaqueModuleWrapper):
    def __init__(
        self, graph_module: "OpaqueGraphModule", module: Module, parameter_names: Tuple[str]
    ):
        # Sets up a nice naming convention within the graph. Nodes will be named to match the name
        # of the module in the module hierarchy, and the function call will appear as
        # "opaque_module_call_ModuleClassName".
        self.__name__ = module.__class__.__name__
        self.__module__ = "opaque_module_call"
        super().__init__(graph_module, module, parameter_names)

    def __call__(self, *args, **kwargs) -> Any:
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

        num_parameters = len(self.parameter_names)

        original_forward = _unbound_method(self.module.forward)
        with _patched_attributes(
            self.graph_module,
            dict((name, args[i]) for i, name in enumerate(self.parameter_names)),
        ), _patched_children(self.graph_module, pass_patch_args):
            return original_forward(self.graph_module, *args[num_parameters:], **kwargs)


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


class ParameterWrapper:
    """Dummy function to call to place parameters of opaque modules within the graph so the user
    will have a hook for patching them. Returns the parameter's value so we can pass the possibly
    patched value as an argument to our module wrapper.
    """

    def __init__(self, parameter_name: str, module: Module):
        self.parameter_name = parameter_name
        self.module = module
        self.__name__ = parameter_name
        self.__module__ = "opaque_module_parameter"

    def __call__(self, *args, **kwargs) -> None:
        return getattr(self.module, self.parameter_name)


class OpaqueGraphModule(GraphModule):
    """OpaqueGraphModule constructs a GraphModule from a :class:`torch.nn.Module` without using
    :func:`torch.compile`. This results in a graph that can only be patched at submodule inputs,
    outputs, buffers, and parameters. An OpaqueGraphModule may have CompiledGraphModules as
    submodules, which can be patched normally, but note that if this module makes multiple calls
    to these children, patches will be applied to each such invocation.
    """

    _graphpatch_original_module: OpaqueModuleWrapper

    def _construct_graph(self) -> Graph:
        graph = Graph()

        # Set up placeholder nodes from module's forward()
        module_args = {}
        module_kwargs = {}
        for name, arg in inspect.signature(
            self._graphpatch_original_module.module.forward
        ).parameters.items():
            if arg.default is inspect._empty:
                module_args[name] = graph.placeholder(name)
            else:
                module_kwargs[name] = graph.placeholder(name, default_value=arg.default)

        # TODO: we should probably just getattr. also buffers
        parameter_nodes = []
        for name, _ in self._graphpatch_original_module.module.named_parameters(recurse=False):
            parameter_nodes.append(
                graph.call_function(
                    ParameterWrapper(name, self._graphpatch_original_module.module), (), {}
                )
            )

        call_forward = graph.call_function(
            self._graphpatch_original_module,
            tuple(parameter_nodes) + tuple(module_args.values()),
            module_kwargs,
        )
        call_forward.meta["_graphpatch_hidden"] = True

        # Insert submodules as siblings of the opaque module call (which is hidden); this gives
        # simpler canonical names for them. Note that we do not actually call these modules here
        # when executing the graph; they are implicitly called at the call_forward node by the
        # original module.
        child_modules = list(self._graphpatch_original_module.module.named_children())
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

    def __init__(self, original_module: Module):
        if isinstance(original_module, OpaqueGraphModule):
            self._graphpatch_original_module = original_module._graphpatch_original_module
            graph = deepcopy(original_module.graph)
        else:
            self._graphpatch_original_module = OpaqueModuleWrapper(
                self,
                original_module,
                tuple(n for n, _ in original_module.named_parameters(recurse=False)),
            )
            graph = self._construct_graph()
        super().__init__(original_module, graph, "OpaqueGraphModule")

        # Clone attributes from the original module. We skip submodules, because those will be
        # replaced by GraphModules immediately afterwards. Make shallow copies of parameters/buffers
        # out of memory concerns (TODO: make configurable?)
        for k, v in original_module.__dict__.items():
            if k in ("_modules", "_parameters", "_buffers", "_graphpatch_original_module"):
                continue
            self.__dict__[k] = deepcopy(v)

        self._parameters = copy(original_module._parameters)
        self._buffers = copy(original_module._buffers)
