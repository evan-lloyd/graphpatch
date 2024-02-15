import inspect
from typing import Any

from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module

from ..optional.dataclasses import dataclass


@dataclass(frozen=True)
class _OpaqueModuleWrapper:
    module: Module


class OpaqueModuleWrapper(_OpaqueModuleWrapper):
    def __init__(self, module: Module):
        # Sets up a nice naming convention within the graph. Nodes will be named to match the name
        # of the module in the module hierarchy, and the function call will appear as
        # "opaque_module_call_ModuleClassName".
        self.__name__ = module.__class__.__name__
        self.__module__ = "opaque_module_call"
        super().__init__(module)

    def __call__(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)


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
    will have a hook for patching them.
    """

    def __init__(self, parameter_name: str):
        self.parameter_name = parameter_name
        self.__name__ = parameter_name
        self.__module__ = "opaque_module_parameter"

    def __call__(self, *args, **kwargs) -> None:
        return


def opaque_graph_module(original_module):
    graph = Graph()

    # Set up placeholder nodes from module's forward()
    module_args = {}
    module_kwargs = {}
    for name, parameter in inspect.signature(original_module.forward).parameters.items():
        if parameter.default is inspect._empty:
            module_args[name] = graph.placeholder(name)
        else:
            module_kwargs[name] = graph.placeholder(name, default_value=parameter.default)

    # Use a wrapper so that torch doesn't insert the original module into the module hierarchy.
    wrapped_original = OpaqueModuleWrapper(original_module)

    call_forward = graph.call_function(wrapped_original, tuple(module_args.values()), module_kwargs)
    call_forward.meta["_graphpatch_hidden"] = True

    # Insert submodules as siblings of the opaque module call (which is hidden); this gives simpler
    # canonical names for them.
    for name, _ in original_module.named_children():
        call_child = graph.call_function(child_wrapper := ChildModuleWrapper(name), (), {})
        # In some edge cases (eg, shadowing a previous operation), the name may get transformed
        child_wrapper.node_name = call_child.name

    graph.output((call_forward,))
    return GraphModule(original_module, graph)
