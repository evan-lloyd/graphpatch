import inspect

from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleWrapper:
    module: Module

    __name__ = "opaque_module_call"

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


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
    wrapped_original = ModuleWrapper(original_module)

    # TODO: does this play nice with accelerate?
    call_forward = graph.call_function(wrapped_original, tuple(module_args.values()), module_kwargs)
    graph.output((call_forward,))
    return GraphModule(original_module, graph)
