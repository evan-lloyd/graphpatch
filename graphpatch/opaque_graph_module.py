import inspect

from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.nn import Module
from dataclasses import dataclass


@dataclass
class ModuleWrapper:
    module: Module


class OpaqueGraphModule(GraphModule):
    """OpaqueGraphModule is a wrapper around :class:`torch.nn.Module` that constructs a GraphModule
    that simply returns the result of calling the wrapped module. This lets us include uncompilable
    modules, or modules that the user wishes to skip compiling, in the same graph hierarchy as
    modules that we *do* compile.

    Parameters:
        original_module: The :class:`Module <torch.nn.module>` to wrap.
    """

    def __init__(self, original_module: Module):
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
        self._original_module = ModuleWrapper(original_module)

        # Call the forward function directly, so we don't have to mangle the module hierarchy by
        # inserting a dummy child module.
        # TODO: does this play nice with accelerate?
        call_function = graph.call_function(
            original_module.forward, tuple(module_args.values()), module_kwargs
        )
        graph.output(call_function)

        super().__init__(original_module, graph, "OpaqueGraphModule")
