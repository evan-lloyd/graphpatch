from typing import Any, Tuple

from torch import compile
from torch.fx import Graph, GraphModule
from torch.nn import Module

from .graphpatch_module import GraphPatchModule


class CompiledGraphModule(GraphPatchModule):
    pass


def compile_module(module: Module, *args, **kwargs) -> Tuple[CompiledGraphModule, Any]:
    graph_module = GraphModule({}, Graph())

    def callback(gm: GraphModule, *args, **kwargs) -> GraphModule:
        nonlocal graph_module
        graph_module = gm
        # There is no hook to choose a subclass of GraphModule to create during compilation, so
        # dynamically make it a subclass of CompiledGraphModule. GraphModules are always created
        # by torch as the sole instance of a dynamically generated class, so this is safe.
        gm.__class__.__bases__ = (CompiledGraphModule,) + gm.__class__.__bases__
        gm.__class__.__name__ = CompiledGraphModule.__name__
        return gm

    # We need to actually run inference to generate a GraphModule, which gets passed to
    # our callback above.
    output = compile(backend=callback, dynamic=True, fullgraph=True)(module)(*args, **kwargs)

    if not isinstance(graph_module, CompiledGraphModule):
        raise ValueError("Compilation callback was never called.")
    return graph_module, output
