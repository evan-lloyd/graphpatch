from typing import Any, Tuple

from torch import compile
from torch.fx import Graph, GraphModule
from torch.nn import Module

from .. import hacks
from .graphpatch_module import GraphPatchModule


class CompiledGraphModule(GraphPatchModule):
    pass


def compile_module(module: Module, *args, **kwargs) -> Tuple[CompiledGraphModule, Any]:
    try:
        hacks._CURRENTLY_COMPILING = True
        graph_module = GraphModule({}, Graph())

        def callback(gm: GraphModule, *args, **kwargs) -> GraphModule:
            nonlocal graph_module
            graph_module = gm
            # There is no hook to choose a subclass of GraphModule to create during compilation, so
            # dynamically make it a subclass of CompiledGraphModule. GraphModules are always created
            # by torch as the sole instance of a dynamically generated class, so this is safe.
            assert gm.__class__ is not GraphModule

            # We don't want to get back a LazyGraphModule, which now happens in 2.3.
            if hacks.TORCH_VERSION >= (2, 3):
                from torch.fx._lazy_graph_module import _LazyGraphModule

                if _LazyGraphModule in gm.__class__.__bases__:
                    # Force an actual compilation of the GraphModule, which we need downstream.
                    gm.real_recompile()
                    gm.__class__.__bases__ = (CompiledGraphModule,) + tuple(
                        GraphModule if c is _LazyGraphModule else c for c in gm.__class__.__bases__
                    )
            else:
                gm.__class__.__bases__ = (CompiledGraphModule,) + gm.__class__.__bases__
            gm.__class__.__name__ = CompiledGraphModule.__name__
            gm._init(module)
            hacks._CURRENTLY_COMPILING = False
            return gm

        # We need to actually run inference to generate a GraphModule, which gets passed to
        # our callback above.
        compile(backend=callback, dynamic=True, fullgraph=True)(module)(*args, **kwargs)

        if not isinstance(graph_module, CompiledGraphModule):
            raise ValueError("Compilation callback was never called.")
        return graph_module
    finally:
        hacks._CURRENTLY_COMPILING = False
