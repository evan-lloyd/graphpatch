from typing import Any

from torch import compile
from torch.fx import Graph, GraphModule
from torch.nn import Module

from .. import hacks
from .graphpatch_module import GraphPatchModule


class CompiledGraphModule(GraphPatchModule):
    """CompiledGraphModule is a subclass of :class:`torch.fx.GraphModule`. It is essentially the
    output of a successful run of :func:`torch.compile` with some minor modifications made by
    ``graphpatch``.
    """

    pass


def compile_module(module: Module, *args: Any, **kwargs: Any) -> CompiledGraphModule:
    try:
        hacks._CURRENTLY_COMPILING = True
        graph_module = GraphModule({}, Graph())

        def callback(gm: GraphModule, *args: Any, **kwargs: Any) -> GraphModule:
            nonlocal graph_module
            graph_module = gm
            # There is no hook to choose a subclass of GraphModule to create during compilation, so
            # dynamically make it a subclass of CompiledGraphModule. GraphModules are always created
            # by torch as the sole instance of a dynamically generated class, so this is safe.
            assert type(gm) is not GraphModule

            # We don't want to get back a LazyGraphModule, which now happens in 2.3.
            if hacks.TORCH_VERSION >= (2, 3):
                from torch.fx._lazy_graph_module import _LazyGraphModule

                if _LazyGraphModule in type(gm).__bases__:
                    # Force an actual compilation of the GraphModule, which we need downstream.
                    gm.real_recompile()
                    type(gm).__bases__ = (CompiledGraphModule,) + tuple(
                        GraphModule if c is _LazyGraphModule else c for c in type(gm).__bases__
                    )
            else:
                type(gm).__bases__ = (CompiledGraphModule,) + type(gm).__bases__
            type(gm).__name__ = CompiledGraphModule.__name__
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
