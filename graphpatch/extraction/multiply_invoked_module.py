from typing import Any

from torch.nn import ModuleList


class MultiplyInvokedModule(ModuleList):
    """Wrapper around a module that was invoked multiple times by its parent when graphpatch
    converted it into a GraphModule. This allows you to patch distinct invocations independently.

    Example:
        .. code::

            class Foo(Module):
                def __init__(self):
                    self.bar = Linear((3, 3))

                def forward(self, x, y):
                    return self.bar(x) + self.bar(y)

        .. ipython::
            :verbatim:

            In [1]: pg = PatchableGraph(Foo(), inputs)
            In [2]: print(pg._graph_module)
            Out [2]:
                CompiledGraphModule(
                    (bar): MultiplyInvokedModule(
                        (0-1): 2 x CompiledGraphModule()
                    )
                )
            In [3]: pg.graph
            Out [3]:

        You can patch the two calls to the submodule "bar" independently via :
        >>> with pg.patch({"bar_0": ZeroPatch(), "bar_1": AddPatch(value=1)}):
            ...

    """

    _graphpatch_invocation_index: int

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._graphpatch_invocation_index = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: surely any sane module will never vary how many times it calls its submodules
        # and the modulo doesn't matter? But we may want to make this configurable between
        # round-robin or throwing an exception, possibly a global "strict" mode?
        index = self._graphpatch_invocation_index % len(self._modules)
        self._graphpatch_invocation_index = index + 1
        return self[index](*args, **kwargs)
