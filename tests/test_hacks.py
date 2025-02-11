import pytest
from torch import ones
from torch._dynamo.exc import Unsupported
from torch.nn import Module

from graphpatch.extraction.compiled_graph_module import compile_module
from graphpatch.hacks import avoid_inlining, is_allowed_in_graph


def test_avoid_inlining():

    def forbidden():
        print("print not allowed!")

    def hacked():
        forbidden()

    class ForbiddenModule(Module):
        def forward(self):
            forbidden()
            return ones((1,))

    class HackedModule(Module):
        def forward(self):
            hacked()
            return ones((1,))

    with pytest.raises(Unsupported):
        compile_module(ForbiddenModule())

    assert not is_allowed_in_graph(hacked)
    with avoid_inlining(hacked):
        assert is_allowed_in_graph(hacked)
        compiled = compile_module(HackedModule())
    # Should not persist outside context
    assert not is_allowed_in_graph(hacked)

    assert compiled()[0].equal(ones((1,)))
    nodes = {n.name: n for n in compiled.graph.nodes}
    assert "hacked" in nodes
    assert nodes["hacked"].op == "call_function"
    assert nodes["hacked"].target is hacked
