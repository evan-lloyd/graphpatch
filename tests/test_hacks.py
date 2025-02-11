from typing import Optional

import pytest
import torch
from torch._dynamo.exc import Unsupported
from torch.nn import Module

from graphpatch import hacks
from graphpatch.extraction.compiled_graph_module import compile_module


def test_avoid_inlining():

    def forbidden():
        print("print not allowed!")

    def hacked():
        forbidden()

    class ForbiddenModule(Module):
        def forward(self):
            forbidden()
            return torch.ones((1,))

    class HackedModule(Module):
        def forward(self):
            hacked()
            return torch.ones((1,))

    with pytest.raises(Unsupported):
        compile_module(ForbiddenModule())

    # Is the patch effective?
    assert not hacks.is_allowed_in_graph(hacked)
    with hacks.avoid_inlining(hacked):
        assert hacks.is_allowed_in_graph(hacked)
        compiled = compile_module(HackedModule())
    # Should not persist outside context
    assert not hacks.is_allowed_in_graph(hacked)

    assert compiled()[0].equal(torch.ones((1,)))
    nodes = {n.name: n for n in compiled.graph.nodes}
    assert "hacked" in nodes
    assert nodes["hacked"].op == "call_function"
    assert nodes["hacked"].target is hacked


def test_opaqify():
    # @torch.library.custom_op("graphpatch::data_dependent_op", mutates_args=())
    @hacks.allow_in_graph
    # @hacks.disable
    def data_dependent_op(x: torch.Tensor) -> torch.Tensor:
        print(x)
        if torch.all(x == 1):
            return torch.ones_like(x) * 2
        return torch.zeros_like(x)

    # @data_dependent_op.register_fake
    # def fake_op(x):
    #     return torch.empty_like(x)

    class DataDependentModule(Module):
        def forward(self, x):
            return data_dependent_op(x)

    # with pytest.raises(Unsupported):
    #     compile_module(DataDependentModule(), torch.ones((3,)))

    with hacks.opaqify(data_dependent_op):
        compiled = compile_module(DataDependentModule(), torch.ones((3,)))

    print(compiled.code)
    breakpoint()
