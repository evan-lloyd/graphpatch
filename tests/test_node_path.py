from typing import Optional

import pytest
from syrupy.extensions.json import JSONSnapshotExtension
from syrupy.extensions.amber.serializer import AmberDataSerializer

from graphpatch.hacks import TORCH_VERSION
from graphpatch.meta import NodeData, NodeMeta, wrap_node_path
from graphpatch.meta.graph_meta import WrappedCode
from graphpatch.meta.node_path import NodeShapePath
from tests.fixtures.deeply_nested_output_module import DeeplyNestedOutputModule


class SingleFileAmber(JSONSnapshotExtension):
    _file_extension = "ambr"

    def serialize(self, data, **kwargs):
        return AmberDataSerializer.serialize(data, **kwargs)

    @classmethod
    def dirname(cls, *, test_location):
        original = JSONSnapshotExtension.dirname(test_location=test_location)
        return f"{original}/{test_location.testname}"

    @classmethod
    def get_snapshot_name(cls, *, test_location, index):
        return str(index)


class MockNodeMeta(NodeMeta):
    shape: Optional[NodeData[int]]
    hidden: bool = False

    def __init__(self, shape):
        self.shape = shape
        self.hidden = False


@pytest.fixture
def path_data():
    return NodeData(
        _path="",
        _children={
            "foo": NodeData(
                _path="foo",
                _children={
                    "bar": NodeData(
                        _path="foo.bar",
                        _children={
                            "sub_0": NodeData(
                                _path="foo.bar.sub_0",
                                _value=MockNodeMeta(
                                    shape=NodeData(
                                        _path="",
                                        _children={
                                            "sub_0": NodeData(
                                                _path="sub_0",
                                                _children={
                                                    "baz": NodeData(
                                                        _path="sub_0.baz",
                                                        _value=0,
                                                        _original_type=int,
                                                    )
                                                },
                                                _original_type=dict,
                                            ),
                                            "sub_1": NodeData(
                                                _path="sub_1",
                                                _value=0,
                                                _original_type=int,
                                            ),
                                        },
                                        _original_type=tuple,
                                    )
                                ),
                                _original_type=dict,
                            ),
                            "sub_1": NodeData(
                                _path="foo.bar.sub_1",
                                _value=MockNodeMeta(shape=None),
                                _original_type=tuple,
                            ),
                        },
                        _original_type=tuple,
                    ),
                },
                _original_type=dict,
            ),
        },
        _original_type=dict,
    )


def test_wrap_node_path(path_data):
    node_path = wrap_node_path(path_data)

    # Node hierarchy should have been preserved
    for key in [
        "",
        "foo",
        "foo.bar",
        "foo.bar.sub_0",
        "foo.bar.sub_1",
    ]:
        assert node_path._dig(key)._path == key

    # Append shape paths following "|"
    for key in ["sub_0", "sub_0.baz", "sub_1"]:
        assert node_path._dig(f"foo.bar.sub_0.{key}")._path == f"foo.bar.sub_0|{key}"

    # We should be able to access each path by getattr
    assert node_path.foo is node_path._dig("foo")
    assert node_path.foo.bar is node_path._dig("foo.bar")
    assert node_path.foo.bar.sub_0 is node_path._dig("foo.bar.sub_0")
    assert node_path.foo.bar.sub_1 is node_path._dig("foo.bar.sub_1")
    assert node_path.foo.bar.sub_0.sub_0 is node_path._dig("foo.bar.sub_0.sub_0")
    assert node_path.foo.bar.sub_0.sub_0.baz is node_path._dig("foo.bar.sub_0.sub_0.baz")
    assert node_path.foo.bar.sub_0.sub_1 is node_path._dig("foo.bar.sub_0.sub_1")


def test_node_path_autocomplete(path_data):
    node_path = wrap_node_path(path_data)

    assert set(node_path.__dir__()) == {"foo", "_code", "_shape"}
    assert set(node_path.foo.__dir__()) == {"bar", "_code", "_shape"}
    assert set(node_path.foo.bar.__dir__()) == {"sub_0", "sub_1", "_code", "_shape"}
    assert set(node_path.foo.bar.sub_0.__dir__()) == {"sub_0", "sub_1", "_code", "_shape"}
    assert set(node_path.foo.bar.sub_0.sub_0.__dir__()) == {"baz", "_code", "_shape"}
    assert set(node_path.foo.bar.sub_0.sub_0.baz.__dir__()) == {"_code", "_shape"}
    assert set(node_path.foo.bar.sub_0.sub_1.__dir__()) == {"_code", "_shape"}


@pytest.mark.parametrize("all_patchable_graphs", ["compiled", "opaque"], indirect=True)
def test_patchable_graph_graph_repr(all_patchable_graphs, snapshot):
    snapshot = snapshot.with_defaults(extension_class=SingleFileAmber)

    # Note we get slightly different structures for newer versions of torch, due to it retaining
    # more not-actually-used nodes. In future releases we should use our own logic to clean up
    # graphs after extraction, which should eliminate this discrepancy.
    if TORCH_VERSION >= (2, 4):
        torch_version_suffix = "2_4"
    elif TORCH_VERSION >= (2, 2):
        torch_version_suffix = "2_2-2_3"
    elif TORCH_VERSION >= (2, 1):
        torch_version_suffix = "2_1"
    else:
        torch_version_suffix = "2_0"

    for pg_name, pg in all_patchable_graphs.items():
        assert repr(pg.graph) == snapshot(
            name=f"{pg_name}_{torch_version_suffix}"
        ), f"Snapshot mismatch for {pg_name}"


def test_node_path_shape(patchable_deeply_nested_output_module):
    pg = patchable_deeply_nested_output_module
    # We should be able to throw a "._shape" in anywhere
    assert pg.graph._shape.add == DeeplyNestedOutputModule._shape
    assert pg.graph.add._shape == DeeplyNestedOutputModule._shape
    assert pg.graph.output.sub_0.sub_0._shape == DeeplyNestedOutputModule._shape
    assert pg.graph.output.sub_0._shape.sub_0 == DeeplyNestedOutputModule._shape
    assert pg.graph.output._shape.sub_0.sub_0 == DeeplyNestedOutputModule._shape
    assert pg.graph._shape.output.sub_0.sub_0 == DeeplyNestedOutputModule._shape


def test_node_path_code(patchable_deeply_nested_output_module, snapshot):
    pg = patchable_deeply_nested_output_module
    if TORCH_VERSION >= (2, 1):
        assert pg.graph._code == snapshot(name="2_1")
    else:
        assert pg.graph._code == snapshot(name="2_0")
    assert str(pg.graph.output._code) == (
        "return ((linear_0,), [([getitem_5], getitem_6), ([getitem_9], getitem_10), ([getitem_13], getitem_14)],"
        " {'nested_dict': [(linear_1,)]})"
    )


def test_protected_names(patchable_protected_name_module, protected_name_module_inputs):
    pg = patchable_protected_name_module
    pg(protected_name_module_inputs)

    # Special REPL properties should not have been shadowed by nodes.
    assert isinstance(pg.graph._code, WrappedCode)
    assert isinstance(pg.graph._shape, NodeShapePath)
    assert "_code" not in pg.graph
    assert "_shape" not in pg.graph

    # We should *not* have changed the targets of any placeholders, since that would break
    # calling code if using kwargs.
    base_nodes = list(pg._graph_module.graph.nodes)
    assert [n.target for n in base_nodes if n.op == "placeholder"] == [
        "_shape",
        "_code",
        "sub_shape",
        "_graphpatch_args",
    ]

    # The renamed node should still be in the graph.
    assert any(n.node.name == "_code" for n in pg._meta.values() if n.node is not None)
