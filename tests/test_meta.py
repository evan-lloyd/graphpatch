import pytest
import torch

from graphpatch.meta import NodeData, NodeShape, wrap_node_data, wrap_node_shape
from graphpatch.meta.node_data import NodeDataWrapper


@pytest.fixture
def nested_shape():
    ones_1 = torch.ones((1, 2, 3))
    ones_2 = torch.ones((4, 5, 6))
    return wrap_node_shape(
        {
            "foo": (ones_1, ones_1, {"bar": (ones_2,)}),
            "baz": ones_2,
        }
    )


def test_node_shape_base_case():
    assert wrap_node_shape(torch.ones((1, 2, 3)))._value.shape == torch.Size((1, 2, 3))


def test_node_shape_tuple():
    output_shape = wrap_node_shape((torch.ones((1, 2, 3)), torch.ones((4, 5, 6))))
    assert set(output_shape.keys()) == {"sub_0", "sub_1"}
    assert output_shape["sub_0"].shape == torch.Size((1, 2, 3))
    assert output_shape["sub_1"].shape == torch.Size((4, 5, 6))


def test_node_shape_mapping_interface(nested_shape):
    shape_1 = NodeShape(_shape=torch.Size((1, 2, 3)), _data_type="Tensor")
    shape_2 = NodeShape(_shape=torch.Size((4, 5, 6)), _data_type="Tensor")

    # keys
    assert set(nested_shape.keys()) == {"foo.sub_0", "foo.sub_1", "foo.sub_2.bar.sub_0", "baz"}
    assert set(nested_shape["foo"].keys()) == {"sub_0", "sub_1", "sub_2.bar.sub_0"}
    with pytest.raises(Exception):
        nested_shape["baz"].keys()

    # items
    assert list(nested_shape.items()) == [
        ("baz", shape_2),
        ("foo.sub_0", shape_1),
        ("foo.sub_1", shape_1),
        ("foo.sub_2.bar.sub_0", shape_2),
    ]
    assert list(nested_shape["foo"].items()) == [
        ("sub_0", shape_1),
        ("sub_1", shape_1),
        ("sub_2.bar.sub_0", shape_2),
    ]
    with pytest.raises(Exception):
        nested_shape["baz"].items()

    # values
    assert list(nested_shape.values()).count(shape_1) == 2
    assert list(nested_shape.values()).count(shape_2) == 2
    assert list(nested_shape["foo"].values()) == [shape_1, shape_1, shape_2]
    with pytest.raises(Exception):
        nested_shape["baz"].values()

    # get
    assert nested_shape.get("foo.sub_2") == nested_shape["foo"]["sub_2"]
    assert nested_shape.get("foo.nothere") is None
    assert nested_shape.get("baz") == shape_2
    with pytest.raises(Exception):
        nested_shape["baz"].get("foo")

    # __len__
    assert len(nested_shape) == 4
    assert len(nested_shape["foo"]) == 3
    assert len(nested_shape["foo.sub_2"]) == 1

    # __iter__
    assert set(nested_shape) == {"foo.sub_0", "foo.sub_1", "foo.sub_2.bar.sub_0", "baz"}
    assert set(nested_shape["foo"]) == {"sub_0", "sub_1", "sub_2.bar.sub_0"}

    # __contains__
    assert "foo" not in nested_shape
    assert "foo.sub_0" in nested_shape
    assert "foo.sub_2.bar.sub_0" in nested_shape
    assert "bar.sub_0" in nested_shape["foo.sub_2"]
    assert "bar" not in nested_shape["foo.sub_2"]
    assert "sub_0" in nested_shape["foo.sub_2.bar"]

    # __reversed__
    assert list(reversed(nested_shape["foo"])) == ["sub_2.bar.sub_0", "sub_1", "sub_0"]

    # __eq__
    assert nested_shape["baz"] == nested_shape["foo"]["sub_2"]["bar"]["sub_0"]
    assert nested_shape["foo"] == NodeData(
        _children={
            "sub_0": NodeData(
                _value=NodeShape(_shape=torch.Size((1, 2, 3)), _data_type="Tensor"),
                _original_type=torch.Tensor,
                _path="foo.sub_0",
            ),
            "sub_1": NodeData(
                _value=NodeShape(_shape=torch.Size((1, 2, 3)), _data_type="Tensor"),
                _original_type=torch.Tensor,
                _path="foo.sub_1",
            ),
            "sub_2": NodeData(
                _children={
                    "bar": NodeData(
                        _children={
                            "sub_0": NodeData(
                                _value=NodeShape(_shape=torch.Size((4, 5, 6)), _data_type="Tensor"),
                                _original_type=torch.Tensor,
                                _path="foo.sub_2.bar.sub_0",
                            )
                        },
                        _original_type=tuple,
                        _path="foo.sub_2.bar",
                    ),
                },
                _original_type=dict,
                _path="foo.sub_2",
            ),
        },
        _original_type=tuple,
        _path="",
    )
    assert nested_shape["foo"] != nested_shape["baz"]

    # __getitem__
    assert nested_shape["foo.sub_2.bar.sub_0"] == NodeShape(
        _shape=torch.Size((4, 5, 6)), _data_type="Tensor"
    )
    assert nested_shape["foo.sub_0"] == NodeShape(_shape=torch.Size((1, 2, 3)), _data_type="Tensor")
    assert nested_shape["baz"] == NodeShape(_shape=torch.Size((4, 5, 6)), _data_type="Tensor")


@pytest.fixture
def data_to_wrap():
    return ({"foo": [1, 2, 3]}, (5, 6, (7, {"bar": (8, 9)})))


def test_wrap(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    assert wrapped._original_type == tuple
    assert wrapped["sub_0"]._original_type == dict
    assert wrapped["sub_0"]["foo"]._original_type == list
    assert wrapped["sub_1"]._original_type == tuple
    assert wrapped["sub_1"]["sub_2"]._original_type == tuple
    assert wrapped["sub_1"]["sub_2"]["sub_1"]._original_type == dict
    assert wrapped["sub_1"]["sub_2"]["sub_1"]["bar"]._original_type == tuple


def test_unwrap(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    assert wrapped.unwrap() == data_to_wrap


def test_map_in_place(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    wrapped.map_in_place(lambda x: x + 1)
    unwrapped = wrapped.unwrap()
    assert unwrapped == ({"foo": [2, 3, 4]}, (6, 7, (8, {"bar": (9, 10)})))


def test_map(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    str_mapped = wrapped.map(lambda _, x: str(x))
    unwrapped = str_mapped.unwrap()
    assert unwrapped == ({"foo": ["1", "2", "3"]}, ("5", "6", ("7", {"bar": ("8", "9")})))


def test_filter(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    filtered = wrapped.filter(lambda _, x: x > 2 and x < 8)
    unwrapped = filtered.unwrap()
    assert unwrapped == ({"foo": [3]}, (5, 6, (7,)))


def test_replace(data_to_wrap):
    wrapped = wrap_node_data(data_to_wrap)
    wrapped.replace("sub_0.foo.sub_1", lambda _: 5)
    wrapped.replace("sub_1.sub_2.sub_1.bar.sub_0", lambda _: 100)
    unwrapped = wrapped.unwrap()
    assert unwrapped[0]["foo"][1] == 5
    assert unwrapped[1][2][1]["bar"][0] == 100


def test_custom_wrapping_class():
    class Foo:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return self.x == other.x and self.y == other.y

    class FooTainer(NodeData[Foo]):
        def handle_unwrap(self):
            if self._original_type == "Foo":
                return Foo(self["x"], self["y"])
            return NodeData._UNHANDLED_VALUE

    class CustomNodeDataWrapper(NodeDataWrapper):
        def __init__(self):
            super().__init__(FooTainer)

        def handle_wrap(self, data, path):
            if isinstance(data, Foo):
                return self.make_wrapper(
                    _original_type=data.__class__.__name__,
                    _children={
                        "x": self.wrap(data.x),
                        "y": self.wrap(data.y),
                    },
                    _path=path,
                )
            return NodeData._UNHANDLED_VALUE

    data = {"foo": (Foo(1, 2), 3, [4, 5])}
    wrapped = CustomNodeDataWrapper().wrap(data)
    assert wrapped["foo.sub_0.x"] == 1
    assert wrapped["foo.sub_0.y"] == 2
    assert wrapped["foo.sub_1"] == 3

    unwrapped = wrapped.unwrap()
    assert unwrapped == data
