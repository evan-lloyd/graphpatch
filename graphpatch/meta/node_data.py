from collections import deque
from copy import deepcopy
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from ..optional.dataclasses import dataclass
from ..optional.typing_extensions import TypeAlias, TypeGuard

NodeDataType = TypeVar("NodeDataType")
OtherNodeDataType = TypeVar("OtherNodeDataType")
MaybeNodeData: TypeAlias = Union["NodeData[NodeDataType]", "Literal[NodeData.Sentinels._NO_VALUE]"]
MaybeNodeDataType: TypeAlias = Union[NodeDataType, "Literal[NodeData.Sentinels._NO_VALUE]"]
MaybeOtherNodeDataType: TypeAlias = Union[
    OtherNodeDataType, "Literal[NodeData.Sentinels._NO_VALUE]"
]
MaybeHandledData: TypeAlias = Union[
    "NodeData[NodeDataType]", "Literal[NodeData.Sentinels._UNHANDLED_VALUE]"
]


class _LeafNode(Protocol[NodeDataType]):
    _children: "Literal[NodeData.Sentinels._NO_VALUE]"
    _value: NodeDataType
    _original_type: type
    _path: str


class _InternalNode(Protocol[NodeDataType]):
    _value: MaybeNodeDataType[NodeDataType]
    _children: Dict[str, "NodeData[NodeDataType]"]
    _original_type: type
    _path: str


def _is_leaf(node: "NodeData[NodeDataType]") -> TypeGuard[_LeafNode[NodeDataType]]:
    return (
        node._children is NodeData.Sentinels._NO_VALUE
        and node._value is not NodeData.Sentinels._NO_VALUE
    )


def _is_internal(node: "NodeData[NodeDataType]") -> TypeGuard[_InternalNode[NodeDataType]]:
    return node._children is not NodeData.Sentinels._NO_VALUE


@dataclass(kw_only=True)
class NodeData(Generic[NodeDataType]):
    """Storage class for (possibly nested) arbitrary data associated with the output of a pytorch FX
    node. Implements the interface for abc.Mapping. Users should never need to access the underlying
    tree structure directly.

    Todo:
        * Replace recursion with iteration, in case of deeply nested structures.
    """

    class Sentinels(Enum):
        _NO_VALUE = "_NO_VALUE"
        _UNHANDLED_VALUE = "_UNHANDLED_VALUE"

    _NO_VALUE: ClassVar[Literal[Sentinels._NO_VALUE]] = Sentinels._NO_VALUE
    _UNHANDLED_VALUE: ClassVar[Literal[Sentinels._UNHANDLED_VALUE]] = Sentinels._UNHANDLED_VALUE

    _children: Union[
        Dict[str, "NodeData[NodeDataType]"], Literal[Sentinels._NO_VALUE]
    ] = Sentinels._NO_VALUE
    _value: Union[NodeDataType, Literal[Sentinels._NO_VALUE]] = Sentinels._NO_VALUE
    _original_type: type
    _path: str

    @property
    def is_leaf(self) -> bool:
        return _is_leaf(self)

    def _as_leaf(self) -> _LeafNode[NodeDataType]:
        assert _is_leaf(self)
        return self

    def _as_internal(self) -> _InternalNode[NodeDataType]:
        assert _is_internal(self)
        return self

    @property
    def is_internal(self) -> bool:
        return self._children is not NodeData.Sentinels._NO_VALUE

    def handle_unwrap(self) -> Any:
        """Override in subclasses to implement custom unwrapping behavior. Return
        NodeData._UNHANDLED_VALUE to run the base class' default behavior.
        """
        return NodeData._UNHANDLED_VALUE

    def keys(self) -> Iterator[str]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")

        def _generator() -> Iterator[str]:
            for path, node in self._iter_items():
                if node._value is not NodeData._NO_VALUE:
                    yield path

        return _generator()

    def items(self) -> Iterator[Tuple[str, NodeDataType]]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")

        def _generator() -> Iterator[Tuple[str, NodeDataType]]:
            for path, node in self._iter_items():
                if node._value is not NodeData._NO_VALUE:
                    yield path, node._value

        return _generator()

    def values(self) -> Iterator[NodeDataType]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")

        def _generator() -> Iterator[NodeDataType]:
            for _, node in self._iter_items():
                if node._value is not NodeData._NO_VALUE:
                    yield node._value

        return _generator()

    def reverse_topological_order(self) -> Iterator[NodeDataType]:
        """Yield children before their parents, but otherwise in forward order."""
        queue = deque([self])
        result_stack: List[NodeData[NodeDataType]] = []
        while queue:
            cur = queue.popleft()
            result_stack.append(cur)

            if cur._children is NodeData._NO_VALUE:
                continue

            queue.extend(reversed(cur._children.values()))
        while result_stack:
            node = result_stack.pop()
            if node._value is not NodeData._NO_VALUE:
                yield node._value

    def get(
        self, path: str, default: Optional[NodeDataType] = None
    ) -> Union[Optional[NodeDataType], "NodeData[NodeDataType]"]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")
        try:
            dig_result = self._dig(path)
            if dig_result._value is not NodeData._NO_VALUE:
                return dig_result._value
            return dig_result

        except KeyError:
            return default

    def __len__(self) -> int:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")
        return len(list(self.keys()))

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __contains__(self, path: str) -> bool:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempting to use mapping interface on leaf node.")
        return path in self.keys()

    def __reversed__(self) -> Iterator[str]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempt to use mapping interface on leaf node")

        def _generator() -> Iterator[str]:
            keys = list(self.keys())
            for path in reversed(keys):
                yield path

        return _generator()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeData):
            return False

        return self._value == other._value and self._children == other._children

    def __getitem__(self, path: str) -> Union[NodeDataType, "NodeData[NodeDataType]"]:
        if self._children is NodeData._NO_VALUE:
            raise ValueError("Attempt to use mapping interface on leaf node")
        dig_result = self._dig(path)
        if dig_result._value is not NodeData._NO_VALUE:
            return dig_result._value
        return dig_result

    def _iter_items(self) -> Iterator[Tuple[str, "NodeData[NodeDataType]"]]:
        queue = deque([("", self)])
        while queue:
            prefix, cur = queue.popleft()
            yield prefix, cur
            if cur._children is NodeData._NO_VALUE:
                continue

            queue.extend(
                (f"{(prefix + '.') if prefix else ''}{key}", value)
                for key, value in cur._children.items()
            )

    def _depth_first_items(self) -> Iterator[Tuple[str, "NodeData[NodeDataType]"]]:
        stack = [("", self)]
        while stack:
            prefix, cur = stack.pop()
            yield prefix, cur
            if cur._children is NodeData._NO_VALUE:
                continue

            stack.extend(
                (f"{(prefix + '.') if prefix else ''}{key}", value)
                for key, value in reversed(cur._children.items())
            )

    def _dig(self, path: Optional[str]) -> "NodeData[NodeDataType]":
        cur = self
        if path:
            for key in path.split("."):
                if cur._children is NodeData._NO_VALUE:
                    raise KeyError(f"Missing node in path {path}")
                cur = cur._children[key]
        return cur

    def map_in_place(self, fn: Callable[[NodeDataType], NodeDataType]) -> None:
        """Replaces values at all nodes with the result of fn(previous_value)."""
        for _, node in self._iter_items():
            if node._value is not NodeData._NO_VALUE:
                node._value = fn(node._value)

    def filter(
        self,
        predicate: Callable[[str, MaybeNodeDataType[NodeDataType]], bool],
        node_constructor: Optional[Callable[..., "NodeData[OtherNodeDataType]"]] = None,
        root_prefix: str = "",
    ) -> "MaybeNodeData[OtherNodeDataType]":
        """Returns a new NodeData tree with the same structure as this one, but with all values
        failing predicate removed."""

        return self.map(
            lambda path, value: (
                value
                if (value is not NodeData._NO_VALUE and predicate(path, value))
                else NodeData._NO_VALUE
            ),
            node_constructor,
            root_prefix,
        )

    def map(
        self,
        fn: Callable[[str, NodeDataType], MaybeNodeDataType[NodeDataType]],
        node_constructor: Optional[Callable[..., "NodeData[OtherNodeDataType]"]] = None,
        root_prefix: str = "",
    ) -> "MaybeNodeData[OtherNodeDataType]":
        """Returns a new NodeData tree with the same structure as this one, but with all values
        replaced with fn(previous_value), which may return a different type of data. If fn returns
        NodeData._NO_VALUE, omit it from the result unless it has children with values."""

        def default_node_constructor(**kwargs: Any) -> NodeData[OtherNodeDataType]:
            return NodeData[OtherNodeDataType](**kwargs)

        if node_constructor is None:
            node_constructor = default_node_constructor
        new_nodes: Dict[str, MaybeNodeData[OtherNodeDataType]] = {}
        queue = deque([(root_prefix, self)])
        node_stack = []
        while queue:
            prefix, cur = queue.popleft()
            node_stack.append((prefix, cur))
            if cur._children is NodeData._NO_VALUE:
                continue

            queue.extend(
                (f"{(prefix + '.') if prefix else ''}{key}", value)
                for key, value in cur._children.items()
            )
        # Process bottom-up, so children will be constructed by the time we reach their parents.
        for path, node in reversed(node_stack):
            if node._value is not NodeData._NO_VALUE:
                value = fn(path, node._value)
            else:
                value = NodeData._NO_VALUE
            children: Union[
                Dict[str, MaybeNodeData[OtherNodeDataType]], Literal[NodeData.Sentinels._NO_VALUE]
            ] = NodeData._NO_VALUE
            if node._children is not NodeData._NO_VALUE:
                children = {
                    k: new_nodes[child_path]
                    for k in node._children
                    if (child_path := f"{path + '.' if path else ''}{k}") in new_nodes
                }
                if len(children) == 0:
                    children = NodeData._NO_VALUE
            if value is not NodeData._NO_VALUE or children is not NodeData._NO_VALUE:
                new_nodes[path] = node_constructor(
                    _value=value,
                    _children=children,
                    _original_type=node._original_type,
                    _path=path,
                )
        root = new_nodes.get(root_prefix, NodeData._NO_VALUE)
        return root

    def replace(self, path: Optional[str], fn: Callable[[NodeDataType], NodeDataType]) -> None:
        node = self._dig(path)
        if node._value is not NodeData._NO_VALUE:
            node._value = fn(node._value)
        else:
            raise ValueError("Cannot replace() on an internal node")

    def unwrap(
        self, handle_unwrap: Optional[Callable[["NodeData[NodeDataType]"], Any]] = None
    ) -> Any:
        if handle_unwrap is not None:
            function_handled_unwrap = handle_unwrap(self)
            if function_handled_unwrap is not NodeData._UNHANDLED_VALUE:
                return function_handled_unwrap

        subclass_handled_value = self.handle_unwrap()

        if subclass_handled_value is not NodeData._UNHANDLED_VALUE:
            return subclass_handled_value

        # NB: with default container types, we assume internal nodes have no _value
        if self._children is NodeData._NO_VALUE:
            return self._value
        elif issubclass(self._original_type, (tuple, list)):
            return self._original_type(t.unwrap(handle_unwrap) for t in self._children.values())
        elif issubclass(self._original_type, (dict,)):
            return self._original_type(
                **{k: v.unwrap(handle_unwrap) for k, v in self._children.items()}
            )

        raise ValueError(
            f"Unhandled container type {self._original_type}."
            " To fix, you can implement a subclass of NodeData with a custom handle_unwrap method,"
            " or pass a custom handle_unwrap to this function."
        )

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "NodeData[NodeDataType]":
        """Override deepcopy implementation to copy _value first, so that we'll copy Graphs before
        Nodes.
        """
        # Make it truthy.
        if not memo:
            memo = {None: None}
        return NodeData[NodeDataType](
            _value=deepcopy(self._value, memo),
            _children=deepcopy(self._children, memo),
            _original_type=self._original_type,
            _path=self._path,
        )


@dataclass(kw_only=True)
class PrettyPrintedNodeData(NodeData[NodeDataType]):
    """Pretty print this node data as a tree. Support getattr access and __dir__ for autocompletion
    in REPL/notebooks.

    Attributes:
        show_containers: Display information about original container types and sizes?
        unwrap_leaves: When accessing a leaf via repeated getattr, if present, return
            unwrap_leaves(leaf._value). For example, in a NodeShape tree, this lets us return the
            wrapped torch.Size instead of a node when the user has reached a leaf.
    """

    show_containers: bool = False
    unwrap_leaves: Optional[Callable[[NodeDataType], Any]] = None

    def __dir__(self) -> List[str]:
        if self._children is not NodeData._NO_VALUE:
            return list(self._children.keys())
        return []

    def __getattr__(self, key: str) -> Any:
        if self._children is not NodeData._NO_VALUE and key in self._children:
            if self._children[key].is_leaf and self.unwrap_leaves is not None:
                return self.unwrap_leaves(self._children[key]._as_leaf()._value)
            return self._children[key]
        raise AttributeError(f"No node {key} at {self._path or '<root>'}")

    def __repr__(self) -> str:
        lines = []
        all_items = list(self._depth_first_items())
        cur_bars = {""}
        is_root = True
        for name, node in all_items:
            if is_root:
                name = self._path or "<root>"
                indent = 0
                indent_str = ""
            else:
                indent = name.count(".") + 1
                # Look ahead to see if we're the last sibling at this indent level
                parent_name = ".".join(name.split(".")[:-1])
                parent = self._dig(parent_name)._as_internal()
                if node is next(reversed(parent._children.values())):
                    cur_bars.remove(parent_name)
                    joiner = "└─"
                else:
                    joiner = "├─"
                if node._children is not NodeData._NO_VALUE:
                    cur_bars.add(name)

                indent_str = ""
                for i in range(indent - 1):
                    if any(((b.count(".") + 1) if b != "" else 0) == i for b in cur_bars):
                        indent_str += "│ "
                    else:
                        indent_str += "  "
                indent_str += joiner

            if hasattr(node._value, "_graphpatch_graph_repr"):
                container_info = f": {node._value._graphpatch_graph_repr()}"  # type: ignore
            elif self.show_containers and node._children is not NodeData._NO_VALUE:
                container_info = f": {node._original_type.__name__}({len(node._children)})"
            else:
                container_info = ""

            lines.append(
                indent_str
                + (name.split(".")[-1] if not is_root else name)
                + container_info
                + (
                    f": {node._value}"
                    if node._value is not NodeData._NO_VALUE and str(node._value) != ""
                    else ""
                )
            )
            is_root = False
        return "\n".join(lines)


class NodeDataWrapper(Generic[NodeDataType]):
    """Base class to handle customizable wrapping of arbitrary data into a NodeData structure.
    Derived classes can override handle_wrap to implement custom behavior for specific data types.

    Attributes:
      _node_data_type: Subclass of NodeData to instantiate at each node.

    Methods:
      handle_wrap: Called before invoking default wrapping behavior. Overriding methods must return
      return either a NodeData instance, or NodeData._UNHANDLED_VALUE to fall-through to the
      default behavior.
      make_wrapper: Helper to instantiate an instance of self._node_data_type.
      handle_leaf: Default handler for data not matching any recognized container type. If
      overriding, this must always return a NodeData instance.
      wrap: Public API entrypoint; should not be overridden.
    """

    def __init__(self, node_data_type: Type[NodeData[NodeDataType]] = NodeData[NodeDataType]):
        self._node_data_type = node_data_type

    def handle_wrap(self, data: Any, path: str) -> MaybeHandledData:  # type: ignore[type-arg]
        """Override in subclasses to implement custom behavior depending on data. Return
        NodeData._UNHANDLED_VALUE to run the base class' default behavior.
        """
        return NodeData._UNHANDLED_VALUE

    def make_wrapper(self, **kwargs: Any) -> NodeData[NodeDataType]:
        return self._node_data_type(**kwargs)

    def handle_leaf(self, data: Any, path: str) -> NodeData[NodeDataType]:
        return self.make_wrapper(_original_type=type(data), _value=data, _path=path)

    def wrap(
        self,
        data: Any,
        path: str = "",
    ) -> "NodeData[NodeDataType]":
        subclass_handled_value = self.handle_wrap(data, path)
        if subclass_handled_value is not NodeData._UNHANDLED_VALUE:
            return subclass_handled_value

        if path == "":
            prefix = ""
        else:
            prefix = f"{path}."

        if isinstance(data, (tuple, list)):
            return self.make_wrapper(
                _original_type=type(data),
                _children={
                    f"sub_{i}": self.wrap(c, f"{prefix}sub_{i}") for i, c in enumerate(data)
                },
                _path=path,
            )
        elif isinstance(data, dict):
            return self.make_wrapper(
                _original_type=type(data),
                _children={k: self.wrap(data[k], f"{prefix}{k}") for k in data},
                _path=path,
            )
        else:
            return self.handle_leaf(data, path)


def wrap_node_data(data: NodeDataType) -> NodeData[NodeDataType]:
    return NodeDataWrapper[NodeDataType]().wrap(data)


def make_pretty_printed(
    node_data: NodeData[NodeDataType],
    show_containers: bool = False,
    unwrap_leaves: Optional[Callable[[NodeDataType], Any]] = None,
    root_prefix: str = "",
) -> PrettyPrintedNodeData[NodeDataType]:
    # Need the cast here, since there doesn't currently seem to be a way to tell mypy that our
    # constructor returns a more specific type. This would be easy with higher-kinded types, but
    # currently requires Python 3.12: https://github.com/python/mypy/issues/15238
    return cast(
        PrettyPrintedNodeData[NodeDataType],
        node_data.map(
            lambda _, value: value,
            lambda **kwargs: PrettyPrintedNodeData(
                show_containers=show_containers, unwrap_leaves=unwrap_leaves, **kwargs
            ),
            root_prefix=root_prefix,
        ),
    )
