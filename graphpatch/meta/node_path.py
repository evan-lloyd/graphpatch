from pprint import pformat
from typing import Any, ClassVar, List, Optional, Union, cast

from ..optional.dataclasses import dataclass
from . import GraphMeta, NodeData, NodeMeta, NodeShape, PrettyPrintedNodeData
from .node_data import MaybeNodeDataType


@dataclass(kw_only=True, repr=False)
class NodePath(PrettyPrintedNodeData[Union[GraphMeta, NodeMeta]]):
    """Helper class to set up autocomplete for finding nodes in notebooks and REPL."""

    MAX_COMPLETIONS: ClassVar[int] = 20
    _node_meta: Optional[NodeMeta] = None

    def __hash__(self) -> int:
        return self._path.__hash__()

    def __str__(self) -> str:
        return self.to_path()

    def __dir__(self) -> List[str]:
        return super().__dir__() + ["_code", "_shape"]

    def __getattr__(self, key: str) -> Any:
        if key == "_code":
            if isinstance(self._value, (GraphMeta, NodeMeta)):
                return self._value.code
            elif isinstance(self._node_meta, NodeMeta):
                return self._node_meta.code
        if key == "_shape":
            if self._children is not NodeData._NO_VALUE:
                return self.map(
                    lambda _, v: v,
                    lambda **kwargs: NodeShapePath(
                        show_containers=True,
                        unwrap_leaves=lambda value: value._shape,
                        **kwargs,
                    ),
                )
            if isinstance(self._value, NodeShape):
                return self._value._shape
            elif isinstance(self._value, NodeMeta):
                return self._value.shape
        return super().__getattr__(key)

    def to_path(self, allow_internal: bool = True) -> str:
        if not allow_internal and (
            self._value is NodeData._NO_VALUE or self._children is not NodeData._NO_VALUE
        ):
            completions = [n._path for _, n in self._iter_items() if n.is_leaf]
            if len(completions) > NodePath.MAX_COMPLETIONS:
                completions = completions[: NodePath.MAX_COMPLETIONS] + ["..."]
            raise ValueError(
                f"Incomplete node path: '{self._value}'. Possible completions:\n"
                + pformat(
                    completions,
                    indent=4,
                    compact=True,
                )
            )
        return self._path


class NodeShapePath(PrettyPrintedNodeData[NodeShape]):
    """Separate wrapper for NodeShape, which doesn't have custom handling for _code and _shape."""

    def __dir__(self) -> List[str]:
        return PrettyPrintedNodeData.__dir__(self)

    def __getattr__(self, key: str) -> Any:
        return PrettyPrintedNodeData.__getattr__(self, key)


def wrap_node_path(meta: NodeData[Union[GraphMeta, NodeMeta]]) -> NodePath:
    def make_node_path(**kwargs: Any) -> NodePath:
        node_path = kwargs["_path"]
        node = kwargs["_value"]

        def shape_value(
            path: str, value: MaybeNodeDataType[NodeShape]
        ) -> MaybeNodeDataType[NodeShape]:
            return value

        def make_shape(**kwargs: Any) -> NodePath:
            shape_path = kwargs["_path"]
            kwargs["_path"] = f"{node_path}|{shape_path}"
            return NodePath(show_containers=True, _node_meta=node, **kwargs)

        if (
            isinstance(node, NodeMeta)
            and kwargs["_children"] is NodeData._NO_VALUE
            and node.shape is not None
        ):
            kwargs["_original_type"] = node.shape._original_type
            kwargs["_value"] = node.shape._value
            kwargs["_node_meta"] = node
            kwargs["_children"] = cast(NodePath, node.shape.map(shape_value, make_shape))._children

        return NodePath(show_containers=True, **kwargs)

    def maybe_meta(
        path: str,
        value: MaybeNodeDataType[Union[GraphMeta, NodeMeta]],
    ) -> MaybeNodeDataType[Union[GraphMeta, NodeMeta]]:
        return value

    node_path = cast(
        NodePath,
        meta.map(
            maybe_meta,
            make_node_path,
        ),
    )
    return node_path
