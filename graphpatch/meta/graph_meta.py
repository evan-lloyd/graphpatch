from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

from torch import Size, Tensor
from torch.fx.graph import Graph, _Namespace
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.nn import Module

from ..optional.accelerate import ModelHook
from ..optional.dataclasses import dataclass
from .node_data import MaybeHandledData, NodeData, NodeDataWrapper

_GRAPHPATCH_RESERVED_NAMES = {"_code", "_shape"}


@dataclass(kw_only=True)
class NodeShape:
    _shape: Optional[Size]
    _data_type: str

    @property
    def shape(self) -> Union[Size, str]:
        if self._shape is not None:
            return self._shape
        return self._data_type

    def __str__(self) -> str:
        if self._shape is not None:
            return f"Tensor({', '.join(str(s) for s in self._shape)})"
        return self._data_type


class NodeShapeWrapper(NodeDataWrapper[NodeShape]):
    def handle_leaf(self, data: Any, path: str) -> NodeData[NodeShape]:
        if isinstance(data, Tensor):
            return self.make_wrapper(
                _original_type=data.__class__,
                _value=NodeShape(_shape=data.shape, _data_type="Tensor"),
                _path=path,
            )

        return self.make_wrapper(
            _original_type=data.__class__,
            _value=NodeShape(_shape=None, _data_type=data.__class__.__name__),
            _path=path,
        )


def wrap_node_shape(data: Any) -> NodeData[NodeShape]:
    return NodeShapeWrapper().wrap(data)


@dataclass
class WrappedCode:
    """Wrapped string, so users can see graph/node code printed nicely without calling print()."""

    code: str

    def __str__(self) -> str:
        return self.code

    def __repr__(self) -> str:
        return self.code


@dataclass(kw_only=True)
class _BaseMeta:
    """Meta-info to associate with each node or subgraph in a GraphModule.

    Attributes:
        name: Qualified name of this node (eg: "layers_0.mlp.add")
        local_name: Name of this node within its parent graph (eg: "add")
        is_graph: Is there a subgraph associated with this node?
        shape: Shape of the value output at this node.
        parent: Fully qualified name of the parent graph of this node (eg: "layers_0.mlp")
    """

    name: str
    local_name: str
    is_graph: ClassVar[bool]
    shape: Optional[NodeData[NodeShape]] = None
    parent: str
    code: WrappedCode


@dataclass(kw_only=True)
class NodeMeta(_BaseMeta):
    """Meta-info to associate with a non-graph node in a GraphModule."""

    node: Node
    is_graph: ClassVar[bool] = False

    def __str__(self) -> str:
        return ""

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "NodeMeta":
        if id(self) not in memo:
            memo[id(self)] = NodeMeta(
                name=self.name,
                local_name=self.local_name,
                shape=deepcopy(self.shape, memo),
                parent=self.parent,
                node=memo[self.node],
                code=self.code,
            )
        return cast(NodeMeta, memo[id(self)])


@dataclass(kw_only=True)
class GraphMeta(_BaseMeta):
    """Meta-info to associate with a subgraph node in a GraphModule.

    Attributes:
        accelerate_hook: The accelerate ModuleHook associated with this GraphModule, if any.
        nodes:
    """

    is_graph: ClassVar[bool] = True
    accelerate_hook: Optional[ModelHook]
    node: Optional[Node]
    nodes: Dict[str, Union["NodeMeta", "GraphMeta"]]
    graph: Graph
    graph_module_name: str

    def __str__(self) -> str:
        return ""

    def __deepcopy__(self, memo: Dict[Any, Any]) -> "GraphMeta":
        """Graph's deepcopy clones its nodes, but we need object identity for some of our
        manipulations. Hence we need to manually set our torch.fx.Node instances to the newly cloned
        objects, rather than accepting the default deepcopy behavior.
        """
        # Conveniently, Graph's deepcopy populates memo with a mapping from previous node objects to
        # their corresponding cloned values, so we can just look them up. Make sure the value we
        # pass in is truthy so the implementation keeps the reference.
        if not memo:
            memo = {None: None}
        graph_copy = deepcopy(self.graph, memo)
        # ... convenient, except the output node not being added.
        original_output_node = next(n for n in self.graph.nodes if n.op == "output")
        memo[original_output_node] = next(n for n in graph_copy.nodes if n.op == "output")
        return GraphMeta(
            name=self.name,
            local_name=self.local_name,
            shape=deepcopy(self.shape, memo),
            parent=self.parent,
            accelerate_hook=deepcopy(self.accelerate_hook, memo),
            node=cast(Node, memo.get(self.node)),
            nodes=deepcopy(self.nodes, memo),
            code=self.code,
            graph=graph_copy,
            graph_module_name=self.graph_module_name,
        )


class GraphMetaWrapper(NodeDataWrapper[Union[GraphMeta, NodeMeta]]):
    def _name_for(self, node: Node, namespace: _Namespace) -> str:
        """torch.compile() will rename placeholder nodes if the function happens to have a parameter
        that shadows a global. For example, the input to torch.nn.Linear is named "input," but the
        node gets named "input_1", which is not what a user would expect when patching that value.
        So for placeholders, we override the local_name with target, which should match the name in
        the signature of forward().
        """
        # node.target is always a string for placeholders.
        name = cast(str, node.target) if node.op == "placeholder" else node.name

        # Disallow special names to protect our REPL functionality. Borrowing the "sub_"
        # behavior torch already uses for leading underscores on submodules. Use the graph
        # namespace to avoid edge case of collisions with existing nodes.
        if name in _GRAPHPATCH_RESERVED_NAMES:
            # The node itself will be cached in the namespace during compilation, which we can bust
            # by registering vs (self, node). This will also cache for multiple calls to _name_for
            # as is desirable.
            name = namespace.create_name("sub_" + name[1:], (self, node))
        return name

    def _code_for(
        self,
        module: GraphModule,
        node: Optional[Node],
        output_meta: Optional[NodeData[NodeMeta]] = None,
    ) -> WrappedCode:
        # node is a call_module into the current module.
        if node is None or node.graph is not module.graph:
            base_code = module.code
            # Show the context of the original code calling this module.
            if node is not None and node.meta.get("stack_trace"):
                code = f"Calling context:\n{node.meta['stack_trace']}\nCompiled code:\n"
            else:
                code = ""
            # Last 4 lines are nonsense we added to handle unflattening compile() output, which the
            # user doesn't need to see.
            code += "\n".join(base_code.split("\n")[:-4]).strip()
            # Add in our prettified return statement.
            if output_meta is not None and output_meta._value is not NodeData._NO_VALUE:
                code += f"\n    {output_meta._value.code}"
        elif node.op == "output":
            # Use the inputs to our match_shape function to pretty-print the node's output
            # module._graphpatch_output_indexes
            match_shape_node = list(node._input_nodes.keys())[0]
            output_args = match_shape_node.args[1:]

            def pretty_print_output(
                path: str, index: Any
            ) -> Union[Literal[NodeData.Sentinels._NO_VALUE], WrappedCode]:
                if not isinstance(index, int):
                    return NodeData._NO_VALUE
                return WrappedCode(cast(Node, output_args[index]).name)

            mapped = module._graphpatch_output_indexes.map(pretty_print_output)
            code = f"return {mapped.unwrap()}"
        else:
            code = str(node.meta.get("stack_trace", "<no source available>"))

        return WrappedCode(code.strip())

    def handle_wrap(self, data: Any, path: str) -> MaybeHandledData:
        if not isinstance(data, GraphModule):
            return NodeData._UNHANDLED_VALUE

        # Create meta nodes bottom-up, so we can construct this iteratively rather than recursively
        node_meta: DefaultDict[str, Dict[str, NodeData[Union[GraphMeta, NodeMeta]]]] = defaultdict(
            dict
        )
        graph_module_stack: List[Tuple[str, GraphModule]] = []
        module_stack: List[Tuple[str, Module]] = [("", data)]
        # First figure out the module hierarchy in terms of the name of the node that calls each
        # submodule in its parent graph.
        while module_stack:
            cur_name, cur_module = module_stack.pop()
            if not isinstance(cur_module, GraphModule):
                continue
            if cur_name == "":
                name_prefix = ""
            else:
                name_prefix = f"{cur_name}."

            graph_module_stack.append((cur_name, cur_module))
            for node in cur_module.graph.nodes:
                if node.op == "call_module" and isinstance(
                    target := cur_module.get_submodule(node.target), GraphModule
                ):
                    qual_name = f"{name_prefix}{node.name}"
                    module_stack.append((qual_name, target))
                    graph_module_stack.append((qual_name, target))

        while graph_module_stack:
            cur_name, cur_module = graph_module_stack.pop()
            if cur_name == "":
                name_prefix = ""
            else:
                name_prefix = f"{cur_name}."

            for node in cur_module.graph.nodes:
                if node.meta.get("_graphpatch_hidden", False):
                    continue
                if node.op == "call_module" and isinstance(
                    target := data.get_submodule(f"{name_prefix}{node.target}"), GraphModule
                ):
                    sub_nodes = node_meta[f"{name_prefix}{node.name}"]
                    node_meta[cur_name][node.name] = NodeData(
                        _original_type=Graph,
                        _children=sub_nodes,
                        _value=GraphMeta(
                            name=f"{name_prefix}{node.name}",
                            local_name=node.name,
                            accelerate_hook=getattr(target, "_hf_hook", None),
                            node=node,
                            nodes={
                                k: v._value
                                for k, v in sub_nodes.items()
                                if v._value is not NodeData._NO_VALUE
                            },
                            parent=cur_name,
                            graph=target.graph,
                            graph_module_name=f"{name_prefix}{node.target}",
                            code=self._code_for(
                                target, node, cast(NodeData[NodeMeta], sub_nodes["output"])
                            ),
                        ),
                        _path=f"{name_prefix}{node.name}",
                    )
                else:
                    namespace = cur_module.graph._graph_namespace
                    node_meta[cur_name][self._name_for(node, namespace)] = NodeData(
                        _original_type=Node,
                        _value=NodeMeta(
                            name=f"{name_prefix}{self._name_for(node, namespace)}",
                            local_name=self._name_for(node, namespace),
                            node=node,
                            parent=cur_name,
                            code=self._code_for(cur_module, node),
                        ),
                        _path=f"{name_prefix}{self._name_for(node, namespace)}",
                    )

        return NodeData(
            _original_type=Graph,
            _children=node_meta[""],
            _path="",
            _value=GraphMeta(
                name="",
                local_name="",
                accelerate_hook=getattr(data, "_hf_hook", None),
                node=None,
                nodes={
                    k: v._value
                    for k, v in node_meta[""].items()
                    if v._value is not NodeData._NO_VALUE
                },
                code=self._code_for(data, None, cast(NodeData[NodeMeta], node_meta[""]["output"])),
                parent="",
                graph=data.graph,
                graph_module_name="",
            ),
        )


def wrap_graph_module(graph_module: GraphModule) -> NodeData[Union[GraphMeta, NodeMeta]]:
    return GraphMetaWrapper(NodeData[Union[GraphMeta, NodeMeta]]).wrap(graph_module)


class OutputArgumentIndexWrapper(NodeDataWrapper[int]):
    """Maps the true structure of a module's output to each node's depth-first index, matching the
    flattening order of torch.compile(). This lets us recreate the original output shape despite
    torch.compile()'s flattening.
    """

    def __init__(self, child_output_ids: Set[int], *args: Any, **kwargs: Any):
        super().__init__(NodeData[int], *args, **kwargs)
        self.cur_index: int = 0
        self.id_map: Dict[int, int] = {}
        self.child_output_ids: Set[int] = child_output_ids

    def handle_wrap(self, data: Any, path: str) -> MaybeHandledData:
        # Short-circuit in case a child module outputs a container; we don't want to dig into the
        # container elements, since the whole container will be passed as one unit.
        if id(data) in self.child_output_ids:
            if id(data) not in self.id_map:
                self.id_map[id(data)] = self.cur_index
                self.cur_index += 1
            return self.make_wrapper(
                _value=self.id_map[id(data)], _original_type=data.__class__, _path=path
            )
        return NodeData._UNHANDLED_VALUE

    def handle_leaf(self, data: Any, path: str) -> NodeData[int]:
        if not isinstance(data, Tensor):
            return self.make_wrapper(_original_type=data.__class__, _value=-1, _path=path)
        if id(data) not in self.id_map:
            self.id_map[id(data)] = self.cur_index
            self.cur_index += 1
        return self.make_wrapper(
            _original_type=data.__class__, _value=self.id_map[id(data)], _path=path
        )


def wrap_output_argument_index(data: Any, child_output_ids: Set[int]) -> NodeData[int]:
    return OutputArgumentIndexWrapper(child_output_ids).wrap(data)
