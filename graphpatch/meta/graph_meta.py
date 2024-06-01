from collections import defaultdict, deque
from copy import deepcopy
from typing import (
    Any,
    ClassVar,
    DefaultDict,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from torch import Size, Tensor
from torch.fx.graph import Graph, _Namespace
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from ..extraction.compiled_graph_module import CompiledGraphModule
from ..extraction.graphpatch_module import GraphPatchModule
from ..extraction.opaque_graph_module import OpaqueGraphModule, SubmoduleWrapper
from ..optional.dataclasses import dataclass, field
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
                _original_type=type(data),
                _value=NodeShape(_shape=data.shape, _data_type="Tensor"),
                _path=path,
            )

        return self.make_wrapper(
            _original_type=type(data),
            _value=NodeShape(_shape=None, _data_type=type(data).__name__),
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
    hidden: bool


@dataclass(kw_only=True)
class NodeMeta(_BaseMeta):
    """Meta-info to associate with a non-graph node in a GraphModule.

    Attributes:
        node: torch.fx.Node instance this meta-info is associated to.
        parameter_expected: Does torch expect the output of this node to be a Parameter?
    """

    node: Node
    is_graph: ClassVar[bool] = False
    parameter_expected: bool

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
                hidden=self.hidden,
                parameter_expected=self.parameter_expected,
            )
        return cast(NodeMeta, memo[id(self)])


@dataclass(kw_only=True)
class GraphMeta(_BaseMeta):
    """Meta-info to associate with a subgraph node in a GraphModule.

    Attributes:
        node: torch.fx.Node instance this meta-info is associated to. None for the root.
        nodes: Dictionary mapping node names to child meta-info.
        graph: torch.fx.Graph instance this meta-info is associated to.
        graph_module_name: Name of the graph_module within the module hierarchy this meta-info is
            associated to.
        graph_module_class: Class of the graph_module, so we can easily distinguish between opaque
            and compiled graphs when deserializing.
    """

    is_graph: ClassVar[bool] = True
    node: Optional[Node]
    nodes: Dict[str, Union["NodeMeta", "GraphMeta"]]
    graph: Graph
    graph_module_name: str
    graph_module_class: Type[GraphModule]

    def __str__(self) -> str:
        return ""

    def _graphpatch_graph_repr(self) -> str:
        return self.graph_module_class.__name__

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
            node=cast(Node, memo.get(self.node)),
            nodes=deepcopy(self.nodes, memo),
            code=self.code,
            graph=graph_copy,
            graph_module_name=self.graph_module_name,
            graph_module_class=self.graph_module_class,
            hidden=self.hidden,
        )


@dataclass(kw_only=True)
class ModuleName:
    """Utility class to map the divergence between node names and names in the Module hierarchy."""

    @staticmethod
    def _prefix_for(name: str) -> str:
        if name == "":
            return ""
        return f"{name}."

    meta: str
    meta_prefix: str = field(init=False)
    module: str
    module_prefix: str = field(init=False)

    def __post_init__(self) -> None:
        self.meta_prefix = ModuleName._prefix_for(self.meta)
        self.module_prefix = ModuleName._prefix_for(self.module)


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
        # Strip asterisk from varargs, since that would be invalid as an identifier.
        name = name.replace("*", "")

        # Disallow special names to protect our REPL functionality by adding the "sub_" prefix,
        # which mirrors how torch.compile() handles node names that would be invalid identifiers.
        if name in _GRAPHPATCH_RESERVED_NAMES:
            # The node itself will be cached in the namespace during compilation, which we can bust
            # by registering vs (self, node). This will also cache for multiple calls to _name_for
            # as is desirable.
            name = namespace.create_name("sub" + name, (self, node))
        elif (
            node.op == "placeholder"
            and name in namespace._used_names
            and namespace._obj_to_name.get(node) != name
        ):
            # Node will have been registered by name, not target, so make sure our target name isn't
            # already taken!
            name = namespace.create_name(name, (self, node))
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
            match_shape_node = list(node._input_nodes.keys())[0]
            output_args = match_shape_node.args[1:]

            def pretty_print_output(path: str, index: OutputArgumentIndex) -> WrappedCode:
                if index.index is None:
                    return WrappedCode("")
                return WrappedCode(cast(Node, output_args[index.index]).name)

            if module._graphpatch_output_indexes._value.should_unwrap:
                mapped = module._graphpatch_output_indexes.map(pretty_print_output)
                code = f"return {mapped.unwrap()}"
            else:
                code = f"return {match_shape_node.args[1]}"
        else:
            code = str(node.meta.get("stack_trace", "<no source available>"))

        return WrappedCode(code.strip())

    def _graph_module_class_for(self, module: GraphPatchModule) -> Type[GraphPatchModule]:
        # NB: we can't just return __class__ because torch creates a unique subclass per instance,
        # which isn't picklable.
        if isinstance(module, OpaqueGraphModule):
            return OpaqueGraphModule
        return CompiledGraphModule

    def _graph_module_target(self, node: Node) -> Optional[GraphPatchModule]:
        if (
            # Real call to submodule.
            node.op == "call_module"
            # Dummy call from opaque module to submodule, which we should display as a graph.
            or (node.op == "call_function" and isinstance(node.target, SubmoduleWrapper))
        ) and isinstance(
            target := node.graph.owning_module.get_submodule(str(node.target)),
            GraphModule,
        ):
            return target  # type: ignore
        return None

    def _graph_module_hierarchy(
        self, root_module: GraphModule
    ) -> Iterator[Tuple[ModuleName, GraphModule]]:
        """Iterate through the GraphModule hierarchy such that children are returned before their
        parents, but in the order in which they should be added to the graph.
        """
        module_queue: Deque[Tuple[ModuleName, GraphModule]] = deque(
            [(ModuleName(meta="", module=""), root_module)]
        )
        result_stack: List[Tuple[ModuleName, GraphModule]] = []
        while module_queue:
            name, module = module_queue.popleft()

            result_stack.append((name, module))

            for node in reversed(module.graph.nodes):
                if (target := self._graph_module_target(node)) is not None:
                    module_queue.append(
                        (
                            ModuleName(
                                meta=f"{name.meta_prefix}{node.name}",
                                module=f"{name.module_prefix}{node.target}",
                            ),
                            target,
                        )
                    )
        while result_stack:
            yield result_stack.pop()

    def handle_wrap(self, data: Any, path: str) -> MaybeHandledData:  # type: ignore[type-arg]
        if not isinstance(data, GraphModule):
            return NodeData._UNHANDLED_VALUE

        # Create meta nodes bottom-up so we can construct this iteratively rather than recursively.
        node_meta: DefaultDict[str, Dict[str, NodeData[Union[GraphMeta, NodeMeta]]]] = defaultdict(
            dict
        )
        for name, module in self._graph_module_hierarchy(data):
            for node in module.graph.nodes:
                if target := self._graph_module_target(node):
                    sub_nodes = node_meta[f"{name.meta_prefix}{node.name}"]
                    node_meta[name.meta][node.name] = NodeData(
                        _original_type=self._graph_module_class_for(target),
                        _children=sub_nodes,
                        _value=GraphMeta(
                            name=f"{name.meta_prefix}{node.name}",
                            local_name=node.name,
                            node=node,
                            nodes={
                                k: v._value
                                for k, v in sub_nodes.items()
                                if v._value is not NodeData._NO_VALUE
                            },
                            parent=name.meta,
                            graph=target.graph,
                            graph_module_name=f"{name.module_prefix}{node.target}",
                            graph_module_class=self._graph_module_class_for(target),
                            code=self._code_for(
                                target, node, cast(NodeData[NodeMeta], sub_nodes["output"])
                            ),
                            hidden=node.meta.get("_graphpatch_hidden", False),
                        ),
                        _path=f"{name.meta_prefix}{node.name}",
                    )
                else:
                    namespace = module.graph._graph_namespace
                    node_meta[name.meta][self._name_for(node, namespace)] = NodeData(
                        _original_type=Node,
                        _value=NodeMeta(
                            name=f"{name.meta_prefix}{self._name_for(node, namespace)}",
                            local_name=self._name_for(node, namespace),
                            node=node,
                            parent=name.meta,
                            code=self._code_for(module, node),
                            hidden=node.meta.get("_graphpatch_hidden", False),
                            parameter_expected=False,
                        ),
                        _path=f"{name.meta_prefix}{self._name_for(node, namespace)}",
                    )

        return NodeData(
            _original_type=self._graph_module_class_for(data),
            _children=node_meta[""],
            _path="",
            _value=GraphMeta(
                name="",
                local_name="",
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
                graph_module_class=self._graph_module_class_for(data),
                hidden=node.meta.get("_graphpatch_hidden", False),
            ),
        )


def wrap_graph_module(graph_module: GraphModule) -> NodeData[Union[GraphMeta, NodeMeta]]:
    return GraphMetaWrapper(NodeData[Union[GraphMeta, NodeMeta]]).wrap(graph_module)


@dataclass
class OutputArgumentIndex:
    index: Optional[int]
    should_unwrap: bool


class OutputArgumentIndexWrapper(NodeDataWrapper[OutputArgumentIndex]):
    """Maps the true structure of a module's output to each node's depth-first index, matching the
    flattening order of torch.compile(). This lets us recreate the original output shape despite
    torch.compile()'s flattening.
    """

    child_output_ids: Set[int]
    cur_index: int
    id_map: Dict[int, OutputArgumentIndex]
    should_unwrap: bool

    def __init__(self, child_output_ids: Set[int], should_unwrap: bool):
        super().__init__(NodeData[OutputArgumentIndex])
        self.child_output_ids = child_output_ids
        self.cur_index = 0
        self.id_map = {}
        self.should_unwrap = should_unwrap

    def handle_wrap(self, data: Any, path: str) -> MaybeHandledData:  # type: ignore[type-arg]
        # Short-circuit in case a child module outputs a container; we don't want to dig into the
        # container elements, since the whole container will be passed as one unit.
        if id(data) in self.child_output_ids:
            if id(data) not in self.id_map:
                # TODO: should_unwrap=False? does this actually work?
                self.id_map[id(data)] = OutputArgumentIndex(self.cur_index, self.should_unwrap)
                self.cur_index += 1
            return self.make_wrapper(
                _value=self.id_map[id(data)], _original_type=type(data), _path=path
            )
        return NodeData._UNHANDLED_VALUE

    def handle_leaf(self, data: Any, path: str) -> NodeData[OutputArgumentIndex]:
        # TODO: handle other value types; should determine whether the value came from a node in
        # the graph (add it), or not (don't)
        if not isinstance(data, Tensor):
            return self.make_wrapper(
                _original_type=type(data), _value=NodeData._NO_VALUE, _path=path
            )
        if id(data) not in self.id_map:
            self.id_map[id(data)] = OutputArgumentIndex(self.cur_index, self.should_unwrap)
            self.cur_index += 1
        return self.make_wrapper(
            _original_type=type(data), _value=self.id_map[id(data)], _path=path
        )


def wrap_output_argument_index(
    data: Any, child_output_ids: Set[int], should_unwrap: bool
) -> NodeData[OutputArgumentIndex]:
    wrapper = OutputArgumentIndexWrapper(child_output_ids, should_unwrap).wrap(data)
    if wrapper._value is NodeData._NO_VALUE:
        wrapper._value = OutputArgumentIndex(None, should_unwrap)
    return wrapper
