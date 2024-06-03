from collections import defaultdict
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
from warnings import warn

import torch
from torch import Tensor, no_grad
from torch.fx.graph import CodeGen, _Namespace
from torch.fx.graph_module import GraphModule, _forward_from_src
from torch.fx.node import Node, Node as FXNode
from torch.nn import Module, Parameter
from torch.serialization import FILE_LIKE
from typing_extensions import TypedDict

from . import hacks
from .exceptions import GraphPatchWarning
from .extraction import ExtractionOptions, extract
from .meta import (
    GraphMeta,
    NodeData,
    NodeMeta,
    NodePath,
    wrap_node_data,
    wrap_node_path,
    wrap_node_shape,
)
from .optional.transformers import GenerationConfig, GenerationMixin, PretrainedConfig
from .optional.typing_extensions import TypeAlias
from .patch import Patch, PatchableValue

GraphPatchArgs = TypedDict(
    "GraphPatchArgs",
    {"_trace_output_shape": bool, "patch": Mapping[str, List[Patch[Tensor]]]},
    total=False,
)

FileLike: TypeAlias = FILE_LIKE


def _make_patch_wrapper(
    node_qual_name: str,
    inner_fn: Callable[[GraphModule, List[Any]], Any],
    graph_module: GraphModule,
    node: NodeMeta,
) -> Callable[[List[Any]], Any]:
    def maybe_clone(value: Any) -> Any:
        # TODO (minor optimization): there should be cases where not cloning the original output has
        # no possible side-effects, in which case we can skip.
        if isinstance(value, Tensor):
            return value.clone()
        return value

    def patch_wrapper(*outer_args: Any, _graphpatch_args: Optional[GraphPatchArgs] = None) -> Any:
        patch_args = _graphpatch_args or {}
        patches = (patch_args.get("patch") or {}).get(node_qual_name, [])

        # TODO (optimization): don't compute if patch will entirely overwrite
        output = inner_fn(graph_module, *outer_args)

        # TODO: it should be possible to extract shapes during the initial graph extraction
        if patch_args.get("_trace_output_shape", False):
            node.shape = wrap_node_shape(output)
            node.parameter_expected = isinstance(output, Parameter)

        if not patches:
            return output

        wrapped_output = wrap_node_data(output)

        if any(p.requires_clone for p in patches):
            wrapped_output.map_in_place(maybe_clone)

        for patch in patches:
            wrapped_output.replace(patch.path, patch)

        unwrapped_output = wrapped_output.unwrap()

        # User convenience for patches that replace Parameters, allowing bare Tensors to be passed
        # without torch raising an exception.
        if node.parameter_expected and not isinstance(unwrapped_output, Parameter):
            return Parameter(unwrapped_output)
        return unwrapped_output

    return patch_wrapper


class PatchableGraph(Module):
    """PatchableGraph is a wrapper around :class:`torch.nn.Module` allowing activation patching at
    any tensor operation. It is the main entrypoint for ``graphpatch``'s functionality.

    Internally, PatchableGraph builds a :class:`torch.fx.GraphModule` for the module and each of its
    submodules using :func:`torch.compile`. This exposes the computational structure of the module
    while still being equivalent to the original--you can perform any operation you would with the
    original module using the PatchableGraph. In case compilation fails--``compile()`` is not yet
    compatible with all model code--PatchableGraph will fall back to patching submodule input,
    output, parameters, and buffers. See :class:`ExtractionOptions` for options controlling this
    behavior and :ref:`notes_on_compilation` for more discussion.

    To perform activation patching, use the :meth:`patch <graphpatch.PatchableGraph.patch>` context
    manager. This method takes a mapping from :ref:`NodePaths <node_path>` to lists of
    :ref:`Patch <patch>` to apply at the corresponding node. Note that the activation patches will
    only be applied inside the context block; using the PatchableGraph outside such a block is
    equivalent to running the original module.

    Example:
        >>> from graphpatch import PatchableGraph, ZeroPatch
        >>> my_llm, my_tokenizer = MyLLM(), MyTokenizer()
        >>> my_inputs = MyTokenizer("Hello, ")
        >>> patchable_graph = PatchableGraph(my_llm, **my_inputs)
        # Patch the input to the third layer's MLP
        >>> with patchable_graph.patch({"layers_2.mlp.x": [ZeroPatch()]):
        >>>    patched_output = patchable_graph(**my_inputs)

    Parameters:
        module: The :class:`Module <torch.nn.Module>` to wrap.
        extraction_options_and_args: Arguments (example inputs) to be passed to the module during
            :func:`torch.compile`. If the first argument is an :class:`ExtractionOptions` instance,
            apply them during graph extraction.
        extraction_kwargs: Keyword arguments to be passed to the module during
            :func:`torch.compile()`.
    """

    _graphpatch_class_name = "PatchableGraph"

    def __new__(cls: Type["PatchableGraph"], *args: Any, **kwargs: Any) -> "PatchableGraph":
        """Create a bespoke class on instantiation ala GraphModule, so we can freely modify it
        later to support transformers generation methods if needed.
        """

        class PatchableGraphInstance(PatchableGraph):
            pass

        return super().__new__(PatchableGraphInstance)

    def __init__(
        self,
        module: Module,
        *extraction_options_and_args: Tuple[Union[Any, ExtractionOptions], ...],
        **extraction_kwargs: Dict[str, Any],
    ):
        super().__init__()
        type(self).__name__ = self._graphpatch_class_name

        # Pull extraction args from positional arguments, if present; otherwise, use defaults.
        if len(extraction_options_and_args) > 0:
            if isinstance(extraction_options_and_args[0], ExtractionOptions):
                extraction_options: ExtractionOptions = extraction_options_and_args[0]
                extraction_args = extraction_options_and_args[1:]
            else:
                extraction_options = ExtractionOptions()
                extraction_args = extraction_options_and_args
        else:
            extraction_options = ExtractionOptions()
            extraction_args = ()

        graph_module, meta = extract(
            module,
            extraction_options,
            *extraction_args,
            **extraction_kwargs,
        )
        if graph_module is None or meta is None:
            raise ValueError("Unable to extract graph.")

        self._graphpatch_override_generation_kwargs = {"use_cache": False, "cache_position": None}
        self._patch_context: Optional[Dict[str, List[Patch[Tensor]]]] = None
        self._graph_module = graph_module
        self._meta = meta
        self._original_graph = deepcopy(meta)
        # In torch >= 2.1.0, FakeTensors get attached in each FXNode's meta, but they are
        # unpicklable.
        for node_meta in self._original_graph.values():
            if node_meta.node is not None:
                node_meta.node.meta.pop("example_value", None)
        self._make_patchable()
        self._trace_output_shapes(*extraction_args, **extraction_kwargs)
        self._node_path = wrap_node_path(self._meta)
        self._is_saving = False

        self.config = None
        self.generation_config = None
        try:
            if extraction_options.copy_transformers_generation_config and isinstance(
                module, GenerationMixin
            ):
                self._copy_transformers_generation_config(
                    type(module), module.config, module.generation_config
                )
        except Exception as exc:
            warn(
                (
                    "Failed to set up transformers generation on the PatchableGraph. You will be"
                    " unable to use convenience methods like generate(). You can disable setup of"
                    " generation by setting copy_transformers_generation_config=False in your"
                    " extraction options.\n\n"
                    "Exception:\n"
                    "**************************\n"
                    f"{exc}\n"
                    "**************************\n\n"
                    "User code:\n"
                ),
                GraphPatchWarning,
                stacklevel=2,
            )

    @staticmethod
    def load(file: FileLike, *args: Any, **kwargs: Any) -> "PatchableGraph":
        """Wrapper around :func:`torch.load()`. All the normal caveats around pickling apply; you
        should not load() anything you downloaded from the Internet.

        Future versions of graphpatch will likely implement a more secure serialization scheme and
        disable the built-in torch.load().
        """
        return cast(
            PatchableGraph,
            torch.load(file, *args, **kwargs),
        )

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Wrapper around :func:`torch.save()` because some PatchableGraph internals may need to be
        handled specially before pickling.

        Future versions of graphpatch will likely implement a more secure serialization scheme and
        disable the built-in torch.save().
        """
        self._is_saving = True
        try:
            torch.save(self, *args, **kwargs)
        finally:
            self._is_saving = False

    @classmethod
    def _unpickle(
        cls,
        state_dict: Dict[str, Any],
        parameter_names: Set[str],
        _original_graph: NodeData[Union[NodeMeta, GraphMeta]],
        _generation_mixin_masquerade_class: Optional[Type[GenerationMixin]],
        config: Optional[PretrainedConfig],
        generation_config: Optional[GenerationConfig],
        _graphpatch_override_generation_kwargs: Dict[str, Any],
    ) -> "PatchableGraph":
        deserialized_instance = cls.__new__(cls)
        super().__init__(deserialized_instance)
        deserialized_instance._patch_context = None
        deserialized_instance._original_graph = deepcopy(_original_graph)
        deserialized_instance._meta = _original_graph

        if _generation_mixin_masquerade_class is not None:
            deserialized_instance._copy_transformers_generation_config(
                _generation_mixin_masquerade_class, config, generation_config
            )
        deserialized_instance._graphpatch_override_generation_kwargs = (
            _graphpatch_override_generation_kwargs
        )

        state_by_submodule: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # For cloned graphs (and probably tied weights?) we will have multiple instances of the same
        # object in our state dict. Make sure we maintain that relationship by only constructing
        # Parameters once.
        state_entries_to_parameters: Dict[Any, Parameter] = {}
        for qualified_name, state_entry in state_dict.items():
            [*parent_path, target] = qualified_name.split(".")
            if isinstance(state_entry, Tensor) and state_entry in state_entries_to_parameters:
                parameter = state_entries_to_parameters[state_entry]
            elif qualified_name in parameter_names:
                parameter = Parameter(
                    state_dict[qualified_name],
                    requires_grad=(state_dict[qualified_name].requires_grad),
                )
                state_entries_to_parameters[state_entry] = parameter
            else:
                parameter = state_entry
            state_by_submodule[".".join(parent_path)][target] = parameter

        # GraphModule is not built to handle nested graphs, so re-inflate each sub-graph
        # individually. Process children before their parents so we can do this in one pass.
        submodules_by_parent: Dict[str, Dict[str, Module]] = defaultdict(dict)

        for meta in deserialized_instance._meta.reverse_topological_order():
            if not isinstance(meta, GraphMeta):
                continue
            name = meta.graph_module_name
            if name != "":
                name = f"_graph_module.{name}"
            else:
                name = "_graph_module"
            parent = cast(GraphMeta, deserialized_instance._meta[meta.parent])
            if parent.graph_module_name != "":
                parent_name = f"_graph_module.{parent.graph_module_name}"
            else:
                parent_name = "_graph_module"
            target = cast(str, meta.node.target) if meta.node is not None else ""

            local_submodules = submodules_by_parent[name]
            local_state = state_by_submodule[name]

            graph_module = meta.graph_module_class(
                {
                    **local_state,
                    **local_submodules,
                },
                meta.graph,
                meta.graph_module_class.__name__,
            )

            submodules_by_parent[parent_name][str(target)] = graph_module

        deserialized_instance._graph_module = cast(
            GraphModule, submodules_by_parent["_graph_module"][""]
        )
        deserialized_instance._make_patchable()
        deserialized_instance._node_path = wrap_node_path(deserialized_instance._meta)
        deserialized_instance._is_saving = False

        return deserialized_instance

    def __reduce__(self) -> Tuple[Callable[..., "PatchableGraph"], Tuple[Any, ...]]:
        """Set up custom serialization for when user calls torch.save(), since our node wrappers are
        unpicklable.
        """
        if not self._is_saving:
            raise ValueError(
                "Do not call torch.save() directly on PatchableGraph. Instead, call self.save()."
            )

        # Use keep_vars to maintain object identity between cloned (or tied) parameters. We may
        # want to clean these up by creating a "canonical" detached version, since we probably
        # don't want to persist autograd state.
        state_dict = self.state_dict(keep_vars=True)

        # Handle non-state attributes, like variance_epsilon in LlamaRMSNorm, and for
        # OpaqueGraphModule. We can find the ones that matter by searching all our subgraphs for
        # references.
        for node in self._original_graph.values():
            if (
                isinstance(node, NodeMeta)
                and node.node.op == "get_attr"
                # Don't want to pickle OpaqueGraphModule itself.
                and node.node.target != "_graphpatch_self"
            ):
                target = cast(str, node.node.target)
                parent = cast(GraphMeta, self._original_graph[node.parent])
                parent_graph = parent.graph_module_name
                key = f"_graph_module.{parent_graph + ('.' if parent_graph else '')}{target}"
                if key in state_dict:
                    continue
                parent = cast(GraphMeta, self._original_graph[node.parent])
                graph_module = self._graph_module.get_submodule(parent.graph_module_name)
                state_dict[key] = getattr(graph_module, target)
        return (
            PatchableGraph._unpickle,
            (
                state_dict,
                {name for name, _ in self.named_parameters()},
                self._original_graph,
                type(self).__bases__[-1] if isinstance(self, GenerationMixin) else None,
                self.config,
                self.generation_config,
                self._graphpatch_override_generation_kwargs,
            ),
        )

    # Using the more idiomatic decorator version breaks tab-completion in IPython. This seems to be
    # because it uses the annotated types to determine completable values, but we want it to use
    # our custom __dir__ instead. This is a hacky way to prevent IPython from seeing the true type.
    graph = property(
        lambda self: self._node_path,
        doc="""graph(NodePath)
    Convenience property for working in REPL and notebook environments. Exposes
    the full :class:`NodePath <meta.NodePath>` hierarchy of this PatchableGraph via recursive attribute
    access. Children of the current node can be tab-completed at each step. Has a custom
    ``__repr__()`` to display the subgraph rooted at the current path.

    Example:

    .. ipython::
        :verbatim:

        In [1]: pg.graph
        Out[1]:
        <root>: Graph(3)
        ├─x: Tensor(3, 2)
        ├─linear: Graph(5)
        │ ├─input: Tensor(3, 2)
        │ ├─weight: Tensor(3, 2)
        │ ├─bias: Tensor(3)
        │ ├─linear: Tensor(3, 3)
        │ └─output: Tensor(3, 3)
        └─output: Tensor(3, 3)

        In [2]: pg.graph.linear._code
        Out[2]:
        Calling context:
        File "/Users/evanlloyd/graphpatch/tests/fixtures/minimal_module.py", line 16, in forward
            return self.linear(x)
        Compiled code:
        def forward(self, input : torch.Tensor):
            input_1 = input
            weight = self.weight
            bias = self.bias
            linear = torch._C._nn.linear(input_1, weight, bias);  input_1 = weight = bias = None
            return linear

        In [3]: pg.graph.output._shape
        Out[3]: torch.Size([3, 3])

    Also see :ref:`node_path` for more discussion and examples.
    """,
    )

    @contextmanager
    def patch(
        self,
        patch_map: Dict[
            Union[str, NodePath], Union[List[Patch[PatchableValue]], Patch[PatchableValue]]
        ],
    ) -> Iterator[None]:
        """Context manager that will cause the given activation patches to be applied when running
        inference on the wrapped module.

        Parameters:
            patch_map: A mapping from :ref:`NodePath <node_path>` to a :ref:`patch` or list of
                :ref:`Patches <patch>` to apply to each respective node during inference.

        Yields:
                A context in which the given activation patch(es) will be applied when calling
                ``self.forward()``.

        Raises:
            KeyError: If any :ref:`NodePath <node_path>` in ``patch_map`` does not exist in the graph.
            ValueError: If ``patch_map`` has any invalid types.
        """
        if (
            not isinstance(patch_map, dict)
            or any(not isinstance(k, (str, NodePath)) for k in patch_map)
            or any(not isinstance(v, (list, Patch)) for v in patch_map.values())
            or any(
                any(not isinstance(p, Patch) for p in v)
                for v in patch_map.values()
                if isinstance(v, list)
            )
        ):
            raise ValueError(
                "patch_map must be a dictionary mapping strings or NodePaths to (lists of) Patch."
            )
        prev_patch_context = self._patch_context

        # Convert any singleton patch values into lists.
        for k, v in patch_map.items():
            if isinstance(v, Patch):
                patch_map[k] = [v]

        # Convert any NodePaths into str.
        converted_patch_map: Dict[str, List[Patch[Tensor]]] = {}
        invalid_paths = []
        for k, v in patch_map.items():
            if isinstance(k, NodePath):
                try:
                    converted_patch_map[k.to_path(allow_internal=False)] = v  # type: ignore
                except ValueError:
                    invalid_paths.append(str(k._value))
            else:
                converted_patch_map[k] = v  # type: ignore

        # Validate the given node_paths; we should fail early if the given patches can't be applied.
        for node_path in converted_patch_map.keys():
            if "|" in node_path:
                [node_name, *parsed_path] = node_path.split("|")
                if len(parsed_path) > 1:
                    invalid_paths.append(node_path)
                    continue
                path = parsed_path[0]
            else:
                node_name = node_path
                path = None

            # Must be a leaf.
            meta = self._meta.get(node_name)
            if not isinstance(meta, NodeMeta):
                invalid_paths.append(node_path)
                continue

            # Validate path; must reach a leaf of shape.
            if path is None:
                continue
            if meta.shape is None or path not in meta.shape or meta.shape._dig(path).is_internal:
                invalid_paths.append(node_path)

        if invalid_paths:
            raise KeyError(f"Invalid node_path(s): {', '.join(invalid_paths)}")

        # NB: using a shallow copy intentionally here; otherwise we disrupt the Autograd graph when
        # the patches have tensor arguments!
        if self._patch_context is None:
            self._patch_context = converted_patch_map
        else:
            self._patch_context = {k: copy(v) for k, v in self._patch_context.items()}
            for key, patches in converted_patch_map.items():
                if key in self._patch_context:
                    self._patch_context[key].extend(patches)
                else:
                    self._patch_context[key] = patches

        # Parse paths for nodes with nested contents.
        keys_to_update = [key for key in self._patch_context if "|" in key]
        for key in keys_to_update:
            [node_name, path] = key.split("|")
            patches = self._patch_context.get(node_name, [])
            for p in self._patch_context[key]:
                p.path = path
            patches.extend([p for p in self._patch_context[key]])
            self._patch_context[node_name] = patches
            del self._patch_context[key]

        try:
            yield
        finally:
            self._patch_context = prev_patch_context

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # TODO: replace nodes just-in-time (rather than having overhead at every node)
        if self._patch_context is not None:
            return self._graph_module(
                *args, **kwargs, _graphpatch_args={"patch": self._patch_context}
            )
        else:
            return self._graph_module(*args, **kwargs)

    def _trace_output_shapes(self, *args: Any, **kwargs: Any) -> None:
        with no_grad():
            self._graph_module(*args, **kwargs, _graphpatch_args={"_trace_output_shape": True})

        assert self._original_graph is not None
        # Copy over traced shapes to _original_graph so they'll be included when serializing
        for name, meta in self._original_graph.items():
            original_meta = self._meta[name]
            if isinstance(meta, NodeMeta) and isinstance(original_meta, NodeMeta):
                meta.shape = original_meta.shape

    def _add_patch_args(self, meta: NodeMeta, patch_args_node: FXNode) -> None:
        """Adds patch args into the call to the opaque module wrapper."""
        new_kwargs = dict(**meta.node.kwargs)
        new_kwargs["_graphpatch_args"] = patch_args_node
        meta.node.kwargs = new_kwargs

    def _wrap_graph(
        self, meta: GraphMeta, graph_module: GraphModule, patch_args_node: FXNode
    ) -> None:
        """Wraps a submodule to allow patching its inputs and outputs."""

        # Placeholder nodes must be Nodes, not Graphs, so this cast is safe.
        placeholders: List[NodeMeta] = [
            cast(NodeMeta, n)
            for n in meta.nodes.values()
            if n.node is not None and n.node.op == "placeholder"
        ]
        # Similarly, output nodes must be Nodes.
        output_node: NodeMeta = next(
            cast(NodeMeta, n)
            for n in meta.nodes.values()
            if n.node is not None and n.node.op == "output"
        )

        # Add a function call for each placeholder to handle patches on this module's inputs.
        for input_node in placeholders:
            # _graphpatch_args is always the last placeholder.
            with graph_module.graph.inserting_after(patch_args_node):
                # TODO: type expression?
                wrapper_node = graph_module.graph.call_function(
                    _make_patch_wrapper(input_node.name, lambda _, x: x, graph_module, input_node),
                    (input_node.node,),
                    {"_graphpatch_args": patch_args_node},
                )
                # delete_user_cb lets wrapper node retain the original input.
                input_node.node.replace_all_uses_with(
                    wrapper_node,
                    delete_user_cb=lambda node: node is not wrapper_node,
                    propagate_meta=True,
                )

        # Add a handler for module output patching. Technically, we could re-use the wrapper that
        # will be around whatever value is actually being returned, but being able to refer to
        # "submodule.output" is a much nicer user interface.
        with graph_module.graph.inserting_before(output_node.node):
            wrapper_node = graph_module.graph.call_function(
                _make_patch_wrapper(
                    output_node.name,
                    lambda _, *x: x if len(output_node.node.args) > 1 else x[0],
                    graph_module,
                    output_node,
                ),
                output_node.node.args,
                {"_graphpatch_args": patch_args_node},
            )
            output_node.node.args = (wrapper_node,)

    def _replace_node(
        self, node: FXNode, graph_module: GraphModule, patch_args_node: FXNode, node_qual_name: str
    ) -> FXNode:
        """Wraps a node in our graph with a function that records the output and can perform
        activation patches. We construct an inner function equivalent to the original node by
        creating and compiling a dummy graph treating the arguments as placeholders.
        """
        graph = node.graph
        name_copy = node.name
        input_nodes_copy = list(node.all_input_nodes)
        placeholder_nodes = []
        for input_node in input_nodes_copy:
            placeholder_nodes.append(
                Node(
                    graph,
                    input_node.name,
                    "placeholder",
                    input_node.name,
                    (),
                    {},
                )
            )
        output_node = Node(graph, "output", "output", name_copy, (node,), {})

        # TODO: do this without CodeGen/making a dummy module; we should just be able to set up the
        # function call directly.
        source = CodeGen()._gen_python_code(
            [*placeholder_nodes, node, output_node], "self", _Namespace()
        )
        inner_fn = _forward_from_src(source.src, source.globals)

        node_meta = self._meta[node_qual_name]
        assert isinstance(node_meta, NodeMeta)

        replacement_node: Node = Node(
            graph,
            name_copy,
            "call_function",
            _make_patch_wrapper(node_qual_name, inner_fn, graph_module, node_meta),
            tuple(input_nodes_copy),
            {"_graphpatch_args": patch_args_node},
        )
        with graph.inserting_before(node.next):
            hacks.replace_node_keeping_original_name(node, replacement_node, name_copy)
        return replacement_node

    def _make_patchable(self) -> None:
        """Wrap all nodes and graphs with a handler for the current activation patching context."""
        context_nodes = {}

        def add_context(meta: Union[NodeMeta, GraphMeta]) -> Union[NodeMeta, GraphMeta]:
            if isinstance(meta, GraphMeta):
                # Add a placeholder to receive the patching context
                last_placeholder = None
                for n in (n for n in meta.graph.nodes if n.op == "placeholder"):
                    # Must insert before varkwargs.
                    if not n.target.startswith("**"):
                        last_placeholder = n

                if last_placeholder is not None:
                    insertion_context = meta.graph.inserting_after(last_placeholder)
                else:
                    insertion_context = meta.graph.inserting_before(None)
                with insertion_context:
                    context_nodes[meta.name] = meta.graph.placeholder(
                        name="_graphpatch_args", default_value=None
                    )
            return meta

        def wrap_graph_nodes(meta: Union[NodeMeta, GraphMeta]) -> Union[NodeMeta, GraphMeta]:
            if isinstance(meta, GraphMeta):
                self._wrap_graph(
                    meta,
                    cast(GraphModule, self._graph_module.get_submodule(meta.graph_module_name)),
                    context_nodes[meta.name],
                )
                # This is the node in the parent that calls this graph_module
                if meta.node is not None:
                    new_kwargs = {k: v for k, v in meta.node.kwargs.items()}
                    new_kwargs["_graphpatch_args"] = context_nodes[meta.parent]
                    meta.node.kwargs = new_kwargs
            # Need to pass patching arguments down to opaque modules at their real call sites.
            # TODO: should probably be an attribute on meta?
            elif meta.node.target == "_opaque_module_call":
                self._add_patch_args(meta, context_nodes[meta.parent])
            return meta

        def wrap_nodes(meta: Union[NodeMeta, GraphMeta]) -> Union[NodeMeta, GraphMeta]:
            if (
                meta.node is None
                or meta.node.op in ("placeholder", "output")
                or meta.is_graph
                or meta.hidden
            ):
                return meta
            self._replace_node(
                meta.node,
                cast(
                    GraphModule,
                    self._graph_module.get_submodule(
                        cast(GraphMeta, self._meta[meta.parent]).graph_module_name
                    ),
                ),
                context_nodes[meta.parent],
                meta.name,
            )
            return meta

        def recompile(meta: Union[NodeMeta, GraphMeta]) -> Union[NodeMeta, GraphMeta]:
            if isinstance(meta, GraphMeta):
                graph_module = cast(
                    GraphModule, self._graph_module.get_submodule(meta.graph_module_name)
                )
                graph_module.recompile()
            return meta

        self._meta.map_in_place(add_context)
        self._meta.map_in_place(wrap_graph_nodes)
        self._meta.map_in_place(wrap_nodes)
        self._meta.map_in_place(recompile)

    def _copy_transformers_generation_config(
        self,
        cls: Type[GenerationMixin],
        config: PretrainedConfig,
        generation_config: GenerationConfig,
    ) -> None:
        type(self).__bases__ = (
            PatchableGraph,
            cls,
        )
        self.config = deepcopy(config)
        self.generation_config = deepcopy(generation_config)

        # transformers uses inspect on this method, so we need to pretend to have the same signature
        @wraps(cls.prepare_inputs_for_generation)
        def override_generation_kwargs(*args: Any, **kwargs: Any) -> Any:
            kwargs.update(self._graphpatch_override_generation_kwargs)
            return cls.prepare_inputs_for_generation(self, *args, **kwargs)

        self.prepare_inputs_for_generation = override_generation_kwargs

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Workaround for Llama prepare_inputs_for_generation inspecting its own module hierarchy.
            # TODO: might want to flatten hierarchy by one level, i.e. not have a root _graph_module
            if "_modules" in self.__dict__ and name not in self._modules:
                if name in self._graph_module._modules:
                    return self._graph_module._modules[name]
            raise
