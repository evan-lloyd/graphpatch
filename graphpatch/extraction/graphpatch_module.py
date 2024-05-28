from collections import OrderedDict, deque
from copy import copy, deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from torch.fx import Graph, GraphModule
from torch.fx.graph import PythonCode
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX

from ..optional.accelerate import ModelHook
from ..optional.typing_extensions import TypeGuard
from .multiply_invoked_module import MultiplyInvokedModule

# TODO: resolve circular import more cleanly
if TYPE_CHECKING:
    from ..meta import OutputArgumentIndex

_GRAPHPATCH_MODULE_SERIALIZATION_KEYS = (
    "_graphpatch_submodules",
    "_graphpatch_output_indexes",
    # _graphpatch_accelerate_hook handled specially, since it may contain the model's weights.
)


def _deepcopy_hook_without_weights_map(hook: ModelHook) -> ModelHook:
    NONEXISTENT = object()
    if hook is None:
        return None
    try:
        orig_weights_map = getattr(hook, "weights_map", NONEXISTENT)
        orig_tied_params_map = getattr(hook, "tied_params_map", NONEXISTENT)
        hook.weights_map = None
        hook.tied_params_map = None
        copied_hook = deepcopy(hook)
    finally:
        if orig_weights_map is not NONEXISTENT:
            hook.weights_map = orig_weights_map
        else:
            del hook.weight_map
        if orig_tied_params_map is not NONEXISTENT:
            hook.tied_params_map = orig_tied_params_map
        else:
            del hook.tied_params_map
        return copied_hook


class GraphPatchModule(GraphModule):
    _graphpatch_accelerate_hook: Optional[ModelHook]
    _graphpatch_submodules: Dict[
        str,
        Tuple[
            Union[Type[ModuleList], Type[ModuleDict], None],
            Tuple[str, ...],
        ],
    ]
    _graphpatch_output_indexes: "OutputArgumentIndex"

    def set_extra_state(self, state: Any) -> None:
        for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS:
            setattr(self, k, deepcopy(state[k]))
        self._graphpatch_accelerate_hook = _deepcopy_hook_without_weights_map(
            state["_graphpatch_accelerate_hook"]
        )
        if hasattr(state["_graphpatch_accelerate_hook"], "weights_map"):
            self._graphpatch_accelerate_hook.weights_map = copy(
                state["_graphpatch_accelerate_hook"].weights_map
            )
        if hasattr(state["_graphpatch_accelerate_hook"], "tied_params_map"):
            self._graphpatch_accelerate_hook.tied_params_map = copy(
                state["_graphpatch_accelerate_hook"].tied_params_map
            )

    def get_extra_state(self) -> Any:
        state = {k: deepcopy(getattr(self, k)) for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS}
        state["_graphpatch_accelerate_hook"] = _deepcopy_hook_without_weights_map(
            self._graphpatch_accelerate_hook
        )
        if hasattr(self._graphpatch_accelerate_hook, "weights_map"):
            state["_graphpatch_accelerate_hook"].weights_map = copy(
                self._graphpatch_accelerate_hook.weights_map  # type: ignore
            )
        if hasattr(self._graphpatch_accelerate_hook, "tied_params_map"):
            state["_graphpatch_accelerate_hook"].tied_params_map = copy(
                self._graphpatch_accelerate_hook.tied_params_map  # type: ignore
            )
        return state

    def __getitem__(self, index: Any) -> Optional[Module]:
        """Convenience to get access to submodules that would be inaccessible to ordinary attribute
        access due to names not being identifiers. Useful for Sequential, and possibly any
        user-defined container-like Modules.
        """
        if isinstance(index, int):
            return self._modules[str(index)]
        return self._modules[index]

    @staticmethod
    def _is_container(module: Optional[Module]) -> TypeGuard[Union[ModuleList, ModuleDict]]:
        return isinstance(module, (ModuleList, ModuleDict))

    @staticmethod
    def _container_passthrough(
        children: Dict[str, Optional[Module]]
    ) -> Iterator[Tuple[str, Optional[Module]]]:
        child_modules = deque(children.items())
        while child_modules:
            name, submodule = child_modules.popleft()
            yield name, submodule

            if GraphPatchModule._is_container(submodule):
                child_modules.extend([(f"{name}.{n}", m) for n, m in submodule._modules.items()])

    @staticmethod
    def _child_modules(children: Dict[str, Optional[Module]]) -> Iterator[Tuple[str, Module]]:
        for name, submodule in GraphPatchModule._container_passthrough(children):
            if not GraphPatchModule._is_container(submodule) and submodule is not None:
                yield name, submodule

    def _child_containers(self) -> Iterator[Tuple[str, Optional[Module]]]:
        for name, submodule in GraphPatchModule._container_passthrough(self._modules):
            if self._is_container(submodule):
                yield name, submodule

    def _set_submodules_for_serialization(self) -> None:
        self._graphpatch_submodules = {
            name: (  # type: ignore
                (
                    type(submodule)
                    if type(submodule) in (ModuleList, ModuleDict, MultiplyInvokedModule)
                    else None
                ),
                tuple(submodule._modules.keys()),
            )
            for name, submodule in self._container_passthrough(self._modules)
            if submodule is not None
        }

    def _init(
        self,
        root: Union[Module, Dict[str, Any]],
        accelerate_hook: Optional[ModelHook] = None,
    ) -> None:
        """Separating out actual initialization logic because __init__ will not get called for
        CompiledGraphModule.
        """
        # Deserializing from pickle.
        if isinstance(root, dict):
            self.set_extra_state(root[_EXTRA_STATE_KEY_SUFFIX])
            # Initialize containers, which aren't included in the state directly. Reversed is
            # important, since we need to add children to the state dict before processing their
            # parents.
            for name, (
                container_class,
                container_keys,
            ) in reversed(list(self._graphpatch_submodules.items())):
                if container_class in (ModuleList, MultiplyInvokedModule):
                    root[name] = container_class(root[f"{name}.{k}"] for k in container_keys)  # type: ignore
                elif container_class is ModuleDict:
                    root[name] = ModuleDict({k: root[f"{name}.{k}"] for k in container_keys})
            for name in self._graphpatch_submodules.keys():
                [*parent_path, local_name] = name.split(".")
                parent_module = self.get_submodule(".".join(parent_path))
                setattr(parent_module, local_name, root[name])
            # Fix ordering of _modules. GraphModule constructor will have set them based on
            # invocation order, not the original ordering.
            orig_modules = self._modules
            self._modules = OrderedDict()
            for name in self._graphpatch_submodules:
                if name in orig_modules:
                    self._modules[name] = orig_modules[name]
        elif isinstance(root, GraphPatchModule):
            self.set_extra_state(root.get_extra_state())
        else:
            self._graphpatch_accelerate_hook = accelerate_hook
            self._set_submodules_for_serialization()

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Optional[Graph],
        class_name: str,
        accelerate_hook: Optional[ModelHook] = None,
    ):
        super().__init__(root, graph, class_name)
        self._init(root, accelerate_hook)

    def get_submodule(self, target: str) -> "GraphPatchModule":
        return cast(GraphPatchModule, super().get_submodule(target))

    def named_children(self) -> Iterator[Tuple[str, "GraphPatchModule"]]:
        return cast(Iterator[Tuple[str, "GraphPatchModule"]], super().named_children())

    def modules(self) -> Iterator["GraphPatchModule"]:
        return cast(Iterator[GraphPatchModule], super().modules())

    def named_modules(
        self, memo: Optional[Set["Module"]] = None, prefix: str = "", remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, "GraphPatchModule"]]:
        return cast(
            Iterator[Tuple[str, GraphPatchModule]],
            super().named_modules(memo, prefix, remove_duplicate),
        )

    def recompile(self) -> PythonCode:
        """GraphModule recompile overwrites our forward method, so to add calls to accelerate's
        hooks, we wrap the modified function.
        """
        code: PythonCode = super().recompile()
        compiled_forward = type(self).forward

        def wrapped_forward(self: GraphPatchModule, *args: Any, **kwargs: Any) -> Any:
            if self._graphpatch_accelerate_hook is not None:
                args, kwargs = self._graphpatch_accelerate_hook.pre_forward(self, *args, **kwargs)

            output = compiled_forward(self, *args, **kwargs)

            if self._graphpatch_accelerate_hook is not None:
                output = self._graphpatch_accelerate_hook.post_forward(self, output)

            return output

        type(self).forward = wrapped_forward
        return code
