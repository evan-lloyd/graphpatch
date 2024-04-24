from collections import deque
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple, Type, Union

from torch.fx import Graph, GraphModule
from torch.nn import Module, ModuleDict, ModuleList
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX

from ..optional.accelerate import ModelHook
from .invocation_tracking_module_list import InvocationTrackingModuleList

# TODO: resolve circular import more cleanly
if TYPE_CHECKING:
    from ..meta import OutputArgumentIndex

_GRAPHPATCH_MODULE_SERIALIZATION_KEYS = (
    "_graphpatch_accelerate_hook",
    "_graphpatch_module_containers",
    "_graphpatch_output_indexes",
)


class GraphPatchModule(GraphModule):
    _graphpatch_module_containers: Dict[
        str, Tuple[Union[Type[ModuleList], Type[ModuleDict]], Tuple[str]]
    ]
    _graphpatch_output_indexes: "OutputArgumentIndex"
    _graphpatch_accelerate_hook: Optional[ModelHook]

    def set_extra_state(self, state: Any):
        for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS:
            setattr(self, k, deepcopy(state[k]))

    def get_extra_state(self) -> Any:
        state = {k: getattr(self, k) for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS}
        return deepcopy(state)

    def __getitem__(self, index: Any) -> Module:
        """Convenience to get access to submodules that would be inaccessible to ordinary attribute
        access due to names not being identifiers. Useful for Sequential, and possibly any
        user-defined container-like Modules.
        """
        if isinstance(index, int):
            return self._modules[str(index)]
        return self._modules[index]

    @staticmethod
    def _is_container(module: Module):
        return isinstance(module, (ModuleList, ModuleDict)) and not isinstance(
            module, InvocationTrackingModuleList
        )

    @staticmethod
    def _container_passthrough(
        children: Iterator[Tuple[str, Module]]
    ) -> Iterator[Tuple[str, Module]]:
        child_modules = deque(children)
        while child_modules:
            name, submodule = child_modules.popleft()
            yield name, submodule

            if GraphPatchModule._is_container(submodule):
                child_modules.extend([(f"{name}.{n}", m) for n, m in submodule._modules.items()])

    @staticmethod
    def _child_modules(children: Iterator[Tuple[str, Module]]) -> Iterator[Tuple[str, Module]]:
        for name, submodule in GraphPatchModule._container_passthrough(children):
            if not GraphPatchModule._is_container(submodule):
                yield name, submodule

    def _child_containers(self) -> Iterator[Tuple[str, Module]]:
        for name, submodule in GraphPatchModule._container_passthrough(self._modules.items()):
            if self._is_container(submodule):
                yield name, submodule

    def _set_containers_for_serialization(self):
        self._graphpatch_module_containers = {
            name: (submodule.__class__, tuple(submodule._modules.keys()))
            for name, submodule in self._child_containers()
        }

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Graph,
        class_name: str,
        accelerate_hook: Optional[ModelHook] = None,
    ):
        super().__init__(root, graph, class_name)

        # Deserializing from pickle.
        if isinstance(root, dict):
            self.set_extra_state(root[_EXTRA_STATE_KEY_SUFFIX])
            # Initialize containers, which aren't included in the state directly.
            for name, (
                submodule_class,
                container_keys,
            ) in reversed(self._graphpatch_module_containers.items()):
                if submodule_class in (ModuleList, InvocationTrackingModuleList):
                    root[name] = submodule_class(root[f"{name}.{k}"] for k in container_keys)
                else:
                    root[name] = submodule_class({k: root[f"{name}.{k}"] for k in container_keys})
            for name, _ in self._graphpatch_module_containers.items():
                [*parent_path, local_name] = name.split(".")
                parent_module = self.get_submodule(".".join(parent_path))
                setattr(parent_module, local_name, root[name])
        elif isinstance(root, GraphPatchModule):
            self.set_extra_state(root.get_extra_state())
        else:
            self._graphpatch_accelerate_hook = accelerate_hook
            self._set_containers_for_serialization()

    def recompile(self):
        """GraphModule recompile overwrites our forward method, so to add calls to accelerate's
        hooks, we wrap the modified function.
        """
        code = super().recompile()
        compiled_forward = type(self).forward

        def wrapped_forward(self, *args, **kwargs):
            if self._graphpatch_accelerate_hook is not None:
                args, kwargs = self._graphpatch_accelerate_hook.pre_forward(self, *args, **kwargs)

            output = compiled_forward(self, *args, **kwargs)

            if self._graphpatch_accelerate_hook is not None:
                output = self._graphpatch_accelerate_hook.post_forward(self, output)

            return output

        type(self).forward = wrapped_forward
        return code
