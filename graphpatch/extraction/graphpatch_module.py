from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union

from torch import Tensor
from torch.fx import Graph, GraphModule
from torch.nn import Module, ModuleDict, ModuleList, Parameter
from torch.nn.modules.module import _EXTRA_STATE_KEY_SUFFIX

from .invocation_tracking_module_list import InvocationTrackingModuleList

# TODO: resolve circular import more cleanly
if TYPE_CHECKING:
    from ..meta import OutputArgumentIndex

_GRAPHPATCH_MODULE_SERIALIZATION_KEYS = (
    "_graphpatch_module_containers",
    "_graphpatch_output_indexes",
)


class GraphPatchModule(GraphModule):
    _graphpatch_module_containers: Dict[
        str, Tuple[Union[Type[ModuleList], Type[ModuleDict]], Tuple[str]]
    ]
    _graphpatch_output_indexes: "OutputArgumentIndex"

    def set_extra_state(self, state: Any):
        for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS:
            setattr(self, k, deepcopy(state[k]))

    def get_extra_state(self) -> Any:
        state = {k: getattr(self, k) for k in _GRAPHPATCH_MODULE_SERIALIZATION_KEYS}
        return deepcopy(state)

    def _init_graphpatch_attributes(self):
        # Due to torch.compile() not accepting our CompiledGraphModule class, our actual __init__
        # method gets bypassed, so create a hook here to call in the compilation callback.
        self._graphpatch_module_containers = {}

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: Graph,
        class_name: str,
    ):
        self._init_graphpatch_attributes()

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
