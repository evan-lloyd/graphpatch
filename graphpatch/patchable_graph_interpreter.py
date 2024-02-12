from typing import Any, Dict, Tuple

from torch.fx import Interpreter
from torch.fx.node import Argument, Target

from .opaque_graph_module import opaque_module_call


class PatchableGraphInterpreter(Interpreter):
    def call_function(self, target: Target, args: Tuple[Argument], kwargs: Dict[str, Any]) -> Any:
        print("hrrm")
        if target is opaque_module_call:
            print("oh yeah")
            return self.module._original_module.module(*args, **kwargs)
        return super().call_function(target, args, kwargs)
