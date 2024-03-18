from typing import Any

from torch.nn import ModuleList


class InvocationTrackingModuleList(ModuleList):
    _graphpatch_invocation_index: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._graphpatch_invocation_index = 0

    def forward(self, *args, **kwargs) -> Any:
        # TODO: surely any sane module will never vary how many times it calls its submodules
        # and the modulo doesn't matter? But we may want to make this configurable between
        # round-robin or throwing an exception, possibly a global "strict" mode?
        index = self._graphpatch_invocation_index % len(self._modules)
        self._graphpatch_invocation_index = index + 1
        return self[index](*args, **kwargs)
