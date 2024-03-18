from contextlib import contextmanager
from typing import Any, Iterator

from torch import Tensor, float16
from torch.nn import Module, Parameter
from torch.nn.functional import linear

from ..optional.accelerate import add_hook_to_module
from ..optional.bitsandbytes import Linear8bitLt


class Wrapped8BitLinear(Module):
    def __init__(self, CB: Tensor, SCB: Tensor, bias: Tensor):
        super().__init__()

        if CB is None or SCB is None:
            raise ValueError("Must have CB/SCB values to wrap")

        self.CB = Parameter(CB, requires_grad=False)
        self.SCB = Parameter(SCB.to(float16).unsqueeze(1))
        if bias is not None:
            self.bias = Parameter(bias.to(float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Any:
        upcast_weights = (self.CB * self.SCB) / 127
        return linear(x, upcast_weights, self.bias)


def _wrapped_8bit(module: Linear8bitLt) -> Wrapped8BitLinear:
    # TODO: there's a gotcha where if you happened to run any inference on a base module, CB and SCB
    # will have been deleted; any way around this?
    wrapped = Wrapped8BitLinear(module.weight.CB, module.weight.SCB, module.bias)
    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        add_hook_to_module(wrapped, hook)
    return wrapped


@contextmanager
def wrap_bits_and_bytes(state: Module) -> Iterator[Module]:
    """Wrap any bitsandbytes quantized linear modules, since they use Tensor subclasses which are
    incompatible with the current (2.1.0) torch.compile() implementation.
    """
    original_submodules = {}
    module = state.wrapped_module
    original_module = state.wrapped_module
    for name, submodule in original_module.named_modules():
        if submodule is module and isinstance(module, Linear8bitLt):
            module = _wrapped_8bit(module)
            state.wrapped_module = module
            continue
        path = name.split(".")
        parent = module.get_submodule(".".join(path[:-1]))
        if isinstance(submodule, Linear8bitLt):
            original_submodules[name] = submodule
            setattr(parent, path[-1], _wrapped_8bit(submodule))
    try:
        yield
    finally:
        for name, original in original_submodules.items():
            path = name.split(".")
            parent = module.get_submodule(".".join(path[:-1]))
            setattr(parent, path[-1], original)
