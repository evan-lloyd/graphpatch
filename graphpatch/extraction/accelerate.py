from contextlib import contextmanager
from typing import Iterator, Optional

from torch.nn import Module

from ..optional.accelerate import ModelHook, add_hook_to_module, remove_hook_from_module


@contextmanager
def detach_accelerate_hooks(module: Module) -> Iterator[Optional[ModelHook]]:
    """Temporarily detach accelerate's hooks from the module, since they don't play nice with
    torch.compile(). Return the hook object so we can apply it to the compiled graph.
    """

    hook = getattr(module, "_hf_hook", None)
    if hook is not None:
        remove_hook_from_module(module)
        # Instance-level forward function doesn't play nice with torch.compile
        del module.forward
    try:
        yield hook
    finally:
        if hook is not None:
            add_hook_to_module(module, hook)
