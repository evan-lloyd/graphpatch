from typing import Callable, Optional, Set, Type

from torch.fx import GraphModule
from torch.nn import Module

from ..optional.dataclasses import dataclass, field


@dataclass(kw_only=True)
class ExtractionOptions:
    """Options to control the behavior of `graphpatch` during graph extraction.

    Attributes:
        classes_to_skip_compiling: Set of Module classes to leave uncompiled. These modules will
            only be patchable at their inputs and outputs. May be useful for working around
            compilation issues. Default: set().
        error_on_compilation_failure: Treat failure to compile a submodule as an error, rather than
            falling back to module-level patching. Default: True.
        postprocessing_function: Optional function to call which will modify the generated
            :class:`torch.fx.GraphModule`. This function can modify the underlying
            :class:`torch.fx.Graph` in-place. The original module is passed for reference in case,
            for example, the needed modifications depend on its configuration. Default: None.
        skip_compilation: Skip compilation on all modules. Only module inputs and outputs will be
            patchable. May be useful for faster iteration times if patching intermediate values
            isn't needed. Default: False.
        warn_on_compilation_failure: Issue a warning when compilation fails, but then fall back
            to module-level patching for the failed module(s). Default: False.
    """

    classes_to_skip_compiling: Set[Type[Module]] = field(default_factory=set)
    error_on_compilation_failure: bool = False
    postprocessing_function: Optional[Callable[[GraphModule, Module], None]] = None
    skip_compilation: bool = False
    warn_on_compilation_failure: bool = False
