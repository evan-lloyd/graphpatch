from typing import Callable, Dict, Optional, Set, Type

from torch.fx import GraphModule
from torch.fx.graph import Graph
from torch.nn import Module

from ..optional.dataclasses import dataclass, field


@dataclass(kw_only=True)
class ExtractionOptions:
    """Options to control the behavior of ``graphpatch`` during graph extraction. This is a
    keyword-only dataclass; to construct one, pass any number of options from the below.

    Attributes:
        classes_to_skip_compiling: Set of Module classes to leave uncompiled. These modules will
            only be patchable at their inputs, outputs, parameters, and buffers. May be useful for
            working around compilation issues. Default: ``set()``.
        copy_transformers_generation_config: If the wrapped Module is a huggingface transformers
            implementation, should graphpatch attempt to copy its generation config so generation
            convenience functions like ``generate()`` can be used? Default: ``True``.
        custom_extraction_functions: Optional map from Module classes to callables generating
            :class:`torch.fx.Graph` to be used in place of graphpatch's normal extraction mechanism
            when encountering that class. Advanced feature; should not be necessary for ordinary
            use. See :ref:`custom_extraction_functions`. Default: ``dict()``.
        error_on_compilation_failure: Treat failure to compile a submodule as an error, rather than
            falling back to module-level patching via :class:`OpaqueGraphModule`. Default: ``False``.
        postprocessing_function: Optional function to call which will modify the generated
            :class:`torch.fx.GraphModule`. This function can modify the underlying
            :class:`torch.fx.Graph` in-place. The original module is passed for reference in case,
            for example, the needed modifications depend on its configuration. Advanced feature;
            should not be necessary for ordinary use. Default: ``None``.
        skip_compilation: Skip compilation on all modules. Only module inputs and outputs will be
            patchable. May be useful for faster iteration times if patching intermediate values
            isn't needed. Default: ``False``.
        warn_on_compilation_failure: Issue a warning when compilation fails, but then fall back
            to module-level patching for the failed module(s). Default: ``False``.

    Example:
        .. code::

            options = ExtractionOptions(
                classes_to_skip_compiling={MyUncompilableModule},
                error_on_compilation_failure=True,
            )
            pg = PatchableGraph(my_model, options, **example_inputs)
    """

    classes_to_skip_compiling: Set[Type[Module]] = field(default_factory=set)
    copy_transformers_generation_config: bool = True
    custom_extraction_functions: Dict[Type[Module], Callable[[Module], Graph]] = field(
        default_factory=dict
    )
    error_on_compilation_failure: bool = False
    postprocessing_function: Optional[Callable[[GraphModule, Module], None]] = None
    skip_compilation: bool = False
    warn_on_compilation_failure: bool = False
