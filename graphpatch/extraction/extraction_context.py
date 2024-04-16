from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import LayerNorm, Module, ModuleDict, ModuleList


from .. import hacks
from ..optional.accelerate import ModelHook
from ..optional.bitsandbytes import Linear8bitLt
from .accelerate import detach_accelerate_hooks
from .graphpatch_module import GraphPatchModule
from .layer_norm_wrapper import LayerNormWrapper
from .wrapped_8_bit_linear import Wrapped8BitLinear


# Can't just make this a dataclass because for some reason torch.compile() chokes even if we're
# skipping/including in graph any function that calls __init__ on one of these.
class ModuleInvocation:
    args: List[Any]
    kwargs: Dict[str, Any]
    output: Any = None

    def __init__(
        self,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        output: Any = None,
    ):
        self.args = args
        self.kwargs = kwargs
        self.output = output


class ExtractionState:
    accelerate_hook: Optional[ModelHook]
    children: Dict[str, "ExtractionState"]
    extracted_module: Optional[GraphPatchModule]
    invocations: List[ModuleInvocation]
    name: str
    original_module: Module
    torch_name: str
    wrapped_module: Module

    def __init__(self, name: str, torch_name: str, original_module: Module):
        self.original_module = original_module
        # May get wrapped later when we enter extraction context.
        self.wrapped_module = original_module
        self.accelerate_hook = getattr(original_module, "_hf_hook", None)
        self.extracted_module = None
        self.invocations = []
        self.children = {}
        self.name = name
        self.torch_name = torch_name


@hacks.allow_in_graph
class ExtractionWrapper(Module):
    """Class in which to wrap all non-container modules during tracing and compilation. This serves
    several purposes:
        1. The allow_in_graph decorator prevents torch from inlining submodules so we can compile
        them separately.
        2. Lets us independently trace modules that occur multiple times in the module hierarchy
        under different names.
        3. Lets us work around the incompatibility between certain modules, such as bitsandbytes
        quantized linear and LayerNorm for earlier versions of torch, and the FakeTensors used
        during symbolic tracing.
    """

    _graphpatch_original_module_name: str
    _graphpatch_wrapped_module: Module
    _graphpatch_accelerate_hook: Optional[ModelHook]
    _graphpatch_extraction_state: ExtractionState
    _graphpatch_record_invocations: bool

    def __init__(self, state: ExtractionState, original_name: str, record_invocations: bool):
        super().__init__()
        self._graphpatch_extraction_state = state
        self._graphpatch_original_module_name = original_name
        self._graphpatch_accelerate_hook = state.accelerate_hook
        self._graphpatch_record_invocations = record_invocations
        wrapped_module = state.wrapped_module
        # TODO: abstract/make customizable
        if isinstance(wrapped_module, LayerNorm):
            wrapped_module = LayerNormWrapper(wrapped_module)
        elif isinstance(wrapped_module, Linear8bitLt):
            wrapped_module = Wrapped8BitLinear(wrapped_module)
        # Avoid adding the wrapped module to the module hierarchy.
        object.__setattr__(self, "_graphpatch_wrapped_module", wrapped_module)

    def __deepcopy__(self, memo=None):
        """compile() deep copies modules during symbolic tracing in order to fakify their
        parameters. Since some modules can't be deepcopied in fake mode (bitsandbytes), we need to
        customize our deepcopy to avoid copying the original module stored in our state. This class
        should never be seen by users, so this breaking expected deepcopy semantics won't cause any
        weirdness.
        """
        new_instance = self.__class__.__new__(self.__class__)
        Module.__init__(new_instance)
        new_instance._graphpatch_extraction_state = self._graphpatch_extraction_state
        new_instance._graphpatch_original_module_name = self._graphpatch_original_module_name
        new_instance._graphpatch_accelerate_hook = self._graphpatch_accelerate_hook
        new_instance._graphpatch_record_invocations = self._graphpatch_record_invocations
        with new_instance.substitute_submodules(self._graphpatch_wrapped_module):
            object.__setattr__(
                new_instance,
                "_graphpatch_wrapped_module",
                deepcopy(self._graphpatch_wrapped_module, memo),
            )
        new_instance._modules = deepcopy(self._modules, memo)
        return new_instance

    @contextmanager
    def substitute_submodules(self, target: Optional["ExtractionWrapper"] = None) -> Iterator[None]:
        if target is None:
            target = self._graphpatch_wrapped_module
        orig_modules = target._modules
        target._modules = self._modules
        try:
            yield
        finally:
            target._modules = orig_modules

    @hacks.disable
    def maybe_accelerate_pre_hook(self, *args, **kwargs):
        if self._graphpatch_accelerate_hook is not None:
            return self._graphpatch_accelerate_hook.pre_forward(self, *args, **kwargs)
        return args, kwargs

    @hacks.disable
    def maybe_accelerate_post_hook(self, output: Any):
        if self._graphpatch_accelerate_hook is not None:
            return self._graphpatch_accelerate_hook.post_forward(self, output)
        return output

    @hacks.skip
    def forward(self, *args, **kwargs):
        try:
            # Unfortunately we have to repeat substitute_submodules() functionality here, since
            # torch doesn't support contextmanagers even if we're skipping.
            orig_modules = self._graphpatch_wrapped_module._modules
            self._graphpatch_wrapped_module._modules = self._modules
            args, kwargs = self.maybe_accelerate_pre_hook(*args, **kwargs)
            output = self.maybe_accelerate_post_hook(
                self._graphpatch_wrapped_module(*args, **kwargs)
            )
            if not _in_compilation() and self._graphpatch_record_invocations:
                self._graphpatch_extraction_state.record_invocation(
                    ModuleInvocation(args, kwargs, output)
                )
            return output
        finally:
            self._graphpatch_wrapped_module._modules = orig_modules


def _iter_state_hierarchy(root_state: ExtractionState) -> Iterator[ExtractionState]:
    state_stack = [root_state]
    while state_stack:
        state = state_stack.pop()
        yield state
        state_stack.extend(state.children.values())


@contextmanager
def _wrap_module_hierarchy(root_state: ExtractionState, fn: Callable[[ExtractionState], Module]):
    original_modules: Dict[str, Module] = {}
    for state in _iter_state_hierarchy(root_state):
        original_modules[state.torch_name] = state.wrapped_module
        state.wrapped_module = fn(state)
    for state in _iter_state_hierarchy(root_state):
        for child_state in state.children.values():
            setattr(
                state.wrapped_module,
                child_state.torch_name.split(".")[-1],
                child_state.wrapped_module,
            )
    try:
        yield
    finally:
        for state in _iter_state_hierarchy(root_state):
            state.wrapped_module = original_modules[state.torch_name]
        for state in _iter_state_hierarchy(root_state):
            for child_state in state.children.values():
                setattr(
                    state.wrapped_module,
                    child_state.torch_name.split(".")[-1],
                    original_modules[child_state.torch_name],
                )


def _in_compilation() -> bool:
    return hacks._CURRENTLY_COMPILING


def _in_fake_mode() -> bool:
    return isinstance(torch.empty(0), FakeTensor)


@contextmanager
def _eval_mode(module: Module) -> Iterator[None]:
    """Set a module into eval mode, so we skip including training-only things like dropouts in
    our graph.
    """
    eval_state = module.training

    if eval_state:
        module.eval()

    yield

    if eval_state:
        module.train()


@contextmanager
def compilation_context(root_state: ExtractionState):
    original_root = root_state.wrapped_module
    try:
        with ExitStack() as context_stack:
            context_stack.enter_context(torch.inference_mode())
            context_stack.enter_context(_eval_mode(root_state.wrapped_module))
            context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
            context_stack.enter_context(hacks.allow_builtin_in_graph(root_state.wrapped_module))
            for submodule in root_state.wrapped_module.modules():
                context_stack.enter_context(detach_accelerate_hooks(submodule))
            torch._dynamo.reset()
            yield
    finally:
        root_state.wrapped_module = original_root


@contextmanager
def extraction_context(root_state: ExtractionState):
    def wrap_modules(state: ExtractionState):
        if isinstance(state.wrapped_module, (ModuleDict, ModuleList)):
            return state.wrapped_module
        if root_state.name == "":
            local_name = state.name
        else:
            # Remove parent's prefix
            local_name = state.name.replace(root_state.name + ".", "", 1)
        return ExtractionWrapper(state, local_name, root_state.name == "")

    with _wrap_module_hierarchy(root_state, wrap_modules):
        yield
