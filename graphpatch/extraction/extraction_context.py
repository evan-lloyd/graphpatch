import inspect
from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import is_fake
from torch.nn import Module, ModuleDict, ModuleList

from .. import hacks
from ..optional.accelerate import ModelHook
from .accelerate import detach_accelerate_hooks

# from .bitsandbytes import wrap_bits_and_bytes
from .graphpatch_module import GraphPatchModule


@dataclass
class ModuleInvocation:
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None


class ExtractionState:
    original_module: Module
    extracted_module: Optional[GraphPatchModule]
    wrapped_module: Module
    invocations: List[ModuleInvocation]
    children: Dict[str, "ExtractionState"]
    name: str
    torch_name: str

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


@allow_in_graph
class ExtractionWrapper(Module):
    """Class in which to wrap all non-container modules during tracing and compilation. This serves
    two purposes:
        1. The allow_in_graph decorator prevents torch from inlining submodules so we can compile
        them separately.
        2. Lets us independently trace modules that occur multiple times in the module hierarchy
        under different names.
    """

    _graphpatch_original_module_name: str
    _graphpatch_wrapped_module: Module

    def __init__(self, wrapped: Module, original_name: str):
        super().__init__()
        self._graphpatch_original_module_name = original_name
        # Avoid adding the wrapped module to the module hierarchy.
        object.__setattr__(self, "_graphpatch_wrapped_module", wrapped)

    @contextmanager
    def _substitute_modules(self) -> Iterator[None]:
        orig_modules = self._graphpatch_wrapped_module._modules
        self._graphpatch_wrapped_module._modules = self._modules
        try:
            yield
        finally:
            self._graphpatch_wrapped_module._modules = orig_modules

    def forward(self, *args, **kwargs):
        with self._substitute_modules():
            return self._graphpatch_wrapped_module.forward(*args, **kwargs)


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


@contextmanager
def _tracer_hook(state: ExtractionState) -> Iterator[None]:
    # It's possible that we're retracing after a compilation failure, so make sure we don't persist
    # any invocations from a previous run.
    state.invocations = []

    def pre_hook(module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
        if state.accelerate_hook is not None:
            args, kwargs = state.accelerate_hook.pre_forward(module, *args, **kwargs)

        return (args, kwargs)

    def post_hook(
        module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> Any:
        if state.accelerate_hook is not None:
            output = state.accelerate_hook.post_forward(module, output)

        # Disregard the symbolic tracing step when recording invocations. We can tell if we're
        # tracing if we're in fake mode, checked by creating a tensor and seeing if it's fake.
        if not is_fake(torch.zeros(1)):
            state.invocations.append(ModuleInvocation(args, kwargs, output))
        return output

    with state.wrapped_module.register_forward_pre_hook(
        pre_hook, with_kwargs=True
    ), state.wrapped_module.register_forward_hook(post_hook, with_kwargs=True):
        yield


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
def _allow_compilation(module: Module) -> Iterator[None]:
    """Allows us to compile torch builtins."""
    _orig_skipfiles_allowlist = deepcopy(torch._dynamo.skipfiles.FILENAME_ALLOWLIST)
    torch._dynamo.skipfiles.FILENAME_ALLOWLIST.add(
        getattr(inspect.getmodule(module.__class__), "__file__", None)
    )
    try:
        yield
    finally:
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST = _orig_skipfiles_allowlist


@contextmanager
def compilation_context(root_state: ExtractionState):
    with ExitStack() as context_stack:
        context_stack.enter_context(torch.inference_mode())
        context_stack.enter_context(_eval_mode(root_state.wrapped_module))
        # module = context_stack.enter_context(wrap_bits_and_bytes(module))
        context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
        context_stack.enter_context(_allow_compilation(root_state.wrapped_module))
        for m in root_state.wrapped_module.modules():
            context_stack.enter_context(detach_accelerate_hooks(m))
        torch._dynamo.reset()
        yield


@contextmanager
def _tracing_context(root_state: ExtractionState):
    with ExitStack() as context_stack:
        # We don't want to add hooks to the root, as this would cause torch.compile() to include
        # our tracing code in the resulting graph.
        state_stack = list(root_state.children.values())
        while state_stack:
            state = state_stack.pop()
            context_stack.enter_context(_tracer_hook(state))
            state_stack.extend(state.children.values())
        yield


@contextmanager
def extraction_context(root_state: ExtractionState):
    def wrap_modules(state: ExtractionState):
        if state is root_state or isinstance(state.wrapped_module, (ModuleDict, ModuleList)):
            return state.wrapped_module
        if root_state.name == "":
            local_name = state.name
        else:
            # Remove parent's prefix
            local_name = state.name.replace(root_state.name + ".", "", 1)
        return ExtractionWrapper(state.wrapped_module, local_name)

    if root_state.name == "":
        maybe_tracing_context = _tracing_context
    else:
        maybe_tracing_context = nullcontext
    with _wrap_module_hierarchy(root_state, wrap_modules), maybe_tracing_context(root_state):
        yield
