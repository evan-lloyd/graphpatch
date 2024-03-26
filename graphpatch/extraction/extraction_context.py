import inspect
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch._dynamo.allowed_functions import _allowed_function_ids
from torch._subclasses.fake_tensor import is_fake
from torch.nn import Module, ModuleDict, ModuleList, Sequential

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


class DeduplicationWrapper(Module):
    _graphpatch_original_module_name: str

    def __init__(self, wrapped: Module, original_name: str):
        super().__init__()
        self._graphpatch_original_module_name = original_name
        # Hide from the hierarchy in a tuple, since we want to monkeypatch in the wrapped _modules
        # during inference.
        self.wrapped = (wrapped,)

    @contextmanager
    def _substitute_modules(self) -> Iterator[None]:
        orig_modules = self.wrapped[0]._modules
        self.wrapped[0]._modules = self._modules
        try:
            yield
        finally:
            self.wrapped[0]._modules = orig_modules

    def forward(self, *args, **kwargs):
        with self._substitute_modules():
            return self.wrapped[0].forward(*args, **kwargs)


def _iter_state_hierarchy(root_state: ExtractionState) -> Iterator[ExtractionState]:
    state_stack = [root_state]
    while state_stack:
        state = state_stack.pop()
        yield state
        state_stack.extend(state.children.values())


@contextmanager
def wrap_module_hierarchy(root_state: ExtractionState, fn: Callable[[ExtractionState], Module]):
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
def tracer_hook(state: ExtractionState) -> Iterator[None]:
    def pre_hook(module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
        if state.accelerate_hook is not None:
            args, kwargs = state.accelerate_hook.pre_forward(module, *args, **kwargs)

        return (args, kwargs)

    def post_hook(
        module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> Any:
        if state.accelerate_hook is not None:
            output = state.accelerate_hook.post_forward(module, output)

        # Disregard the symbolic tracing step when recording invocations.
        if not is_fake(output):
            state.invocations.append(ModuleInvocation(args, kwargs, output))
        return output

    with state.wrapped_module.register_forward_pre_hook(
        pre_hook, with_kwargs=True
    ), state.wrapped_module.register_forward_hook(post_hook, with_kwargs=True):
        yield


@contextmanager
def eval_mode(module: Module) -> Iterator[None]:
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
def allow_modules(module_types: List[type]) -> Iterator[None]:
    """Use the undocumented _allowed_function_ids to prevent compile() from inlining the child
    modules, so we can independently compile them into separate GraphModules.
    """
    _orig_allowed_function_ids = deepcopy(_allowed_function_ids.function_ids)
    _orig_skipfiles_allowlist = deepcopy(torch._dynamo.skipfiles.FILENAME_ALLOWLIST)

    # Lets us compile even torch builtins.
    torch._dynamo.skipfiles.FILENAME_ALLOWLIST.add(
        getattr(inspect.getmodule(module_types[0]), "__file__", None)
    )

    _allowed_function_ids.function_ids.update(id(t) for t in module_types)

    try:
        yield
    finally:
        _allowed_function_ids.function_ids = _orig_allowed_function_ids
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST = _orig_skipfiles_allowlist


@contextmanager
def compilation_context(root_state: ExtractionState):
    with ExitStack() as context_stack:
        context_stack.enter_context(torch.inference_mode())
        context_stack.enter_context(eval_mode(root_state.wrapped_module))
        # module = context_stack.enter_context(wrap_bits_and_bytes(module))
        context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
        context_stack.enter_context(
            allow_modules([m.__class__ for m in root_state.wrapped_module.modules()])
        )
        for m in root_state.wrapped_module.modules():
            context_stack.enter_context(detach_accelerate_hooks(m))
        torch._dynamo.reset()
        yield


@contextmanager
def deduplication_context(root_state: ExtractionState):
    def deduplicate_modules(state: ExtractionState):
        if state is root_state or isinstance(state.wrapped_module, (ModuleDict, ModuleList)):
            return state.wrapped_module
        if root_state.torch_name == "":
            local_name = state.torch_name
        else:
            # Remove parent's prefix
            local_name = state.torch_name.replace(root_state.torch_name + ".", "", 1)
        return DeduplicationWrapper(state.wrapped_module, local_name)

    with wrap_module_hierarchy(root_state, deduplicate_modules):
        yield


@contextmanager
def root_context(extraction_state: Dict[str, ExtractionState]):
    with ExitStack() as context_stack:
        context_stack.enter_context(deduplication_context(extraction_state[""]))
        for name, state in extraction_state.items():
            # We don't want to add hooks to the root, as this would cause torch.compile() to include
            # our tracing code in the resulting graph.
            if name == "":
                continue
            context_stack.enter_context(tracer_hook(state))
        yield
