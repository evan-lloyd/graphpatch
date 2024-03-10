import inspect
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch._dynamo.allowed_functions import (
    _allowed_function_ids,
    _disallowed_function_ids,
)
from torch._subclasses.fake_tensor import is_fake
from torch.nn import Module, ModuleDict, ModuleList, Sequential

from .. import hacks
from ..optional.accelerate import ModelHook
from .accelerate import detach_accelerate_hooks
from .bitsandbytes import wrap_bits_and_bytes
from .graphpatch_module import GraphPatchModule

CONTAINER_TYPES = (Sequential, ModuleList, ModuleDict)


@dataclass
class ModuleInvocation:
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None


class ExtractionState:
    original_module: Module
    extracted_module: Optional[GraphPatchModule]
    invocations: List[ModuleInvocation]
    children: Dict[str, "ExtractionState"]
    name: str

    def __init__(self, name: str, original_module: Module):
        self.original_module = original_module
        self.accelerate_hook = getattr(original_module, "_hf_hook", None)
        self.extracted_module = None
        self.invocations = []
        self.children = {}
        self.name = name


@contextmanager
def tracer_hook(
    module: Module, invocations: List[ModuleInvocation], accelerate_hook: Optional[ModelHook]
) -> Iterator[None]:
    def pre_hook(module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
        if accelerate_hook is not None:
            args, kwargs = accelerate_hook.pre_forward(module, *args, **kwargs)

        invocations.append(ModuleInvocation())
        cur_invocation = invocations[-1]
        cur_invocation.args = list(args)
        cur_invocation.kwargs = kwargs

        return (args, kwargs)

    def post_hook(
        module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> Any:
        if accelerate_hook is not None:
            output = accelerate_hook.post_forward(module, output)

        # Disregard the symbolic tracing step when recording invocations.
        if is_fake(output):
            invocations.pop()
        else:
            invocations[-1].output = output
        return output

    pre_handle = None
    post_handle = None
    try:
        pre_handle = module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        post_handle = module.register_forward_hook(post_hook, with_kwargs=True)
        yield
    finally:
        if pre_handle is not None:
            pre_handle.remove()
        if post_handle is not None:
            post_handle.remove()


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
def allow_modules(modules: List[type], module_class: type) -> Iterator[None]:
    """Use the undocumented _allowed_function_ids to prevent compile() from inlining the child
    modules, so we can independently compile them into separate GraphModules.
    """

    module_ids = list(map(id, modules))
    ids_to_remove = [id for id in module_ids if id not in _allowed_function_ids.function_ids]
    module_filename = getattr(inspect.getmodule(module_class), "__file__", None)
    remove_skipfile = None

    # Let us compile even torch builtins
    if module_filename and module_filename not in torch._dynamo.skipfiles.FILENAME_ALLOWLIST:
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.add(module_filename)
        remove_skipfile = module_filename
    remove_id = None
    if id(module_class) in _allowed_function_ids.function_ids:
        remove_id = id(module_class)
        _allowed_function_ids.remove(remove_id)
        _disallowed_function_ids.add(remove_id)

    for _id in module_ids:
        _allowed_function_ids.add(_id)
    try:
        yield
    finally:
        for _id in ids_to_remove:
            _allowed_function_ids.remove(_id)
        if remove_id is not None:
            _allowed_function_ids.add(remove_id)
            _disallowed_function_ids.remove(remove_id)
        if remove_skipfile is not None:
            torch._dynamo.skipfiles.FILENAME_ALLOWLIST.remove(remove_skipfile)


@contextmanager
def compilation_context(module: Module):
    with ExitStack() as context_stack:
        context_stack.enter_context(torch.inference_mode())
        context_stack.enter_context(eval_mode(module))
        module = context_stack.enter_context(wrap_bits_and_bytes(module))
        context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
        context_stack.enter_context(
            allow_modules(
                [m.__class__ for m in module.modules() if m != module],
                module_class=module.__class__,
            )
        )
        for m in module.modules():
            context_stack.enter_context(detach_accelerate_hooks(m))
        torch._dynamo.reset()
        yield module


@contextmanager
def root_context(root_module: Module, extraction_state: Dict[str, ExtractionState]):
    with ExitStack() as context_stack:
        for name, state in extraction_state.items():
            if name == "":
                continue
            context_stack.enter_context(
                tracer_hook(
                    # Root module may have been bits-and-bytes wrapped.
                    state.original_module if name != "" else root_module,
                    state.invocations,
                    state.accelerate_hook,
                )
            )
        yield
