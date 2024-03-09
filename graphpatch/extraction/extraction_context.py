import inspect
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type, Union

import torch
from torch._dynamo.allowed_functions import (
    _allowed_function_ids,
    _disallowed_function_ids,
)
from torch.nn import Module, ModuleDict, ModuleList, Sequential

from .. import hacks
from ..optional.accelerate import ModelHook
from .bitsandbytes import wrap_bits_and_bytes
from .graphpatch_module import GraphPatchModule

CONTAINER_TYPES = (Sequential, ModuleList, ModuleDict)


def children_with_container_passthrough(
    children: Iterator[Tuple[str, Module]], separator: str = "."
) -> Iterator[Tuple[str, Module]]:
    child_modules = [(t[0],) + (t[0],) + (t[1],) for t in children]
    while child_modules:
        torch_qual_name, unrolled_qual_name, submodule = child_modules.pop()
        if isinstance(submodule, (ModuleList, Sequential, ModuleDict)):
            child_modules.extend(
                reversed(
                    [
                        (f"{torch_qual_name}.{n}", f"{torch_qual_name}_{n}", m)
                        for n, m in submodule.named_children()
                    ]
                )
            )
        else:
            yield torch_qual_name, unrolled_qual_name, submodule


@dataclass
class ModuleInvocation:
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    output: Any = None


@dataclass
class SubmoduleInfo:
    parent_qual_name: str
    qual_name: str
    local_name: str
    invocations: List[ModuleInvocation]


class ExtractionState:
    original_module: Module
    extracted_module: Optional[GraphPatchModule]
    children: List[SubmoduleInfo]
    invocations: List[ModuleInvocation]

    def __init__(self, original_module: Module):
        self.original_module = original_module
        self.accelerate_hook = getattr(original_module, "_hf_hook", None)
        self.extracted_module = None
        self.children = []
        self.invocations = []


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
        cur_invocation = invocations[-1]
        cur_invocation.output = output
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


def is_container(module: Union[Module, Type[Module]]) -> bool:
    if isinstance(module, type):
        model_class = module
    else:
        model_class = module.__class__
    return model_class in CONTAINER_TYPES


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
        torch._dynamo.reset()
        yield module


@contextmanager
def root_context(extraction_state: Dict[str, ExtractionState]):
    with ExitStack() as context_stack:
        for name, state in extraction_state.items():
            context_stack.enter_context(
                tracer_hook(
                    state.original_module,
                    state.invocations,
                    state.accelerate_hook,
                )
            )
        yield
    # with ExitStack() as context_stack:
    #     submodule_stack = [(None, None) + t for t in module.named_children()]
    #     while submodule_stack:
    #         container_name, container_index, name, submodule = submodule_stack.pop()

    #         original_name = name

    #         # Need to mirror torch.compile() behavior, which adds this prefix in this situation.
    #         if not name[0].isalpha():
    #             name = "sub" + name

    #         # TODO: handle nested container modules gracefully
    #         # Bit of a quirk with compile(); the named_modules() iterator returns names like
    #         # module_list.0.foo, but the resulting modules will be named like module_list_0.foo
    #         name = re.sub(r"\.(\d+)", lambda match: f"_{match.group(1)}", name)

    #         [*parent_name, local_name] = name.split(".")
    #         parent_name = "_".join(parent_name)
    #         state_name = name.replace(".", "_")
    #         submodule_stack.extend(
    #             [
    #                 (
    #                     name if is_container(submodule) else None,
    #                     n if is_container(submodule) else None,
    #                     f"{original_name}.{n}",
    #                     child,
    #                 )
    #                 for n, child in submodule.named_children()
    #             ]
    #         )
    #         # self.compilation_state.child_state[state_name] = CompilationState(
    #         #     original_module=submodule,
    #         #     local_name=local_name,
    #         #     parent_name=parent_name,
    #         #     original_name=original_name,
    #         #     container_name=container_name,
    #         #     container_index=container_index,
    #         # )
    #         context_stack.enter_context(
    #             tracer_hook(
    #                 submodule,
    #                 extraction_state[state_name].invocations,
    #                 extraction_state[state_name].accelerate_hook,
    #             )
    #         )
    #         yield
