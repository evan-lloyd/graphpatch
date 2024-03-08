import inspect
import re
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch
from torch._dynamo.allowed_functions import (
    _allowed_function_ids,
    _disallowed_function_ids,
)
from torch.nn import Module, ModuleDict, ModuleList, Sequential

from .. import hacks
from ..optional.accelerate import ModelHook
from .accelerate import detach_accelerate_hooks
from .bitsandbytes import wrap_bits_and_bytes

CONTAINER_TYPES = (Sequential, ModuleList, ModuleDict)


@contextmanager
def tracer_hook(
    module: Module, arg_tracker, accelerate_hook: Optional[ModelHook]
) -> Iterator[None]:
    from .graph_extraction import ModuleInvocation

    # compile() calls each module twice, but the first pass has FakeTensors, which we don't want to
    # trace. We need real example inputs for the recursive compile() calls, which fortunately get
    # passed in the second call.

    def pre_hook(module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
        arg_tracker.invocations.append(ModuleInvocation())
        cur_invocation = arg_tracker.invocations[-1]
        if accelerate_hook is not None:
            args, kwargs = accelerate_hook.pre_forward(module, *args, **kwargs)

        cur_invocation.args = list(args)
        cur_invocation.kwargs = kwargs

        return (args, kwargs)

    def post_hook(
        module: Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any
    ) -> Any:
        if accelerate_hook is not None:
            output = accelerate_hook.post_forward(module, output)
        cur_invocation = arg_tracker.invocations[-1]
        cur_invocation.output = output
        # Mark any containers in the output as "seen" so we can short-circuit output shape tracking
        # appropriately in the parent of this module.
        sub_output_stack = [output]
        while sub_output_stack:
            cur_output = sub_output_stack.pop()
            if isinstance(cur_output, (tuple, list)):
                arg_tracker.seen_outputs.add(id(cur_output))
                sub_output_stack.extend(cur_output)
            elif isinstance(cur_output, dict):
                arg_tracker.seen_outputs.add(id(cur_output))
                sub_output_stack.extend(cur_output.values())
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


class CompilationContext:
    original_module: Module
    output: Any
    hooks: Dict[str, ModelHook]
    _context_stack: ExitStack

    def __init__(self, original_module: Module):
        self.output = None
        self.original_module = original_module
        self.module = original_module
        self.hooks = {}
        self.accelerate_hook = None
        self._context_stack = ExitStack()

    def __enter__(self):
        self._context_stack = ExitStack()
        self._context_stack.enter_context(torch.inference_mode())
        self._context_stack.enter_context(eval_mode(self.original_module))
        self.module = self._context_stack.enter_context(wrap_bits_and_bytes(self.original_module))
        self._context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
        self._context_stack.enter_context(
            allow_modules(
                [m.__class__ for m in self.module.modules() if m != self.module],
                module_class=self.module.__class__,
            )
        )
        for name, submodule in self.module.named_modules():
            name = re.sub(r"\.(\d+)", lambda match: f"_{match.group(1)}", name)
            self.hooks[name] = self._context_stack.enter_context(detach_accelerate_hooks(submodule))
            if submodule is self.module:
                self.accelerate_hook = self.hooks[name]
        torch._dynamo.reset()
        return self

    def __exit__(self, *args):
        return self._context_stack.__exit__(*args)


class RootContext:

    def __init__(self, module, compilation_state):
        self._context_stack = ExitStack()
        self.module = module
        self.compilation_state = compilation_state

    def __enter__(self):
        from .graph_extraction import CompilationState

        self._context_stack = ExitStack()
        submodule_stack = [(None, None) + t for t in self.module.named_children()]
        while submodule_stack:
            container_name, container_index, name, submodule = submodule_stack.pop()

            original_name = name

            # Need to mirror torch.compile() behavior, which adds this prefix in this situation.
            if not name[0].isalpha():
                name = "sub" + name

            # TODO: handle nested container modules gracefully
            # Bit of a quirk with compile(); the named_modules() iterator returns names like
            # module_list.0.foo, but the resulting modules will be named like module_list_0.foo
            name = re.sub(r"\.(\d+)", lambda match: f"_{match.group(1)}", name)

            [*parent_name, local_name] = name.split(".")
            parent_name = "_".join(parent_name)
            state_name = name.replace(".", "_")
            submodule_stack.extend(
                [
                    (
                        name if is_container(submodule) else None,
                        n if is_container(submodule) else None,
                        f"{name}.{n}",
                        child,
                    )
                    for n, child in submodule.named_children()
                ]
            )
            self.compilation_state.child_state[state_name] = CompilationState(
                original_module=submodule,
                local_name=local_name,
                parent_name=parent_name,
                original_name=original_name,
                container_name=container_name,
                container_index=container_index,
            )
            self._context_stack.enter_context(
                tracer_hook(
                    submodule,
                    self.compilation_state.child_state[state_name].self_args,
                    self.hooks[name],
                )
            )

        return self

    def __exit__(self, *args, **kwargs):
        return self._context_stack.__exit__(*args, **kwargs)
