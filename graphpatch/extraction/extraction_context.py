from contextlib import ExitStack, contextmanager
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import torch
from torch.nn import LayerNorm, Module, ModuleDict, ModuleList

from .. import hacks
from ..optional.accelerate import ModelHook, add_hook_to_module, remove_hook_from_module
from ..optional.bitsandbytes import Linear8bitLt
from ..optional.typing_extensions import TypeAlias
from .graphpatch_module import GraphPatchModule
from .wrapped_8_bit_linear import Wrapped8BitLinear
from .wrapped_layer_norm import WrappedLayerNorm

CONTAINER_TYPES = (ModuleList, ModuleDict)
WrappedModule: TypeAlias = Union["ExtractionWrapper", ModuleDict, ModuleList]


def init_container(container: Union[ModuleList, ModuleDict]):
    if isinstance(container, ModuleList):
        return container.__class__([Module() for _ in range(len(container))])
    return container.__class__()


def is_container(module: Union[Module, Type[Module]]) -> bool:
    """Strictly checking for built-in container types, since user-derived ones could have forward()."""
    if isinstance(module, type):
        model_class = module
    else:
        model_class = module.__class__
    return model_class in CONTAINER_TYPES


class ModuleInvocation:
    args: List[Any]
    kwargs: Dict[str, Any]
    output: Any

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
    wrapped_module: WrappedModule
    local_name: str

    def __init__(
        self,
        name: str,
        torch_name: str,
        original_module: Module,
        children: Dict[str, "ExtractionState"],
    ):
        self.original_module = original_module
        self.accelerate_hook = getattr(original_module, "_hf_hook", None)
        self.extracted_module = None
        self.invocations = []
        self.children = children
        self.name = name
        self.torch_name = torch_name
        if is_container(original_module):
            self.wrapped_module = init_container(original_module)
        else:
            self.wrapped_module = ExtractionWrapper(self)
        for sub_state in children.values():
            if name == "":
                local_name = sub_state.torch_name
            else:
                local_name = sub_state.torch_name[len(name) + 1 :]
            setattr(self.wrapped_module, local_name, sub_state.wrapped_module)


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

    _graphpatch_wrapped_module: Module
    _graphpatch_extraction_state: ExtractionState
    _graphpatch_record_invocations: bool
    _graphpatch_accelerate_hook: Optional[ModelHook]

    def __init__(self, state: ExtractionState):
        super().__init__()
        self._graphpatch_extraction_state = state
        self._graphpatch_record_invocations = True
        self._graphpatch_accelerate_hook = state.accelerate_hook
        wrapped_module = state.original_module
        # TODO: abstract/make customizable
        if isinstance(wrapped_module, LayerNorm):
            wrapped_module = WrappedLayerNorm(wrapped_module)
        elif isinstance(wrapped_module, Linear8bitLt):
            wrapped_module = Wrapped8BitLinear(wrapped_module)
        # Avoid adding the wrapped module to the module hierarchy.
        object.__setattr__(self, "_graphpatch_wrapped_module", wrapped_module)

    def __deepcopy__(self, memo=None):
        """compile() deep copies modules during symbolic tracing in order to fakify their
        parameters. We need to customize this behavior for the following reasons:
        1) Since some modules can't be deepcopied in fake mode (bitsandbytes), we need to avoid
           copying the original module stored in our state.
        2) If the user is using accelerate for CPU/disk offloading, we need to fakify the weights
           stored in the hook's weights_map.
        This class should never be seen by users, so this breaking expected deepcopy semantics won't
        cause any weirdness.
        """
        new_instance = self.__class__.__new__(self.__class__)
        Module.__init__(new_instance)
        new_instance._graphpatch_extraction_state = self._graphpatch_extraction_state
        new_instance._graphpatch_record_invocations = self._graphpatch_record_invocations
        new_instance._graphpatch_accelerate_hook = deepcopy(self._graphpatch_accelerate_hook)
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
            return self._graphpatch_accelerate_hook.pre_forward(
                self._graphpatch_wrapped_module, *args, **kwargs
            )
        return args, kwargs

    @hacks.disable
    def maybe_accelerate_post_hook(self, output: Any):
        if self._graphpatch_accelerate_hook is not None:
            return self._graphpatch_accelerate_hook.post_forward(
                self._graphpatch_wrapped_module, output
            )
        return output

    @hacks.skip
    def forward(self, *args, **kwargs):
        try:
            # Unfortunately we have to repeat substitute_submodules() functionality here, since
            # torch.compile() doesn't support contextmanagers even if we're skipping.
            orig_modules = self._graphpatch_wrapped_module._modules
            self._graphpatch_wrapped_module._modules = self._modules
            args, kwargs = self.maybe_accelerate_pre_hook(*args, **kwargs)
            output = self.maybe_accelerate_post_hook(
                self._graphpatch_wrapped_module(*args, **kwargs)
            )
            if not hacks.in_compilation() and self._graphpatch_record_invocations:
                self._graphpatch_extraction_state.invocations.append(
                    ModuleInvocation(args, kwargs, output)
                )
            return output
        finally:
            self._graphpatch_wrapped_module._modules = orig_modules


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


@contextmanager
def compilation_context(root_state: ExtractionState):
    with ExitStack() as context_stack:
        context_stack.enter_context(torch.inference_mode())
        context_stack.enter_context(_eval_mode(root_state.wrapped_module))
        context_stack.enter_context(hacks.dynamo_hacks_for_current_torch_version())
        context_stack.enter_context(
            hacks.allow_builtin_in_graph(root_state.wrapped_module._graphpatch_wrapped_module)
        )
        context_stack.enter_context(hacks.patch_module_module(ExtractionWrapper))
        for submodule in root_state.wrapped_module.modules():
            if isinstance(submodule, ExtractionWrapper):
                context_stack.enter_context(
                    detach_accelerate_hooks(submodule._graphpatch_wrapped_module)
                )
        torch._dynamo.reset()
        yield
