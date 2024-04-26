# mypy: ignore-errors

import inspect
import operator
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from functools import partial, partialmethod

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from .optional.accelerate import AVAILABLE as ACCELERATE_AVAILABLE

TORCH_VERSION = tuple(int(v.split("+")[0]) for v in torch.__version__.split("."))

if TORCH_VERSION < (2, 1):
    from torch._dynamo import allow_in_graph, disable, skip  # noqa: F401
else:
    from torch._dynamo.decorators import allow_in_graph, disable, skip  # noqa: F401

_CURRENTLY_COMPILING = False


def in_compilation():
    return _CURRENTLY_COMPILING


def in_fake_mode():
    return isinstance(torch.empty(0), FakeTensor)


def fix_gpt2_bool_buffers(model):
    # Accelerate seems to convert bool buffers to float16 where it really shouldn't
    for name, buffer in model.named_buffers():
        if "attn.bias" not in name:
            continue
        split_name = name.split(".")
        module = model.get_submodule(".".join(split_name[:-1]))
        setattr(module, split_name[-1], buffer.to(torch.bool))


def clean_up_rotary_embedding(module, op):
    # TODO: figure out a more principled way to do this; particularly, why compile()
    # doesn't want to read seq_len from kwargs and instead bakes it in as a constant. Might
    # be specifically related to how slice operations are handled.
    getitem_node = next(
        n
        for n in module.graph.nodes
        if n.target == operator.getitem and n.args[0].name == f"{op}_cached"
    )
    seq_len_node = next(
        n for n in module.graph.nodes if n.op == "placeholder" and n.name == "seq_len"
    )
    with module.graph.inserting_after(seq_len_node):
        slice_node = module.graph.call_function(slice, (None, seq_len_node, None))
    getitem_node.args = getitem_node.args[0:1] + (
        getitem_node.args[1][:2] + (slice_node,) + getitem_node.args[1][3:],
    )


def patch_llama(graph_module, _original_module):
    """At the moment, LlamaRotaryEmbedding doesn't quite compile properly. It is supposed to read
    in the length of the current token sequence, but for some reason compile() converts the argument
    into a constant. This leads to run-time errors when trying to run the compiled model with
    different-length inputs.
    """
    # Improved dynamic shapes in 2.1 on means we don't have to do this.
    if TORCH_VERSION >= (2, 1):
        return
    for name, module in graph_module.named_modules():
        if "rotary_emb" in name:
            clean_up_rotary_embedding(module, "cos")
            clean_up_rotary_embedding(module, "sin")
            module.recompile()


@contextmanager
def monkeypatch_dynamic_shapes():
    """For torch >= 2.1.0. This version improves dynamic shapes in a way that's problematic for our
    use case; the philosophy seems to be one of eager optimization, with run-time checks that
    trigger a re-compile in case the optimistic assumptions are violated. Our current design doesn't
    work with that, since we assume a static GraphModule. In the future, we may want to refactor to
    roll with these re-compilations, in which case we can probably do away with this hack.
    """
    from torch._dynamo.output_graph import OutputGraph, SubgraphTracer
    from torch._dynamo.source import (
        DefaultsSource,
        LocalSource,
        NNModuleSource,
        TensorProperty,
        TensorPropertySource,
    )
    from torch._dynamo.variables import builder
    from torch._dynamo.variables.builder import VariableBuilder
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    def wrap_literal(self, _original, value):
        # Avoids some additional cases of tensor sizes getting specialized.
        if type(value) is int and isinstance(self.get_source(), (LocalSource, NNModuleSource)):
            return self.wrap_unspecialized_primitive(value)
        return _original(self, value)

    def wrap_fx_proxy_cls(_original, target_cls, tx, *args, **kwargs):
        # Tensor sizes of 1 keep getting specialized via some mysterious codepath. We can prevent
        # this by overriding the outputs of any nodes retrieving tensor sizes with only symbolic
        # sizes.
        proxy = kwargs.get("proxy")
        if (
            proxy is not None
            and proxy.node
            and proxy.node.op == "call_method"
            and proxy.node.target == "size"
        ):
            fake_value = builder.get_fake_value(proxy.node, tx)

            def enforce_symbolic(t):
                i, v = t
                if isinstance(v, int):
                    source = TensorPropertySource(
                        LocalSource(local_name=proxy.node.name), TensorProperty.SIZE, i
                    )
                    return tx.fake_mode.shape_env.create_symintnode(
                        tx.fake_mode.shape_env.create_symbol(
                            v,
                            source,
                            dynamic_dim=DimDynamic.DYNAMIC,
                        ),
                        hint=v,
                        source=source,
                    )
                return v

            if isinstance(fake_value, torch.Size):
                fake_value = torch.Size(map(enforce_symbolic, enumerate(fake_value)))
                kwargs["example_value"] = fake_value

        return _original(target_cls, tx, *args, **kwargs)

    def produce_guards(self, _original, *args, **kwargs):
        return []

    def evaluate_expr(
        self, _original, orig_expr, hint=None, fx_node=None, expect_rational=True, **kwargs
    ):
        # We care not for ShapeEnv's assertions and guards; just assume that the original module
        # is using correct shapes!
        import sympy

        if hint is None:
            concrete_val = self.size_hint(orig_expr)
        else:
            concrete_val = sympy.sympify(hint)
        return concrete_val

    def _maybe_guard_eq(self, _original, *args, **kwargs):
        """This prevents many cases of torch deciding to specialize on tensor dimensions. We don't
        care about the guards that would have gotten generated because they aren't present in the
        compiled GraphModule, and we discard the OptimizedModule after compiling."""
        return

    def _maybe_guard_rel(self, _original, *args, **kwags):
        """Same as _maybe_guard_eq, renamed in torch 2.3."""
        return

    def remove_unused_graphargs(self, _original):
        """Remove the dynamic size placeholders, since we monkeypatched away any logic that uses
        them.
        """
        # NB: *not* calling original; we only want to remove shape guard SymInts, since we
        # monkeypatched installing the related guards. Also cached default values.
        for node in self.placeholders:
            arg = node.meta["grapharg"]
            if isinstance(arg.source, (TensorPropertySource, DefaultsSource)):
                self.remove_node(node)

    def create_graph_input(self, _original, name, *args, **kwargs):
        """Unmangles input names, which end up with something hideous like L__foo_"""
        source = kwargs.get("source")
        if source is not None:
            if isinstance(source, NNModuleSource) and name.startswith("L_self_"):
                name = name.replace("L_self_", "", 1)
            elif hasattr(source, "local_name"):
                name = source.local_name
        return _original(self, name, *args, **kwargs)

    def __init__(self, _original, *args, **kwargs):
        _original(self, *args, **kwargs)
        # Tweak some internal flags to avoid specializing on tensor dimensions.
        self.specialize_zero_one = False
        self.duck_shape = False
        self.val_to_var = {}

    patch_map = {
        SubgraphTracer: [create_graph_input],
        OutputGraph: [remove_unused_graphargs],
        ShapeEnv: [
            _maybe_guard_eq if TORCH_VERSION < (2, 3) else _maybe_guard_rel,
            produce_guards,
            __init__,
            evaluate_expr,
        ],
        builder: [wrap_fx_proxy_cls],
        VariableBuilder: [wrap_literal],
    }
    orig_functions = {
        patched_obj: {a.__name__: getattr(patched_obj, a.__name__) for a in attrs}
        for patched_obj, attrs in patch_map.items()
    }

    try:
        for patched_obj, attr_map in orig_functions.items():
            for attr, fn in attr_map.items():
                if isinstance(patched_obj, type):
                    setattr(patched_obj, attr, partialmethod(locals()[attr], fn))
                else:
                    setattr(patched_obj, attr, partial(locals()[attr], fn))
        yield
    finally:
        for patched_obj, attr_map in orig_functions.items():
            for attr, fn in attr_map.items():
                setattr(patched_obj, attr, fn)


_RESERVED_NAMES = frozenset(
    {
        # GraphModule internals
        "_code",
        "code",
        "_graph",
        "graph",
        "_tracer_cls",
        "_tracer_extras",
        "meta",
        # NodePath reserved
        "_code",
        "_shape",
    }
)


def override_reserved_name(name):
    """Override names that would collide with GraphModule or GraphPatch internals, or would
    be otherwise unsuitable. I'd rather use some kind of wrapper to avoid the need to mess with
    names entirely, but I couldn't find a way to do so without adding a ton of complexity.
    """

    def override(part):
        if part in _RESERVED_NAMES:
            return f"{part}_"
        elif part.startswith("_graphpatch_"):
            return f"_{part}"
        return part

    return ".".join(map(override, name.split(".")))


@contextmanager
def monkeypatch_graph_names():
    """Monkeypatches OutputGraph.register_attr_or_module, which was the minimum intervention I
    found to reliably override the "self_" prefix that compile() adds to getattr and submodule node
    names in the compiled graph. We can likely do this in a less hacky way with a custom
    backend/tracer.
    """
    from torch._dynamo.output_graph import OutputGraph

    orig_register_attr = OutputGraph.register_attr_or_module

    def demangle_names(*args, **kwargs):
        [self, target, *names] = args

        def replace(name):
            if not isinstance(name, str):
                return name
            if name.startswith("self."):
                name = name.replace("self.", "", 1)
            if name.startswith("L['self']."):
                name = name.replace("L['self'].", "", 1)
            return name

        names = list(map(replace, names))
        return orig_register_attr(self, target, *names, **kwargs)

    try:
        OutputGraph.register_attr_or_module = demangle_names
        yield
    finally:
        OutputGraph.register_attr_or_module = orig_register_attr


def get_size(target, index):
    return target[index]


def maybe_replace_dynamo_get_item_lambda(node):
    """In compiling GPT2-XL, torch.compile() leaves a call to a local function which makes it
    unpicklable. All it does is retrieve the size of the given torch.Size at the given index:
    https://github.com/pytorch/pytorch/blob/e9ebda29d87ce0916ab08c06ab26fd3766a870e5/torch/_dynamo/variables/lists.py#L391
    We can just do that in a function accessible from global scope, and become picklable again.
    """
    if getattr(node.target, "__name__", None) == "_dynamo_get_item_lambda":
        node.target = get_size
        if node.meta:
            node.meta["source_fn"] = get_size


@contextmanager
def set_dynamo_config():
    """Reconfigure some dynamo options for >= 2.1.0 to get compilations we can work with."""
    config_values = {
        "specialize_int": False,
        "assume_static_by_default": False,
        "automatic_dynamic_shapes": False,
        "capture_scalar_outputs": True,
        "capture_dynamic_output_shape_ops": True,
    }
    orig_values = {key: getattr(torch._dynamo.config, key) for key in config_values}
    for key, value in config_values.items():
        setattr(torch._dynamo.config, key, value)
    try:
        yield
    finally:
        for key, value in orig_values.items():
            setattr(torch._dynamo.config, key, value)


@contextmanager
def allow_builtin_in_graph(module):
    # Same functionality, different names.
    if TORCH_VERSION >= (2, 3):
        allowlist_name = "LEGACY_MOD_INLINELIST"
        allowlist_value = inspect.getmodule(module.__class__).__name__
        skip_module = torch._dynamo.trace_rules
    elif TORCH_VERSION >= (2, 2):
        allowlist_name = "LEGACY_MOD_INLINELIST"
        allowlist_value = inspect.getmodule(module.__class__).__name__
        skip_module = torch._dynamo.skipfiles
    else:
        allowlist_name = "FILENAME_ALLOWLIST"
        allowlist_value = getattr(inspect.getmodule(module.__class__), "__file__", None)
        skip_module = torch._dynamo.skipfiles

    if TORCH_VERSION >= (2, 2):
        # Reset the LRU cache, or our changes will have no effect.
        skip_module.get_legacy_mod_inlinelist.cache_clear()

    allow_list = getattr(skip_module, allowlist_name)
    orig_allow_list = deepcopy(allow_list)
    allow_list.add(allowlist_value)
    try:
        yield
    finally:
        setattr(skip_module, allowlist_name, orig_allow_list)
        # Make sure our patch had no side effect.
        if TORCH_VERSION >= (2, 2):
            skip_module.get_legacy_mod_inlinelist.cache_clear()


@contextmanager
def patch_module_module(cls):
    """Needed for torch >= 2.3, which started disallowing the @disable decorator within our
    ExtractionWrapper class. This lets us hit get inside the "if" here:
    https://github.com/pytorch/pytorch/blob/71d020262793542974cf13b30f2a9099773f015c/torch/_dynamo/variables/functions.py#L326-L334

    Note that we have to undo this change or that leads to problems with pickling later.
    """
    orig_module = cls.__module__
    cls.__module__ = "torch.nn.graphpatch"
    try:
        yield
    finally:
        cls.__module__ = orig_module


@contextmanager
def monkeypatch_accelerate():
    if not ACCELERATE_AVAILABLE:
        yield
        return
    from accelerate import hooks

    orig = hooks.set_module_tensor_to_device

    def set_module_tensor_to_device(module, *accelerate_args, **accelerate_kwargs):
        if not in_fake_mode():
            return orig(module, *accelerate_args, **accelerate_kwargs)
        # This is a workaround for an exception that gets raised by this function when Torch is
        # in fake mode, as happens during compilation. AFAICT it is using the type of the original
        # object to distinguish between buffers and parameters, but because at this point the value
        # will in fact be a FakeTensor, it calls that constructor instead, with arguments that
        # are incompatible with the FakeTensor constructor.
        orig_new = FakeTensor.__new__

        def wrapped_fake_tensor_new(cls, *args, **kwargs):
            # Intercept call with incorrect args, which is just attempting to make a new copy of
            # args[0]. Since that will already have been fakified, we can just return it.
            if not isinstance(args[0], FakeTensorMode):
                return args[0]
            # Another case, we try to fakify an already fake tensor; just return it.
            elif isinstance(args[1], FakeTensor):
                return args[1]

            return orig_new(cls, *args, **kwargs)

        FakeTensor.__new__ = wrapped_fake_tensor_new
        try:
            return orig(module, *accelerate_args, **accelerate_kwargs)
        finally:
            FakeTensor.__new__ = orig_new

    hooks.set_module_tensor_to_device = set_module_tensor_to_device
    try:
        yield
    finally:
        hooks.set_module_tensor_to_device = orig


@contextmanager
def allow_inlining_skipped_functions():
    """Apparently this wasn't supposed to work like this, but we need the un-"fixed" behavior for
    ExtractionWrapper to work properly: https://github.com/pytorch/pytorch/pull/98862
    Work around by catching the added exception and proceeding as if nothing happened.
    """
    from torch._dynamo.exc import Unsupported
    from torch._dynamo import trace_rules
    from torch._dynamo.symbolic_convert import InliningInstructionTranslator

    def check_inlineable(func):
        try:
            return orig_check(func)
        except Exception as e:
            if isinstance(e, Unsupported) and e.msg.startswith(
                "call torch._dynamo.disable() wrapped function"
            ):
                return trace_rules.check_verbose(func, is_inlined_call=True)
            raise

    orig_check = InliningInstructionTranslator.check_inlineable
    InliningInstructionTranslator.check_inlineable = check_inlineable

    try:
        yield
    finally:
        InliningInstructionTranslator.check_inlineable = orig_check


@contextmanager
def dynamo_hacks_for_current_torch_version():
    with ExitStack() as hack_stack:
        if TORCH_VERSION >= (2, 1):
            hack_stack.enter_context(set_dynamo_config())
            hack_stack.enter_context(monkeypatch_dynamic_shapes())
        if TORCH_VERSION >= (2, 3):
            hack_stack.enter_context(allow_inlining_skipped_functions())
        hack_stack.enter_context(monkeypatch_accelerate())
        hack_stack.enter_context(monkeypatch_graph_names())
        yield


def replace_node_keeping_original_name(node, replacement, name):
    """Wrapper around fx.Node.replace_all_uses_with that ensures the replacement node has the same
    name as the original. Has to poke some FX internals to work, hence its presence in hacks.
    """
    node.replace_all_uses_with(replacement, propagate_meta=True)
    node.users = {}
    node.graph.erase_node(node)
    # A little low-level for my liking, but this lets us keep the same name for the replaced
    # node (erase_node doesn't clean up the namespace)
    del node.graph._graph_namespace._obj_to_name[node]
    node.graph._graph_namespace._obj_to_name[replacement] = name
    replacement.name = name
    node.graph._insert(replacement)
