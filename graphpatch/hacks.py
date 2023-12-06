# mypy: ignore-errors

import operator
from contextlib import ExitStack, contextmanager
from functools import partial, partialmethod

import torch

TORCH_VERSION = tuple(int(v.split("+")[0]) for v in torch.__version__.split("."))


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
    from torch._dynamo.utils import get_fake_value
    from torch._dynamo.variables import builder
    from torch._dynamo.variables.builder import VariableBuilder
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    def wrap_literal(self, _original, value):
        # Avoids some additional cases of tensor sizes getting specialized.
        if type(value) is int and isinstance(self.get_source(), (LocalSource, NNModuleSource)):
            return self.wrap_unspecialized_primitive(value)
        return _original(self, value)

    def wrap_fx_proxy_cls(_original, target_cls, tx, *args, **kwargs):
        # Tweak some internal flags to avoid specializing on tensor dimensions.
        tx.fake_mode.shape_env.specialize_zero_one = False
        tx.fake_mode.shape_env.duck_shape = False

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
            fake_value = get_fake_value(proxy.node, tx)

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

    def _maybe_guard_eq(self, _original, *args, **kwargs):
        """This prevents many cases of torch deciding to specialize on tensor dimensions. We don't
        care about the guards that would have gotten generated because they aren't present in the
        compiled GraphModule, and we discard the OptimizedModule after compiling."""
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

    patch_map = {
        SubgraphTracer: [create_graph_input],
        OutputGraph: [remove_unused_graphargs],
        ShapeEnv: [_maybe_guard_eq, produce_guards],
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


@contextmanager
def monkeypatch_graph_names():
    """Monkeypatches OutputGraph.register_attr_or_module, which was the minimum intervention I
    found to reliably override the "self_" prefix that compile() adds to getattr and submodule node
    names in the compiled graph. We can likely do this in a less hacky way with a custom
    backend/tracer.
    """
    from torch._dynamo.output_graph import OutputGraph

    orig_method = OutputGraph.register_attr_or_module

    def strip_self_from_names(*args, **kwargs):
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
        return orig_method(self, target, *names, **kwargs)

    try:
        OutputGraph.register_attr_or_module = strip_self_from_names
        yield
    finally:
        OutputGraph.register_attr_or_module = orig_method


@contextmanager
def make_dynamo_ignore_hooks():
    """Only use for torch >= 2.1.0. In that version, torch attempts to compile any hooks you have
    applied to your model, but we need it to ignore them (as it did in 2.0.*) so we can easily
    record module inputs/outputs without affecting the compiled code.
    """
    from torch._dynamo import output_graph
    from torch._dynamo.utils import nnmodule_has_hooks as orig_function
    from torch._dynamo.variables import nn_module

    modules_to_patch = [nn_module, output_graph]

    def dummy_has_hooks(*args, **kwargs):
        return False

    try:
        for m in modules_to_patch:
            m.nnmodule_has_hooks = dummy_has_hooks
        yield
    finally:
        for m in modules_to_patch:
            m.nnmodule_has_hooks = orig_function


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


def dynamo_hacks_for_current_torch_version():
    hack_stack = ExitStack()
    if TORCH_VERSION >= (2, 1):
        hack_stack.enter_context(set_dynamo_config())
        hack_stack.enter_context(make_dynamo_ignore_hooks())
        hack_stack.enter_context(monkeypatch_dynamic_shapes())
    hack_stack.enter_context(monkeypatch_graph_names())
    return hack_stack
