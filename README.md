# graphpatch 0.2.0

Documentation is hosted on [Read the Docs](https://graphpatch.readthedocs.io/en/stable).

## Overview

`graphpatch` is a library for [activation patching](https://graphpatch.readthedocs.io/en/stable/what_is_activation_patching.html#what-is-activation-patching) (often
also referred to as “ablation”) on [PyTorch](https://pytorch.org/docs/stable/index.html) neural network models. You use
it by first wrapping your model in a [`PatchableGraph`](https://graphpatch.readthedocs.io/en/stable/patchable_graph.html#graphpatch.PatchableGraph) and then running operations in a context
created by [`PatchableGraph.patch()`](https://graphpatch.readthedocs.io/en/stable/patchable_graph.html#graphpatch.PatchableGraph.patch):

```python
model = GPT2LMHeadModel.from_pretrained(
   "gpt2-xl",
   device_map="auto",
   quantization_config=BitsAndBytesConfig(load_in_8bit=True),
   torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
inputs = tokenizer(
   "The Eiffel Tower, located in", return_tensors="pt", padding=False
).to(torch.device("cuda"))
# Note that arguments after the first are forwarded as example inputs
# to the model during compilation.
pg = PatchableGraph(model, **inputs, use_cache=False)
# Applies patches to the multiplication result within the activation function of the
# MLP in the 18th transformer layer. ProbePatch records the last observed value at the
# given node, while ZeroPatch zeroes out the value seen by downstream computations.
with pg.patch("transformer.h_17.mlp.act.mul_3": [probe := ProbePatch(), ZeroPatch()]):
   output = pg(**inputs)
# Patches are applied in order. probe.activation holds the value prior
# to ZeroPatch zeroing it out.
print(probe.activation)
```

In contrast to [other approaches](#related-work), `graphpatch` can patch (or record) any
intermediate tensor value without manual modification of the underlying model’s code. See [Working with graphpatch](https://graphpatch.readthedocs.io/en/stable/working_with_graphpatch.html#working-with-graphpatch) for
some tips on how to use the generated graphs.

Note that `graphpatch` activation patches are compatible with [AutoGrad](https://pytorch.org/docs/stable/autograd.html)!
This means that, for example, you can perform optimizations over the `value` parameter to
[`AddPatch`](https://graphpatch.readthedocs.io/en/stable/patch.html#graphpatch.patch.AddPatch):

```python
delta = torch.zeros(size, requires_grad=True, device="cuda")
optimizer = torch.optim.Adam([delta], lr=0.5)
for _ in range(num_steps):
   with graph.patch({node_name: AddPatch(value=delta)):
      logits = graph(**prompt_inputs)
   loss = my_loss_function(logits)
   loss.backward()
   optimizer.step()
```

For a practical usage example, see the [demo](https://github.com/evan-lloyd/graphpatch/tree/main/demos/ROME) of using `graphpatch` to replicate [ROME](https://rome.baulab.info/).

## Prerequisites

The only mandatory requirements are `torch>=2` and `numpy>=1.17`. Version 2+ of `torch` is required
because `graphpatch` leverages [`torch.compile()`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), which was introduced in `2.0.0`, to extract computational graphs from models.
CUDA support is not required. `numpy` is required for full `compile()` support.

Python 3.8–3.11 are supported. Note that `torch` versions prior to `2.1.0` do not support compilation
on Python 3.11; you will get an exception when trying to use `graphpatch` with such a configuration.
No version of `torch` yet supports compilation on Python 3.12.

## Installation

`graphpatch` is available on PyPI, and can be installed via `pip`:

```default
pip install graphpatch
```

Note that you will likely want to do this in an environment that already has `torch`, since `pip` may not resolve
`torch` to a CUDA-enabled version by default. You don’t need to do anything special to make `graphpatch` compatible
with `transformers`, `accelerate`, and `bitsandbytes`; their presence is detected at run-time. However, for convenience,
you can install `graphpatch` with the “transformers” extra, which will install known compatible versions of these libraries along
with some of their optional dependencies that are otherwise mildly inconvenient to set up:

```default
pip install graphpatch[transformers]
```

## Model compatibility

For full functionality, `graphpatch` depends on being able to call [`torch.compile()`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) on your
model. This currently supports a subset of possible Python operations–for example, it doesn’t support
context managers. `graphpatch` implements some workarounds for situations that a native
`compile()` can’t handle, but this coverage isn’t complete. To deal with this, `graphpatch`
has a graceful fallback that should be no worse of a user experience than using module hooks.
In that case, you will only be able to patch an uncompilable submodule’s inputs, outputs,
parameters, and buffers. See [Notes on compilation](https://graphpatch.readthedocs.io/en/stable/notes_on_compilation.html#notes-on-compilation) for more discussion.

## `transformers` integration

`graphpatch` is theoretically compatible with any model in Huggingface’s [transformers](https://huggingface.co/docs/transformers/main/en/index)
library, but note that there may be edge cases in specific model code that it can’t yet handle. For
example, it is not (yet!) compatible with the key-value caching implementation, so if you want full
compilation of such models you should pass `use_cache=False` as part of the example inputs.

`graphpatch` is compatible with models loaded via [accelerate](https://huggingface.co/docs/accelerate/main/en/index) and with 8-bit parameters
quantized by [bitsandbytes](https://pypi.org/project/bitsandbytes/). This means that you can run `graphpatch` on
multiple GPU’s and/or with quantized inference very easily on models provided by `transformers`:

```python
model = LlamaForCausalLM.from_pretrained(
   model_path,
   device_map="auto",
   quantization_config=BitsAndBytesConfig(load_in_8bit=True),
   torch_dtype=torch.float16,
)
pg = PatchableGraph(model, **example_inputs, use_cache=False)
```

For `transformers` models supporting the [`GenerationMixin`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin) protocol, you will
also be able to use convenience functions like [`generate()`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) in
combination with activation patching:

```python
# Prevent Llama from outputting "Paris"
with pg.patch({"lm_head.output": ZeroPatch(slice=(slice(None), slice(None), 3681))}):
   output_tokens = pg.generate(**inputs, max_length=20, use_cache=False)
```

<a id="related-work"></a>

## Alternatives

[`Module hooks`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) are built in to `torch` and can be used for activation
patching. You can even add them to existing models without modifying their code. However, this will only give you
access to module inputs and outputs; accessing or patching intermediate values still requires a manual rewrite.

[TransformerLens](https://transformerlensorg.github.io/TransformerLens/index.html) provides the
[`HookPoint`](https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.hook_points.html#transformer_lens.hook_points.HookPoint) class, which can record and patch intermediate
activations. However, this requires manually rewriting your model’s code to wrap the values you want to make
patchable.

[TorchLens](https://github.com/johnmarktaylor91/torchlens) records and outputs visualizations for every intermediate
activation. However, it is currently unable to perform any activation patching.

## Documentation index

* [API](https://graphpatch.readthedocs.io/en/stable/api.html)
  * [ExtractionOptions](https://graphpatch.readthedocs.io/en/stable/extraction_options.html)
  * [Patch](https://graphpatch.readthedocs.io/en/stable/patch.html)
  * [PatchableGraph](https://graphpatch.readthedocs.io/en/stable/patchable_graph.html)
* [Data structures](https://graphpatch.readthedocs.io/en/stable/data_structures.html)
  * [CompiledGraphModule](https://graphpatch.readthedocs.io/en/stable/compiled_graph_module.html)
  * [MultiplyInvokedModule](https://graphpatch.readthedocs.io/en/stable/multiply_invoked_module.html)
  * [NodePath](https://graphpatch.readthedocs.io/en/stable/node_path.html)
  * [OpaqueGraphModule](https://graphpatch.readthedocs.io/en/stable/opaque_graph_module.html)
* [Notes on compilation](https://graphpatch.readthedocs.io/en/stable/notes_on_compilation.html)
* [What is activation patching?](https://graphpatch.readthedocs.io/en/stable/what_is_activation_patching.html)
* [Working with `graphpatch`](https://graphpatch.readthedocs.io/en/stable/working_with_graphpatch.html)

* [Full index](https://graphpatch.readthedocs.io/en/stable/genindex.html)
