# graphpatch

`graphpatch` is a library for activation patching on [PyTorch](https://pytorch.org/docs/stable/index.html)
neural network models. You use it by first wrapping your model in a `PatchableGraph` and then running
operations in a context created by
`PatchableGraph.patch()`:

```
model = GPT2LMHeadModel.from_pretrained(
   "gpt2-xl",
   device_map="auto",
   load_in_8bit=True,
   torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
inputs = tokenizer(
   "The Eiffel Tower, located in", return_tensors="pt", padding=False
).to(torch.device("cuda"))
# Note that all arguments after the first are forwarded as example inputs
# to the model during compilation; use_cache and return_dict are arguments
# to GPT2LMHeadModel, not graphpatch-specific.
pg = PatchableGraph(model, **inputs, use_cache=False, return_dict=False)
# Applies two patches to the multiplication result within the activation function
# of the MLP in the 18th transformer layer. ProbePatch records the last observed value
# at the given node, while ZeroPatch zeroes out the value seen by downstream computations.
with pg.patch("transformer.h_17.mlp.act.mul_3": [probe := ProbePatch(), ZeroPatch()]):
   output = pg(**inputs)
# Patches are applied in order. probe.activation holds the value prior
# to ZeroPatch zeroing it out.
print(probe.activation)
```

`graphpatch` can patch (or record) any intermediate Tensor value without manual modification of the
underlying model’s code. See full documentation [here](https://graphpatch.readthedocs.io/en/latest/).

# Requirements
`graphpatch` requires `torch>=2` as it uses [`torch.compile()`](https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile) to build the
computational graph it uses for activation patching. As of `torch>=2.1.0`,
Python 3.8&ndash;3.11 are supported. `torch==2.0.*` do not support compilation on Python 3.11; you
will get an exception if you try to use `graphpatch` on such a configuration.

`graphpatch` automatically supports models loaded with features supplied by `accelerate` and
`bitsandbytes`. For example, you can easily use `graphpatch` on multiple GPU's and with quantized
inference:

```
model = LlamaForCausalLM.from_pretrained(
   model_path, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16
)
pg = PatchableGraph(model, **example_inputs)
```

# Installation
`graphpatch` is available on [PyPI](https://pypi.org/project/graphpatch), and can be installed via `pip`:
```
pip install graphpatch
```

Optionally, you can install `graphpatch` with the "transformers" extra to select known compatible versions of `transformers`, `accelerate`, `bitsandbytes`, and miscellaneous optional requirements of these packages to quickly get started with multi-GPU and quantized inference on real-world models:
```
pip install graphpatch[transformers]
```

# Demos
See the [demos](https://github.com/evan-lloyd/graphpatch/tree/main/demos) for some practical usage examples.

# Documentation
See the full documentation on [Read the Docs](https://graphpatch.readthedocs.io/en/latest/).
