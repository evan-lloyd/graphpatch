graphpatch 0.1.0
================
.. py:currentmodule:: graphpatch

Overview
########
``graphpatch`` is a library for activation patching on :std:doc:`PyTorch <torch:index>` neural
network models. You use it by first wrapping your model in a :class:`PatchableGraph` and then running
operations in a context created by :meth:`PatchableGraph.patch`:

.. code::

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

In contrast to :ref:`other approaches <related_work>`, ``graphpatch`` can patch (or record) any
intermediate tensor value without manual modification of the underlying model's code. See :ref:`working_with_graphpatch` for
some tips on how to use the generated graphs.

Note that ``graphpatch`` activation patches are compatible with :std:doc:`AutoGrad <torch:autograd>`!
This means that, for example, you can perform optimizations over the ``value`` parameter to
:class:`AddPatch <patch.AddPatch>`:

.. code::

   delta = torch.zeros(size, requires_grad=True, device="cuda")
   optimizer = torch.optim.Adam([delta], lr=0.5)
   for _ in range(num_steps):
      with graph.patch({node_name: AddPatch(value=delta)):
         logits = graph(**prompt_inputs)
      loss = my_loss_function(logits)
      loss.backward()
      optimizer.step()

For a practical usage example, see the `demo <https://github.com/evan-lloyd/graphpatch/tree/main/demos/ROME>`_ of using ``graphpatch`` to replicate `ROME <https://rome.baulab.info/>`_.

Prerequisites
#############
The only mandatory requirements are ``torch>=2`` and ``numpy>=1.17``. Version 2+ of ``torch`` is required
because ``graphpatch`` leverages :func:`torch.compile`, which was introduced in ``2.0.0``, to extract computational graphs from models.
CUDA support is not required. ``numpy`` is required for full ``compile()`` support.

Python 3.8--3.11 are supported. Note that ``torch`` versions prior to ``2.1.0`` do not support compilation
on Python 3.11; you will get an exception when trying to use ``graphpatch`` with such a configuration.
No version of ``torch`` yet supports compilation on Python 3.12.

``graphpatch`` is theoretically compatible with any model in Huggingface's :std:doc:`transformers <transformers:index>`
library, but note that there may be edge cases in specific model code that it can't yet handle.
``graphpatch`` is tested against and known to work with the ``transformers`` implementations of
:std:doc:`Llama <transformers:model_doc/llama>` and :std:doc:`GPT2 <transformers:model_doc/gpt2>`.

``graphpatch`` is compatible with models loaded via :std:doc:`accelerate <accelerate:index>` and with 8-bit parameters
quantized by `bitsandbytes <https://pypi.org/project/bitsandbytes/>`_. This means that you can run ``graphpatch`` on
multiple GPU's and/or with quantized inference very easily on models provided by ``transformers``:

.. code::

   model = LlamaForCausalLM.from_pretrained(
      model_path,
      device_map="auto",
      quantization_config=BitsAndBytesConfig(load_in_8bit=True),
      torch_dtype=torch.float16,
   )
   pg = PatchableGraph(model, **example_inputs)


Installation
############
``graphpatch`` is available on PyPI, and can be installed via ``pip``:

.. code::

   pip install graphpatch

Note that you will likely want to do this in an environment that already has ``torch``, since ``pip`` may not resolve
``torch`` to a CUDA-enabled version by default. You don't need to do anything special to make ``graphpatch`` compatible
with ``transformers``, ``accelerate``, and ``bitsandbytes``; their presence is detected at run-time. However, for convenience,
you can install ``graphpatch`` with the "transformers" extra, which will install known compatible versions of these libraries along
with some of their optional dependencies that are otherwise mildly inconvenient to set up:

.. code::

   pip install graphpatch[transformers]

.. _related_work:

Alternatives
############
:meth:`Module hooks <torch.nn.Module.register_forward_hook>` are built in to ``torch`` and can be used for activation
patching. You can even add them to existing models without modifying their code. However, this will only give you
access to module inputs and outputs; accessing or patching intermediate values still requires a manual rewrite.

:std:doc:`TransformerLens <transformer_lens:index>` provides the
:class:`HookPoint <transformer_lens.hook_points.HookPoint>` class, which can record and patch intermediate
activations. However, this requires manually rewriting your model's code to wrap the values you want to make
patchable.

`TorchLens <https://github.com/johnmarktaylor91/torchlens>`_ records and outputs visualizations for every intermediate
activation. However, it is currently unable to perform any activation patching.


Documentation index
###################

.. toctree::
   :hidden:

   Overview <self>

.. toctree::
   :maxdepth: 3
   :titlesonly:

   api
   data_structures
   working_with_graphpatch
   notes_on_compilation

* :ref:`Full index <genindex>`
