.. py:currentmodule:: graphpatch

.. _notes_on_compilation:

Notes on compilation
====================
``graphpatch`` makes a best effort to compile every submodule when making a module patchable. This
includes some workarounds for cases where a native :func:`torch.compile` would fail, such as for
quantized linear modules from the ``bitsandbytes`` library, which ``accelerate`` uses for model
quantization. When this fails, by default ``graphpatch`` will fall back to constructing a wrapper
around the original model code that allows for patching inputs, outputs, parameters, and buffers.
As :func:`torch.compile` is still new and somewhat rough to use in practice, I have made the default
behavior to handle this fallback silently; this can be configured by passing :class:`ExtractionOptions`
to :class:`PatchableGraph`.
Note that the original module hierarchy is retained. For example, if you had a module ``foo``
containing a submodule ``bar``, you would get back a GraphModule equivalent to ``foo`` which has
a sub-GraphModule ``bar``, equivalent to the original ``bar``:

.. code::

    PatchableGraph(
        (_graph_module): CompiledGraphModule(
            (bar): OpaqueGraphModule()
        )
    )

Here, ``_graph_module`` is the compiled version of the original root module (``foo``) and the
original hierarchy has been retained.
