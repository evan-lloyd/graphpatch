.. py:currentmodule:: graphpatch

.. _notes_on_compilation:

Notes on compilation
====================
:class:`PatchableGraph` constructs a hierarchy of graphs matching the original submodule hierarchy.
Submodules will always be instances of either :class:`CompiledGraphModule`, :class:`OpaqueGraphModule`,
or container types such as :class:`torch.nn.ModuleDict` and :class:`torch.nn.ModuleList`. For example,
if you had a module ``foo`` containing a submodule ``bar``, you might get back a GraphModule equivalent
to ``foo`` which has a sub-GraphModule ``bar``, equivalent to the original ``bar``:

.. code::

    PatchableGraph(
        (_graph_module): CompiledGraphModule(
            (bar): OpaqueGraphModule()
        )
    )

Here, ``_graph_module`` is the :class:`compiled <CompiledGraphModule>` version of the original root
module (``foo``) and ``foo.bar`` is an :class:`opaque wrapper <OpaqueGraphModule>` around the original
``bar``.

``graphpatch``
makes a best effort to compile every submodule. When this succeeds, the corresponding submodule will
be of class :class:`CompiledGraphModule`. This includes some workarounds for cases where a native
:func:`torch.compile` would fail, such as for quantized linear modules from the ``bitsandbytes``
library, which ``accelerate`` uses for model quantization. When this fails, by default ``graphpatch``
will fall back to constructing a wrapper around the original model code that allows for patching
inputs, outputs, parameters, and buffers. The corresponding submodule will be of class
:class:`OpaqueGraphModule`.

As :func:`torch.compile` is still new and somewhat rough to use in practice, I have made the default
behavior to handle this fallback silently; this can be configured by passing :class:`ExtractionOptions`
to :class:`PatchableGraph`. For example, you can treat compilation failure as an error with the option
``error_on_compilation_failure``, which will give you access to the original exception. PyTorch does
not offer much guidance on diagnosing these errors, but you might start with
:std:doc:`torch:torch.compiler_troubleshooting`.

.. _custom_extraction_functions:

Custom extraction functions
***************************
As an advanced option, you can pass custom functions for handling the conversion of modules into
graphs with the extraction option ``custom_extraction_functions``, which is a dict mapping from
subtypes of :class:`torch.nn.Module` to functions taking a :class:`torch.nn.Module` and outputting
a :class:`torch.fx.Graph`. This may be easier in some cases than diagnosing issues with
:func:`torch.compile`, and gives fine-grained control over the generated graph. Example:

.. code::

    class MyUncompilableModule(Module):
        def forward(self, foo: Tensor, bar: Tensor):
            return uncompilable_operation(foo, bar)

    def extract_my_module(module: Module) -> Graph:
        graph = Graph()

        # Note that placeholders must exactly match the names of the arguments to forward.
        foo = graph.placeholder("foo")
        bar = graph.placeholder("bar")
        operation = graph.call_function(uncompilable_operation, (foo, bar))
        # graphpatch will respect the names of any nodes in the graph, which can make subsequent
        # patching operations easier to parse.
        operation.name = "my_custom_name"

        # Note that the output must be wrapped in a single-element tuple.
        graph.output((operation,))
        return graph

    pg = PatchableGraph(module_instance,
        ExtractionOptions(custom_extraction_functions={MyUncompilableModule: extract_my_module})),
        example_foo,
        example_bar,
    )

When using this option, make sure that your graph has placeholders with targets exactly matching
the names of the inputs to your module's ``forward()`` function. This is needed because ``graphpatch``
runs a sanity check on these inputs to correct them sometimes getting mangled by the normal compilation
process. Your graph's output must also be wrapped in a single-element tuple as in the above example
to match the behavior of :func:`torch.compile`.

For another example, ``graphpatch`` internally uses this mechanism to handle the ``bitsandbytes`` class
:class:`bitsandbytes.nn.Linear8bitLt` to allow patching of the weights as if they were an ordinary
tensor, with the following extraction function that simply manually constructs the desired
:class:`torch.fx.Graph`:

.. code::

    def compile_8_bit_linear(module):
        graph = Graph()
        x = graph.placeholder("x", torch.Tensor)
        cb = graph.get_attr("CB")
        scb = graph.get_attr("SCB")
        bias = graph.get_attr("bias")
        threshold = graph.get_attr("threshold")
        mul = graph.call_function(operator.mul, (cb, scb))
        weight = graph.call_function(operator.truediv, (mul, 127))
        weight.name = "weight"
        output = graph.call_function(matmul_8bit, (x, weight, bias, threshold))
        graph.output((output,))
        return graph

.. _multiple_invocations:

Multiple invocations of a submodule are treated independently
*************************************************************
While this may be a rare edge case in practice, ``graphpatch`` handles cases where a submodule is
called multiple times by treating each instance as an independent graph that can be patched
separately. For a (somewhat contrived) example from a model used in our test cases:

.. code::

    class TupleOutputModule(Module):
        _shape = (2, 3)

        def __init__(self):
            super().__init__()
            self.linear = Linear(*TupleOutputModule._shape)

        def forward(self, x):
            return (self.linear(x), self.linear(x + 1))

    >>> pg = PatchableGraph(TupleOutputModule(), torch.ones(3, 2))
    PatchableGraph(
        (_graph_module): CompiledGraphModule(
            (linear): MultiplyInvokedModule(
                (0-1): 2 x CompiledGraphModule()
            )
        )
    )
    >>> pg.graph
    <root>: CompiledGraphModule
    ├─x: Tensor(3, 2)
    ├─linear_0: CompiledGraphModule
    │ ├─input: Tensor(3, 2)
    │ ├─weight: Tensor(3, 2)
    │ ├─bias: Tensor(3)
    │ ├─linear: Tensor(3, 3)
    │ └─output: Tensor(3, 3)
    ├─add: Tensor(3, 2)
    ├─linear_1: CompiledGraphModule
    │ ├─input: Tensor(3, 2)
    │ ├─weight: Tensor(3, 2)
    │ ├─bias: Tensor(3)
    │ ├─linear: Tensor(3, 3)
    │ └─output: Tensor(3, 3)
    └─output: tuple(2)
    ├─sub_0: Tensor(3, 3)
    └─sub_1: Tensor(3, 3)

Note that ``linear`` is now an instance of :class:`MultiplyInvokedModule`, which is a subclass of
:class:`torch.nn.ModuleList`, and there are two nodes corresponding to it in the graph, ``linear_0``
and ``linear_1``. The two invocations can be patched independently:

.. code::

    with pg.patch(
        {
            "linear_0.output": [AddPatch(value=torch.ones((1,)))],
            "linear_1.output": [ZeroPatch()],
        }
    ):
        ...
