.. py:currentmodule:: graphpatch

.. _working_with_graphpatch:

Working with ``graphpatch``
===========================

``graphpatch`` is based on compiling PyTorch models into :class:`Graphs <torch.fx.Graph>`, exposing
all intermediate Tensor operations. This process is recursive; every submodel is compiled into a subgraph
within the final structure. Each intermediate operation is given a canonical name based on its
position within the overall graph. We call such a name a **NodePath** because it identifies a path
from the root of the graph through intermediate subgraphs. For example, a Tensor addition performed
within a submodule named ``"foo"`` might be named ``"foo.add"``. Or for a real-world example used
in the `ROME demo <https://github.com/evan-lloyd/graphpatch/tree/main/demos/ROME>`_,

.. code::

    "model.layers_8.mlp.down_proj.linear"

selects the node named ``"linear"`` within the compiled graph of the ``"down_proj"`` submodule of
the ``"mlp"`` submodule of the 9th layer of :std:doc:`Llama <transformers:model_doc/llama>`.

.. _node_path:

Inspecting the graph structure
******************************
Because the graph compilation is automatic, the names that are generated are not always intuitive. To
make it easier to locate a specific operation within the Graph, :class:`PatchableGraph` exposes a
property :attr:`graph <PatchableGraph.graph>` that has various functionality to inspect the generated
code and structure.

In IPython or Jupyter, you can tab-complete attributes on this object to select among the possible
next nodes in the path. For example, if ``pg`` is a PatchableGraph wrapping your model:

.. ipython::
   :verbatim:

   In [1]: pg.graph.<TAB>

reveals all of the children of the root node in the graph. This works recursively; you can tab-complete
attributes until reaching a leaf in the graph, at which point no completions will appear. You can
also visualize the hierarchy rooted at the current node path by pressing enter. For example, for one
of our models used in testing:

.. ipython::
    :verbatim:

    In [1]: tuple_pg.graph
    Out[1]:
    <root>: Graph(5)
    ├─x: Tensor(3, 2)
    ├─linear: Graph(5)
    │ ├─input: Tensor(3, 2)
    │ ├─weight: Tensor(3, 2)
    │ ├─bias: Tensor(3)
    │ ├─linear: Tensor(3, 3)
    │ └─output: Tensor(3, 3)
    ├─add: Tensor(3, 2)
    ├─linear_1: Graph(5)
    │ ├─input: Tensor(3, 2)
    │ ├─weight: Tensor(3, 2)
    │ ├─bias: Tensor(3)
    │ ├─linear: Tensor(3, 3)
    │ └─output: Tensor(3, 3)
    └─output: tuple(2)
      ├─sub_0: Tensor(3, 3)
      └─sub_1: Tensor(3, 3)

    In [2]: tuple_pg.graph.linear
    Out[2]:
    linear: Graph(5)
    ├─input: Tensor(3, 2)
    ├─weight: Tensor(3, 2)
    ├─bias: Tensor(3)
    ├─linear: Tensor(3, 3)
    └─output: Tensor(3, 3)

Reviewing compiled code
***********************
For many simple cases, such as module inputs and outputs, the generated node names will be
intuitive. However, for intermediate operations, it may be non-obvious what is actually happening
at a given node. For example, what is going on with ``tuple_pg.graph.add`` in the example above? To
help understand the compiled graphs, each node in ``graph`` also exposes an attribute
named ``_code``. On subgraphs (or the root), this reveals the code that ``torch.compile()``
generated:

.. ipython::
    :verbatim:

    In [2]: pg.graph._code
    Out[2]:
    def forward(self, x : torch.Tensor):
        linear = getattr(self.linear, "0")(x)
        add = x + 1;  x = None
        linear_1 = getattr(self.linear, "1")(add);  add = None
        return (linear, linear_1)

Most ``compile()``-generated code has this structure, where each line consists of value assignments to
variables with the same names as nodes in the graph. In this example, we can see that ``add`` is
getting assigned to the module input plus a constant.

To further track down the context of a given operation, you can also inspect the ``_code`` of leaf nodes.
This reveals the partial stack trace that ``torch.compile()`` maintained for us as it was compiling
the original model code:

.. ipython::
    :verbatim:

    In [3]: pg.graph.add._code
    Out[3]:
    File "/Users/evanlloyd/graphpatch/tests/fixtures/tuple_output_module.py", line 16, in forward
        return (self.linear(x), self.linear(x + 1))

For submodule calls, ``_code`` reveals both the compiled submodule code and the context from
the original model:

.. ipython::
    :verbatim:

    In [5]: pg.graph.linear._code
    Out[5]:
    Calling context:
    File "/Users/evanlloyd/graphpatch/tests/fixtures/tuple_output_module.py", line 16, in forward
        return (self.linear(x), self.linear(x + 1))
    Compiled code:
    def forward(self, input : torch.Tensor):
        input_1 = input
        weight = self.weight
        bias = self.bias
        linear = torch._C._nn.linear(input_1, weight, bias);  input_1 = weight = bias = None
        return linear

Inspecting node shapes
**********************
When constructing activation patches, it can be useful to know what shape is expected for a Tensor
at the target node. You may have noticed in the examples above that ``graph``'s REPL representation
lists shape information next to each node. To get programmatic access to this information as a
``torch.Size`` value, you can use the ``_shape`` attribute on any node:

.. ipython::
    :verbatim:

    In [7]: pg.graph.linear.input._shape
    Out[7]: torch.Size([3, 2])

Note that the listed shapes are those that were observed when running a forward pass on the model
with the example inputs you passed to the ``PatchableGraph`` constructor. This shape may have depended
on contingent factors of the example inputs, such as the batch dimension or number of tokens for a
specific input. You will have to determine whether this is the case based on knowledge of the
underlying model.

NodePath strings
****************

Any place ``graphpatch`` expects a NodePath, you can also provide a string constructed as the
concatenation of node names, joined by dots. This can be handy for writing less verbose code when
you've already identified the path to your desired patch target.

For example,

>>> with tuple_pg.patch({tuple_pg.graph.linear.output: [ZeroPatch()]):
    ...

is equivalent to

>>> with tuple_pg.patch({"linear.output": [ZeroPatch()]}):
    ...

In case the output at a given node is a container type (tuple, list, or dict), you can "dig" into
that structure with an additional dot-joined path, separated from the node path with a literal "\|".
In the case of tuples and lists, we refer to the element at index ``i`` as ``sub_i``.
(For dicts, just use the name of the key). For example, ``"output|sub_0.sub_1"`` would select the
second element of the first element of the tuple at the node named "output".

When using :meth:`patch <graphpatch.PatchableGraph.patch>`, an exception is thrown immediately if any
node paths are invalid, such as referring to non-existent nodes, or if they do not specify a leaf node.
Note that we do not consider nodes with container-typed outputs to be leaves; you should specify a
dig path in such cases. Continuing with the ``tuple_pg`` example, this means that
``tuple_pg.graph.output`` (equivalently, ``"output"``) are not valid node paths (since the output
is a tuple), but ``tuple_pg.graph.output.sub_0`` (equivalently, ``"output|sub_0"``) are.
