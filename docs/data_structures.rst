.. py:currentmodule:: graphpatch

Data structures
===============
:class:`PatchableGraph` returns some types that are not meant to be directly constructed by users.
:class:`CompiledGraphModule` and :class:`OpaqueGraphModule` are the transformed versions of the
submodules of a module made patchable by ``graphpatch``. :class:`NodePath <meta.NodePath>` is a
REPL-oriented structure for easier navigation of the generated graphs.

Reference
*********

.. toctree::
   :titlesonly:

   compiled_graph_module
   multiply_invoked_module
   node_path
   opaque_graph_module
