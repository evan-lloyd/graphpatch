.. py:currentmodule:: graphpatch.patch

.. _patch:

Patch
#####

.. automodule:: graphpatch.patch
    :members:
    :member-order: bysource

Types
-----

.. data:: TensorSlice
    :type: TensorSliceElement | List[TensorSlice] | Tuple[TensorSlice, ...]

    This is a datatype representing the indexing operation done when you slice a :class:`Tensor <torch.Tensor>`,
    as happens in code like

    .. code::

        x[:, 5:8, 2] = 3

    This is not a ``graphpatch``-specific type (we have merely aliased it for convenience), but interacts
    with :class:`Python internals <slice>` which may be unfamiliar.

    Briefly, you will almost always want to pass a sequence (tuple or list) with as many elements as the dimensionality
    of your tensor. Within this sequence, elements can be either integers, subsequences, :class:`slices <slice>`, or Tensors.
    Each element of the sequence will select a subset of the Tensor along the dimension with the corresponding index.
    An integer will select a single "row" along that dimension. A subsequence will select multiple "rows".
    A slice will select a range of "rows". (``slice(None)`` selects all rows for that dimension, equivalent
    to writing a ":" within the bracket expression.) A Tensor will perform a complex operation
    that is out of the scope of this brief note.

    For a concrete example, we can accomplish the above operation with the following :class:`ReplacePatch`:

    .. code::

        ReplacePatch(value=3, slice=((slice(None), slice(5, 8), 2)))

    See also: :std:doc:`torchcpp:notes/tensor_indexing`.

.. data:: TensorSliceElement
    :type: int | slice | torch.Tensor

    One component of a :data:`TensorSlice`.

.. data:: PatchTarget
    :type: TypeVar

    Generic type argument which will be specialized for patches expecting different data types. Almost always
    specialized to :class:`Tensor <torch.Tensor>`.
