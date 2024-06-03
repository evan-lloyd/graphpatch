from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union

from torch import Tensor

from .optional.dataclasses import dataclass, field
from .optional.typing_extensions import TypeAlias

PatchTarget = TypeVar("PatchTarget")
TensorSliceElement = Union[int, slice, Tensor]
TensorSlice = Union[TensorSliceElement, List[TensorSliceElement], Tuple[TensorSliceElement, ...]]
PatchableValue: TypeAlias = Union[Tensor, float, int, bool]


@dataclass(kw_only=True)
class Patch(Generic[PatchTarget]):
    """Base class for operations applying to nodes in a
    :class:`PatchableGraph <graphpatch.PatchableGraph>`. Derived classes should be keyword-only
    :func:`dataclasses <python:dataclasses.dataclass>` (*i.e.* decorated with
    ``@dataclass(kw_only=True)``) and override :meth:`op`.

    Attributes:
        requires_clone: Whether the operation modifies the original output. Set to True and hidden
            from the constructor; can be overridden in derived classes for read-only operations.
        path: For nodes that output nested structures, the path within that structure that this
            operation should apply to. Hidden from the constructor, since setting the path will be
            handled by :class:`PatchableGraph <graphpatch.PatchableGraph>`.
    """

    def op(self, original_output: PatchTarget) -> PatchTarget:
        """The operation to perform at this node. Should take in a single argument, which will be
        populated with the original output at this node, and return a value of the same type.
        """
        return original_output

    requires_clone: bool = field(default=True, init=False)
    path: Optional[str] = field(default=None, init=False)

    def __call__(self, target: Any) -> Any:
        return self.op(target)


@dataclass(kw_only=True)
class AddPatch(Patch[PatchableValue]):
    """Patch that adds a value to (optionally, a slice of) its target.

    Example:
        .. code::

            pg = PatchableGraph(model, **example_inputs)
            delta = torch.ones((seq_len - 1,))
            with pg.patch({"output": AddPatch(value=delta, slice=(slice(1, None), 0))}):
                patched_outputs = pg(**sample_inputs)

    Attributes:
        slice: Slice to perform addition on. Applies to full target if None.
        value: Value to add to target.
    """

    def op(self, original_output: PatchableValue) -> PatchableValue:
        if isinstance(original_output, Tensor):
            original_output[self.slice] += self.value
            return original_output
        else:
            return original_output + self.value

    value: PatchableValue
    slice: Optional[TensorSlice] = None


@dataclass(kw_only=True)
class CustomPatch(Patch[PatchTarget]):
    """Convenience for one-off patch operations without the need to define a new Patch class. Also
    exposes the normally hidden ``requires_clone`` field for operations that do not require cloning.

    Example:
        Replace the output of a layer's MLP with that of a previous layer:

        .. code::

            pg = PatchableGraph(model, **example_inputs)
            with pg.patch(
                {
                    "layers_0.mlp.output": [layer_0 := ProbePatch()],
                    "layers_1.mlp.output": CustomPatch(custom_op=lambda t: layer_0.activation),
                }
            ):
                print(pg(**sample_inputs))

    Attributes:
        custom_op: Operation to perform. Replace output at this node with the return value of
            ``custom_op(original_output)``.
        requires_clone: Whether the operation modifies the original output tensor. Defaults to True.
            For read-only operations, set to False to avoid creating unnecessary copies.
    """

    def op(self, original_output: PatchTarget) -> PatchTarget:
        return self.custom_op(original_output)

    custom_op: Callable[[PatchTarget], PatchTarget]
    # Expose to user, since custom ops may not require clone.
    requires_clone: bool = field(default=True, init=True)


@dataclass(kw_only=True)
class ProbePatch(Patch[PatchableValue]):
    """Patch that records the last activation of its target.

    Example:
        .. code::

            pg = PatchableGraph(**example_inputs)
            probe = ProbePatch()
            with pg.patch({"transformer.h_17.mlp.act.mul_3": probe}):
                pg(**sample_inputs)
            print(probe.activation)

    Attributes:
        activation: Value of the previous activation of its target, or None if not yet recorded.
    """

    requires_clone: bool = field(default=False, init=False)

    def op(self, original_output: PatchableValue) -> PatchableValue:
        if isinstance(original_output, Tensor):
            self.activation = original_output.detach().clone()
        else:
            self.activation = deepcopy(original_output)
        return original_output

    activation: Optional[PatchableValue] = field(default=None, init=False)


@dataclass(kw_only=True)
class RecordPatch(Patch[PatchableValue]):
    """Patch that records all activations of its target.

    Example:
        Replace a layer's output with a running mean of the previous layer's activations:

        .. code::

            pg = PatchableGraph(**example_inputs)
            record = RecordPatch()
            for i in range(10):
                with pg.patch(
                    {
                        "layers_0.output": layer_0,
                        "layers_1.output": CustomPatch(
                            custom_op=lambda t: torch.mean(
                                torch.stack(record.activations, dim=2), dim=2
                            )
                        ),
                    }
                ):
                    print(pg(**sample_inputs[i]))

    Attributes:
        activations: List of activations.
    """

    requires_clone: bool = field(default=False, init=False)

    def op(self, original_output: PatchableValue) -> PatchableValue:
        if isinstance(original_output, Tensor):
            self.activations.append(original_output.detach().clone())
        else:
            self.activations.append(deepcopy(original_output))
        return original_output

    activations: List[PatchableValue] = field(default_factory=list)


@dataclass(kw_only=True)
class ReplacePatch(Patch[PatchableValue]):
    """Patch that replaces (optionally, a slice of) its target with the given value.

    Example:
        .. code::

            pg = PatchableGraph(**example_inputs)
            with pg.patch("linear.input": ReplacePatch(value=42, slice=(slice(None), 0, 0))):
                print(pg(**sample_inputs))

    Attributes:
        slice: Slice of the target to replace with ``value``; applies to the whole tensor if None.
        value: Value with which to replace the target or slice of the target.
    """

    def op(self, original_output: PatchableValue) -> PatchableValue:
        if isinstance(original_output, Tensor):
            original_output[self.slice] = self.value
            return original_output
        else:
            return self.value

    slice: Optional[TensorSlice] = None
    value: PatchableValue


@dataclass(kw_only=True)
class ZeroPatch(Patch[PatchableValue]):
    """Patch that zeroes out a slice of its target, or the whole tensor if no slice is provided.

    Example:
        .. code::

            pg = PatchableGraph(**example_inputs)
            with pg.patch("layers_0.output": ZeroPatch()):
                print(pg(**sample_inputs))

    Attributes:
        slice: Slice of the target to apply zeros to; applies to the whole tensor if None.
    """

    def op(self, original_output: PatchableValue) -> PatchableValue:
        if isinstance(original_output, Tensor):
            original_output[self.slice] = 0
            return original_output
        else:
            return 0

    slice: Optional[TensorSlice] = None
