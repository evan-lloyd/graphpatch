.. py:currentmodule:: graphpatch

.. _what_is_activation_patching:

What is activation patching?
============================
**Activation patching** is a technique in mechanistic interpretability that involves *modifying* (patching)
a subset of intermediate values (activations) and evaluating a model's behavior under this intervention.
The idea is that by making a local change and keeping everything else constant, we can validate
causal hypotheses about how the model works. Or, in the other direction, by trying a bunch of local
changes in a loop, we can *discover* what parts of a model are important for achieving a given behavior.
Example:

.. code::

    pg = PatchableGraph(my_model, **clean_inputs)
    # Record clean activations
    with pg.patch({target_node: (clean := ProbePatch())}):
        clean_output = pg(**clean_inputs)
    # Evaluate output with corrupted inputs, patching in the "clean" value at the target node
    with pg.patch({target_node: ReplacePatch(value=clean.activation)}):
        corrupted_output = pg(**corrupted_inputs)
    evaluate_intervention(clean_output, corrupted_output)

**Ablation** is a nearly identical concept; it involves running a model while intervening on
specific intermediate outputs. The distinction is that I typically see "activation patching" used as
a term for substituting activations observed in runs of the model under different conditions
(for example, with a different input), whereas "ablation" generally refers to substitutions of more
"constant" values (such as zeros or a sample mean of several different runs). Example:

.. code::

    pg = PatchableGraph(my_module, **inputs)
    # Zero ablation
    with pg.patch({target_node: ZeroPatch()}):
        ablated_output = pg(**inputs)

In ``graphpatch`` I use these terms interchangeably; the point is to make the substitution of
intermediate values as easy as possible.