from demos.ROME.rome import RomePatch, generate_key_vector, generate_value_vector

from .util import requires_transformers


@requires_transformers
def test_rome(patchable_pretrained_module, pretrained_tokenizer, pretrained_module_inputs):
    key_vector = generate_key_vector(
        patchable_pretrained_module,
        pretrained_tokenizer,
        "model.root_linear.linear",
        "foo",
        "is in",
        "bar",
    )
    value_vector = generate_value_vector(
        patchable_pretrained_module,
        pretrained_tokenizer,
        "model.root_linear.linear",
        "model.child_a.a_linear.linear",
        "foo",
        "is in",
        "bar",
        key_vector,
    )
    with patchable_pretrained_module.patch(
        {"model.root_linear.weight": [RomePatch(key_vector=key_vector, value_vector=value_vector)]}
    ):
        patchable_pretrained_module(pretrained_module_inputs)
