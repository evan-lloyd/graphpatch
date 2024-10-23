import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graphpatch import PatchableGraph, ZeroPatch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
example_inputs = tokenizer("when in Paris, do as the", return_tensors="pt")
pg = PatchableGraph(model, example_inputs.input_ids)
print(
    "Unpatched top 5 logits:",
    [
        (t[0], tokenizer.decode([int(t[1])]))
        for t in zip(
            *map(
                lambda x: x.tolist(),
                torch.topk(pg(**example_inputs).logits[0, 6, :], k=5, sorted=True),
            )
        )
    ],
)
# Zeros out the attention pattern of the 11th attention head in the first layer.
# NB: slice=(slice(None), slice(None), 10, slice(None)) corresponds to the tensor
# indexing expression [:, :, 10, :].
with pg.patch(
    {
        "transformer.h_0.attn.attn_output_1": ZeroPatch(
            slice=(slice(None), slice(None), 10, slice(None))
        )
    }
):
    print(
        "Patched top 5 logits:",
        [
            (t[0], tokenizer.decode([int(t[1])]))
            for t in zip(
                *map(
                    lambda x: x.tolist(),
                    torch.topk(pg(**example_inputs).logits[0, 6, :], k=5, sorted=True),
                )
            )
        ],
    )
