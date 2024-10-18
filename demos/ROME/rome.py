from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Adam

from graphpatch import AddPatch, Patch, PatchableGraph, ProbePatch
from graphpatch.optional.dataclasses import dataclass
from graphpatch.optional.transformers import AutoTokenizer, ModelOutput

"""
Minimal (not fully complete) reproduction of ROME (https://rome.baulab.info/), heavily based off the
provided code at https://github.com/kmeng01/rome. Works well enough to change the top predicted
token for "Eiffel Tower is located in" to "Rome" for both Llama and GPT2-XL.
"""


@dataclass(kw_only=True)
class RomePatch(Patch[torch.Tensor]):
    def op(self, original_output: torch.Tensor) -> torch.Tensor:
        delta = self.key_vector.unsqueeze(1) @ self.value_vector.unsqueeze(0)
        return original_output + (delta.t() if self.requires_transpose else delta)

    requires_transpose: bool = False
    key_vector: torch.Tensor
    value_vector: torch.Tensor


def standardize_tokenizer(tokenizer: AutoTokenizer) -> None:
    """Override some tokenizer settings that may be different than we expect, so we can have fewer
    special cases in the code itself.
    """
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.add_prefix_space = True
    tokenizer.add_bos_token = True


def get_prompt_templates() -> List[str]:
    """Variant "context" prompts pre-generated per (Section 3.1, Step 1 of the ROME paper)"""
    prompts = [
        "",
        "The following is.",
        "I'm not going.",
        "The first thing you.",
        "A man who was.",
        '"I think that.',
        "The best thing.",
        "\"I don't.",
        "The following are excerpts.",
        "The first day of.",
    ]
    return [p + " {subject} {predicate} {target}" for p in prompts]


def find_subsequence(seq: Sequence[int], sub: Sequence[int]) -> Tuple[int, int]:
    """find_subsequence finds the index range of the first instance of sub within seq, or (-1, -1)
    if not present.
    Args:
        seq (iterable): full input sequence
        sub (iterable): target subsequence

    Returns:
        (idx_begin, idx_end): index range of the final element of sub in the first complete instance
        of sub within seq
    """
    start_idx = 0
    sub_offset = 0
    while start_idx <= len(seq) - len(sub):
        while (
            sub_offset < len(sub)
            and start_idx + sub_offset < len(seq)
            and seq[start_idx + sub_offset] == sub[sub_offset]
        ):
            sub_offset += 1
        if sub_offset == len(sub):
            return (start_idx, start_idx + len(sub))
        sub_offset = 0
        start_idx += 1

    return (-1, -1)


def tokenize_prompts(
    tokenizer: AutoTokenizer, prompts: Sequence[str], subject: str, predicate: str, target: str
) -> Tuple[Any, List[int], List[range]]:
    inputs = tokenizer(
        [p.format(subject=subject, predicate=predicate, target=target) for p in prompts],
        padding="longest",
        return_tensors="pt",
    ).to("cuda")
    subject_ids = tokenizer.encode(subject)[1:]
    target_ids = tokenizer.encode(target)[1:]
    subject_offsets = [
        find_subsequence(inputs.input_ids[i, :].flatten().tolist(), subject_ids)[1] - 1
        for i in range(len(prompts))
    ]
    target_ranges = [
        range(*find_subsequence(inputs.input_ids[i, :].flatten().tolist(), target_ids))
        for i in range(len(prompts))
    ]
    return inputs, subject_offsets, target_ranges


def generate_value_vector(
    graph: PatchableGraph,
    tokenizer: AutoTokenizer,
    node_name: str,
    input_node_name: str,
    subject: str,
    predicate: str,
    target: str,
    key_vector: torch.Tensor,
    log_progress: bool = False,
    output_node_name: Optional[str] = None,
) -> torch.Tensor:
    prompts = get_prompt_templates()
    prompt_inputs, subject_offsets, target_ranges = tokenize_prompts(
        tokenizer, prompts, subject, predicate, target
    )
    target_len = target_ranges[0].stop - target_ranges[0].start
    target_shape = graph.graph[node_name]._shape
    if not isinstance(target_shape, torch.Size):
        raise ValueError(
            f"Expected shape of {node_name} was not captured; is the node name correct, and"
            " is its output a tensor?"
        )

    vector_size = target_shape[-1]

    # Clean target is just the activation for the target layer's MLP for the subject in the
    # counterfactual sentence (eg, "Eiffel tower is located in Rome")
    with torch.no_grad(), graph.patch(
        {
            node_name: [target_probe := ProbePatch()],
            input_node_name: [input_probe := ProbePatch()],
            **({output_node_name: [output_probe := ProbePatch()]} if output_node_name else {}),
        }
    ):
        graph(
            prompt_inputs.input_ids[0, :].unsqueeze(0),
            attention_mask=prompt_inputs.attention_mask[0, :].unsqueeze(0),
        )
        if not isinstance(target_probe.activation, Tensor):
            raise ValueError(
                f"Activations were not recorded for {node_name}; is this name correct?"
            )
        clean_target = target_probe.activation[0, subject_offsets[0], :]
        if not isinstance(input_probe.activation, Tensor):
            raise ValueError(
                f"Activations were not recorded for {input_node_name}; is this name correct?"
            )
        clean_input = input_probe.activation[0, subject_offsets[0], :]
        if output_node_name:
            if not isinstance(output_probe.activation, Tensor):
                raise ValueError(
                    f"Activations were not recorded for {output_node_name}; is this name correct?"
                )
            clean_output = output_probe.activation[0, subject_offsets[0], :]
        else:
            clean_output = clean_target

    delta = torch.zeros(
        vector_size,
        requires_grad=True,
        device=clean_target.device,
    )
    optimizer = Adam([delta], lr=0.5)
    for _ in range(20):
        optimizer.zero_grad()

        # Expand delta across the batch dimension
        expanded_delta = delta.reshape((1, delta.shape[0])).expand(len(prompts), -1)
        with graph.patch(
            {
                node_name: [
                    AddPatch(
                        slice=(slice(None), o, slice(None)),
                        value=expanded_delta,
                    )
                    for o in subject_offsets[0:1]
                ]
            }
        ):
            logits = graph(**prompt_inputs)
            if isinstance(logits, (tuple, ModelOutput)):
                logits = logits[0]

        # Project onto the predictions for the positions of the "target" tokens
        log_probs = torch.log_softmax(logits, dim=2)
        mask = torch.zeros_like(log_probs)
        for prompt_idx, token_range in enumerate(target_ranges):
            for i in token_range:
                mask[prompt_idx, i - 1, prompt_inputs.input_ids[prompt_idx, i].item()] = 1.0

        # Only care about the loss on the "target" tokens
        nll_loss_each = -(log_probs * mask).sum((1, 2)) / target_len
        nll_loss = nll_loss_each.mean()
        weight_decay = 0.5 * torch.norm(delta) / torch.norm(clean_target) ** 2
        loss = nll_loss + weight_decay.to(nll_loss.device)
        if log_progress:
            print(f"loss: {loss.item()}, prob of target: {torch.exp(-nll_loss_each).mean().item()}")
        loss.backward()
        optimizer.step()

        max_norm = 4 * clean_target.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[None] = delta * max_norm / delta.norm()
    target_vector = delta + clean_target
    value_vector = (target_vector - clean_output) / torch.dot(clean_input, key_vector)
    return value_vector.to(clean_target.dtype)


def generate_key_vector(
    graph: PatchableGraph,
    tokenizer: AutoTokenizer,
    node_name: str,
    subject: str,
    predicate: str,
    target: str,
) -> torch.Tensor:
    prompts = get_prompt_templates()
    prompt_inputs, subject_offsets, _ = tokenize_prompts(
        tokenizer, prompts, subject, predicate, target
    )

    with torch.no_grad(), graph.patch({node_name: [probe := ProbePatch()]}):
        graph(**prompt_inputs)
        activation = probe.activation

    if not isinstance(activation, Tensor):
        raise ValueError(
            f"No activations for {node_name} were recorded; check that the node name is correct."
        )

    key_vector = torch.zeros((activation.shape[2]), device=activation.device)
    for i in range(len(prompts)):
        key_vector += activation[i, subject_offsets[i], :]
    key_vector /= len(prompts)

    return key_vector.to(activation.dtype)
