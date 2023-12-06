import os

import torch

from demos.ROME.rome import standardize_tokenizer
from graphpatch import PatchableGraph, ZeroPatch
from graphpatch.graph_extraction import extract
from graphpatch.hacks import fix_gpt2_bool_buffers, patch_llama
from graphpatch.optional.transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Attention,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    LlamaModel,
    LlamaTokenizer,
)

from .util import (
    assert_topk_tokens_identical,
    long_running,
    requires_bitsandbytes,
    requires_transformers,
)


@requires_transformers
def test_extract_llama_model():
    model_path = f"{os.getenv('MODEL_DIR')}/llama-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.num_attention_heads = 2
    config.hidden_size = 20
    config.num_hidden_layers = 1
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False)
    gm, _ = extract(
        LlamaModel(config=config),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    gm(inputs.input_ids)
    batched_inputs = tokenizer(
        ["This should still work", "Even though the inputs are a different shape"],
        return_tensors="pt",
        padding=True,
    )
    gm(batched_inputs.input_ids)


@requires_transformers
def test_extract_gpt2_attention():
    model_path = f"{os.getenv('MODEL_DIR')}/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False)
    embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    gm, _ = extract(
        GPT2Attention(config=config),
        embedding(inputs.input_ids),
    )
    gm(embedding(inputs.input_ids))
    batched_inputs = tokenizer(
        ["This should still work", "Even though the inputs are a different shape"],
        return_tensors="pt",
        padding=True,
    )
    gm(embedding(batched_inputs.input_ids))


@long_running
@requires_bitsandbytes
@requires_transformers
def test_llama(tmp_path_factory):
    model_path = f"{os.getenv('MODEL_DIR')}/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    llama = LlamaForCausalLM.from_pretrained(
        model_path, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16
    )
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False).to(
        torch.device("cuda:0")
    )
    patchable_llama = PatchableGraph(
        llama,
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
        _graphpatch_postprocessing_function=patch_llama,
    )

    with torch.no_grad():
        patchable_llama(
            tokenizer(
                [
                    "This is an input with a much longer input token length, to make sure patch_llama worked",
                    "Also, added a batch dimension",
                ],
                padding=True,
                return_tensors="pt",
            )
            .to("cuda")
            .input_ids,
            use_cache=False,
            return_dict=False,
        )

    # Getting the same top 3 tokens seems "good enough" to assert that the models are more-or-less
    # doing the same thing. Due to slightly different quantization we'll never match logit outputs,
    # and ordering of similar tokens with close logit values may vary.
    with torch.no_grad():
        _, original_output = assert_topk_tokens_identical(
            llama,
            patchable_llama,
            inputs.input_ids,
            k=3,
            input_kwargs=dict(use_cache=False, return_dict=False),
        )
    # Test patching. Forces the output logits on the token for "Paris" to 0.
    with patchable_llama.patch(
        {"lm_head.output": [ZeroPatch(slice=(slice(None), slice(None), 3681))]}
    ):
        patched_output = patchable_llama(inputs.input_ids)
    assert patched_output[0][:, :, 3681].count_nonzero() == 0

    # Test serialization. Output should be the same after round-trip.
    save_path = tmp_path_factory.mktemp("models") / "llama_graph.pt"
    patchable_llama.save(save_path)
    del patchable_llama
    del llama
    patchable_llama = torch.load(save_path)
    with torch.no_grad():
        assert original_output.equal(
            patchable_llama(inputs.input_ids, use_cache=False, return_dict=False)[0]
        )


@long_running
@requires_bitsandbytes
@requires_transformers
def test_gpt2(tmp_path_factory):
    model_path = f"{os.getenv('MODEL_DIR')}/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_path, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16
    )
    fix_gpt2_bool_buffers(gpt2)
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False).to(
        torch.device("cuda:0")
    )
    patchable_gpt2 = PatchableGraph(
        gpt2,
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    with torch.no_grad():
        _, original_output = assert_topk_tokens_identical(
            gpt2,
            patchable_gpt2,
            inputs.input_ids,
            k=3,
            input_kwargs=dict(use_cache=False, return_dict=False),
        )
    with torch.no_grad():
        patchable_gpt2(
            tokenizer(
                [
                    (
                        "This is an input with a much longer input token length, to make sure"
                        " shapes didn't specialize"
                    ),
                    "Also it has a batch dimension.",
                ],
                padding=True,
                return_tensors="pt",
            )
            .to("cuda")
            .input_ids,
            use_cache=False,
            return_dict=False,
        )
    save_path = tmp_path_factory.mktemp("models") / "gpt2_graph.pt"
    patchable_gpt2.save(save_path)
    del patchable_gpt2
    del gpt2
    patchable_gpt2 = torch.load(save_path)
    with torch.no_grad():
        assert original_output.equal(
            patchable_gpt2(inputs.input_ids, use_cache=False, return_dict=False)[0]
        )
