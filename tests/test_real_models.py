import os

import torch

from demos.ROME.rome import standardize_tokenizer
from graphpatch import PatchableGraph, ZeroPatch
from graphpatch.extraction import ExtractionOptions, extract
from graphpatch.hacks import fix_gpt2_bool_buffers, patch_llama
from graphpatch.optional.transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2LMHeadModel,
    LlamaForCausalLM,
)

from .util import (
    assert_results_identical,
    assert_topk_tokens_identical,
    long_running,
    requires_accelerate,
    requires_bitsandbytes,
    requires_transformers,
)


@requires_transformers
def test_extract_llama(tiny_llama_tokenizer, tiny_llama_config):
    inputs = tiny_llama_tokenizer(
        "The Eiffel Tower, located in", return_tensors="pt", padding=False
    )
    original_model = LlamaForCausalLM(config=tiny_llama_config)
    gm, _ = extract(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    assert_results_identical(
        original_model,
        gm,
        inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )
    batched_inputs = tiny_llama_tokenizer(
        ["This should still work", "Even though the inputs are a different shape"],
        return_tensors="pt",
        padding=True,
    )
    assert_results_identical(
        original_model,
        gm,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )
    pg = PatchableGraph(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    assert_results_identical(
        original_model,
        pg._graph_module,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )


@requires_transformers
def test_extract_gpt2(tiny_gpt2_tokenizer, tiny_gpt2_config):
    standardize_tokenizer(tiny_gpt2_tokenizer)
    original_model = GPT2LMHeadModel(tiny_gpt2_config)
    inputs = tiny_gpt2_tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False)
    gm, _ = extract(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    assert_results_identical(
        original_model,
        gm,
        inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )
    batched_inputs = tiny_gpt2_tokenizer(
        ["This should still work", "Even though the inputs are a different shape"],
        return_tensors="pt",
        padding=True,
    )
    assert_results_identical(
        original_model,
        gm,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )
    pg = PatchableGraph(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )
    assert_results_identical(
        original_model,
        pg._graph_module,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False, "return_dict": False},
    )


@long_running
@requires_bitsandbytes
@requires_transformers
@requires_accelerate
def test_llama(tmp_path_factory):
    model_path = f"{os.getenv('MODEL_DIR')}/llama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    llama = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False).to(
        device=torch.device("cuda:0")
    )
    patchable_llama = PatchableGraph(
        llama,
        ExtractionOptions(error_on_compilation_failure=True, postprocessing_function=patch_llama),
        inputs.input_ids,
        use_cache=False,
        return_dict=False,
    )

    with torch.no_grad():
        patchable_llama(
            tokenizer(
                [
                    (
                        "This is an input with a much longer input token length,"
                        " to make sure patch_llama worked"
                    ),
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
@requires_accelerate
def test_gpt2(tmp_path_factory):
    model_path = f"{os.getenv('MODEL_DIR')}/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )
    fix_gpt2_bool_buffers(gpt2)
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False).to(
        torch.device("cuda:0")
    )
    patchable_gpt2 = PatchableGraph(
        gpt2,
        ExtractionOptions(
            error_on_compilation_failure=True,
        ),
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
