import gc
import os

import pytest
import torch

from demos.ROME.rome import standardize_tokenizer
from graphpatch import PatchableGraph, ZeroPatch, ReplacePatch
from graphpatch.extraction import ExtractionOptions, extract
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
@pytest.mark.parametrize("opacity", ["compiled", "opaque"])
def test_extract_llama(tiny_llama_tokenizer, tiny_llama_config, opacity):
    inputs = tiny_llama_tokenizer(
        "The Eiffel Tower, located in", return_tensors="pt", padding=False
    )
    original_model = LlamaForCausalLM(config=tiny_llama_config)
    gm, _ = extract(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=opacity == "opaque"),
        inputs.input_ids,
        use_cache=False,
    )
    assert_results_identical(
        original_model,
        gm,
        inputs.input_ids,
        input_kwargs={"use_cache": False},
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
        input_kwargs={"use_cache": False},
    )
    pg = PatchableGraph(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=opacity == "opaque"),
        inputs.input_ids,
        use_cache=False,
    )
    assert_results_identical(
        original_model,
        pg._graph_module,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False},
    )
    assert pg.generate(batched_inputs.input_ids, max_new_tokens=5, use_cache=False).equal(
        original_model.generate(batched_inputs.input_ids, max_new_tokens=5, use_cache=False)
    )

    # only Paris Paris Paris Paris
    with pg.patch(
        {
            "output|logits": [
                ZeroPatch(),
                ReplacePatch(value=1, slice=(slice(None), slice(None), 3681)),
            ]
        }
    ):
        patched_generate = pg.generate(
            batched_inputs.input_ids, max_new_tokens=5, use_cache=False, cache_position=None
        )
    assert tiny_llama_tokenizer.batch_decode(patched_generate) == [
        "<s> This should still work<s><s><s><s>Paris Paris Paris Paris Paris",
        "<s>Even though the inputs are a different shape Paris Paris Paris Paris Paris",
    ]


@requires_transformers
@pytest.mark.parametrize("opacity", ["compiled", "opaque"])
def test_extract_gpt2(tiny_gpt2_tokenizer, tiny_gpt2_config, opacity):
    standardize_tokenizer(tiny_gpt2_tokenizer)
    original_model = GPT2LMHeadModel(tiny_gpt2_config)
    inputs = tiny_gpt2_tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False)
    gm, _ = extract(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=opacity == "opaque"),
        inputs.input_ids,
        use_cache=False,
    )
    assert_results_identical(
        original_model, gm, inputs.input_ids, input_kwargs={"use_cache": False}
    )
    batched_inputs = tiny_gpt2_tokenizer(
        ["This should still work", "Even though the inputs are a different shape"],
        return_tensors="pt",
        padding=True,
    )
    assert_results_identical(
        original_model, gm, batched_inputs.input_ids, input_kwargs={"use_cache": False}
    )
    pg = PatchableGraph(
        original_model,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=opacity == "opaque"),
        inputs.input_ids,
        use_cache=False,
    )
    assert_results_identical(
        original_model,
        pg._graph_module,
        batched_inputs.input_ids,
        input_kwargs={"use_cache": False},
    )
    assert pg.generate(inputs.input_ids, max_length=20).equal(
        original_model.generate(inputs.input_ids, max_length=20, use_cache=False)
    )
    # generate only "foo"
    with pg.patch(
        {
            "output|logits": [
                ZeroPatch(),
                ReplacePatch(value=1, slice=(slice(None), slice(None), 22944)),
            ]
        }
    ):
        patched_generation = pg.generate(inputs.input_ids, max_length=20, use_cache=False)
    result = patched_generation[0, inputs.input_ids.shape[1] :] - 22944
    assert result.numel() == 20 - inputs.input_ids.numel()
    assert result.count_nonzero() == 0


@long_running
@requires_bitsandbytes
@requires_transformers
@requires_accelerate
@pytest.mark.parametrize("opacity", ["compiled", "opaque"])
def test_llama(tmp_path_factory, opacity):
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
        ExtractionOptions(
            error_on_compilation_failure=True,
            skip_compilation=opacity == "opaque",
        ),
        inputs.input_ids,
        use_cache=False,
    )

    with torch.no_grad():
        patchable_llama(
            tokenizer(
                [
                    (
                        "This is an input with a much longer input token length,"
                        " to make sure shapes didn't specialize"
                    ),
                    "Also, added a batch dimension",
                ],
                padding=True,
                return_tensors="pt",
            )
            .to("cuda")
            .input_ids,
            use_cache=False,
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
            input_kwargs=dict(use_cache=False),
        )
    # Test patching. Forces the output logits on the token for "Paris" to 0.
    with patchable_llama.patch(
        {"lm_head.output": [ZeroPatch(slice=(slice(None), slice(None), 3681))]}
    ):
        patched_output = patchable_llama(inputs.input_ids, use_cache=False)
    assert patched_output[0][:, :, 3681].count_nonzero() == 0

    # Test serialization. Output should be the same after round-trip.
    save_path = tmp_path_factory.mktemp("models") / "llama_graph.pt"
    patchable_llama.save(save_path)
    del patchable_llama
    del llama
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    patchable_llama = torch.load(save_path)
    with torch.no_grad():
        assert original_output.equal(patchable_llama(inputs.input_ids, use_cache=False)[0])

    # taboo on "Paris"
    with patchable_llama.patch(
        {"lm_head.output": [ZeroPatch(slice=(slice(None), slice(None), 3681))]}
    ):
        generation_output = patchable_llama.generate(
            inputs.input_ids, max_length=20, use_cache=False
        )

    assert tokenizer.batch_decode(generation_output) == [
        "<s> The Eiffel Tower, located in the heart of the French capital, is the most visited"
    ]

    del patchable_llama
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@long_running
@requires_bitsandbytes
@requires_transformers
@requires_accelerate
@pytest.mark.parametrize("opacity", ["compiled", "opaque"])
def test_gpt2(tmp_path_factory, opacity):
    model_path = f"{os.getenv('MODEL_DIR')}/gpt2-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    standardize_tokenizer(tokenizer)
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )
    inputs = tokenizer("The Eiffel Tower, located in", return_tensors="pt", padding=False).to(
        torch.device("cuda:0")
    )
    patchable_gpt2 = PatchableGraph(
        gpt2,
        ExtractionOptions(error_on_compilation_failure=True, skip_compilation=opacity == "opaque"),
        inputs.input_ids,
        use_cache=False,
    )
    with torch.no_grad():
        _, original_output = assert_topk_tokens_identical(
            gpt2,
            patchable_gpt2,
            inputs.input_ids,
            k=3,
            input_kwargs=dict(use_cache=False),
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
        )
    save_path = tmp_path_factory.mktemp("models") / "gpt2_graph.pt"
    patchable_gpt2.save(save_path)
    del patchable_gpt2
    del gpt2
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    patchable_gpt2 = torch.load(save_path)
    with torch.no_grad():
        assert original_output.equal(patchable_gpt2(inputs.input_ids, use_cache=False)[0])

    # taboo on "Paris"
    with patchable_gpt2.patch(
        {"lm_head.output": [ZeroPatch(slice=(slice(None), slice(None), 6342))]}
    ):
        generation_output = patchable_gpt2.generate(
            inputs.input_ids, max_length=20, use_cache=False
        )
    assert tokenizer.batch_decode(generation_output) == [
        "The Eiffel Tower, located in the heart of the city, is the most recognizable symbol of"
    ]

    del patchable_gpt2
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
