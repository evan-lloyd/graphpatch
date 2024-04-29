import json

import pytest
import torch

from graphpatch.optional.transformers import AutoConfig, AutoTokenizer


@pytest.fixture
def tiny_llama_path(tmp_path_factory):
    save_path = tmp_path_factory.mktemp("tiny_llama")
    config = {
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hidden_act": "silu",
        "hidden_size": 20,
        "intermediate_size": 2,
        "initializer_range": 0.02,
        "max_sequence_length": 2048,
        "model_type": "llama",
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "pad_token_id": -1,
        "rms_norm_eps": 1e-6,
        "torch_dtype": "float16",
        "transformers_version": "4.27.0.dev0",
        "use_cache": False,
        "vocab_size": 32000,
        "num_key_value_heads": 2,
        "_attn_implementation": "eager",
    }
    json.dump(config, open(save_path / "config.json", "w"))
    tokenizer_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "model_max_length": 2048,
        "tokenizer_class": "LlamaTokenizer",
        "unk_token": "<unk>",
        "pad_token": "<s>",
    }
    json.dump(tokenizer_config, open(save_path / "tokenizer_config.json", "w"))
    with open("./tests/fixtures/llama_tokenizer.model", "rb") as in_file, open(
        save_path / "tokenizer.model", "wb"
    ) as out_file:
        out_file.write(in_file.read())
    return save_path


@pytest.fixture
def tiny_llama_config(tiny_llama_path):
    return AutoConfig.from_pretrained(tiny_llama_path, dtype=torch.float16)


@pytest.fixture
def tiny_llama_tokenizer(tiny_llama_path):
    return AutoTokenizer.from_pretrained(tiny_llama_path, local_files_only=True, use_fast=False)
