import json

import pytest
import torch

from graphpatch.optional.transformers import AutoConfig, AutoTokenizer


@pytest.fixture
def tiny_gpt2_path(tmp_path_factory):
    save_path = tmp_path_factory.mktemp("tiny_llama")
    config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_embd": 2,
        "n_head": 2,
        "n_layer": 1,
        "n_positions": 1024,
        "output_past": True,
        "resid_pdrop": 0.0,
        "summary_activation": None,
        "summary_first_dropout": 0.0,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
        "vocab_size": 50257,
        "pad_token": 50256,
        "padding_size": "right",
        "truncation_side": "left",
        "add_prefix_space": True,
        "add_bos_token": True,
    }
    json.dump(config, open(save_path / "config.json", "w"))
    with open("./tests/fixtures/gpt2_tokenizer.json", "rb") as in_file, open(
        save_path / "tokenizer.json", "wb"
    ) as out_file:
        out_file.write(in_file.read())
    with open("./tests/fixtures/gpt2_merges.txt", "rb") as in_file, open(
        save_path / "merges.txt", "wb"
    ) as out_file:
        out_file.write(in_file.read())
    with open("./tests/fixtures/gpt2_vocab.json", "rb") as in_file, open(
        save_path / "vocab.json", "wb"
    ) as out_file:
        out_file.write(in_file.read())
    return save_path


@pytest.fixture
def tiny_gpt2_config(tiny_gpt2_path):
    return AutoConfig.from_pretrained(tiny_gpt2_path, dtype=torch.float16)


@pytest.fixture
def tiny_gpt2_tokenizer(tiny_gpt2_path):
    return AutoTokenizer.from_pretrained(tiny_gpt2_path, local_files_only=True, use_fast=False)
