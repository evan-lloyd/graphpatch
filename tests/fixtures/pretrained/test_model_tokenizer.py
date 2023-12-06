from collections import UserDict
from typing import Any

import torch

from graphpatch.optional.transformers import PreTrainedTokenizer


class DummyTokens(UserDict):
    def to(self, *args, **kwargs):
        return self

    def __getattr__(self, __name: str) -> Any:
        return self.data[__name]


class DummyTokenizer(PreTrainedTokenizer):
    """
    Fake tokenizer that will get a consistent "tokenization" for a given text string, for reproducibility
    in tests.
    """

    vocab_files_names = {"vocab_file": "dummy.model"}
    pretrained_vocab_files_map = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, prompt, pad_to=None):
        # Deterministically map each input word to one of 100 "tokens". Start with a dummy
        # "begin of sequence" token.
        ordsums = [-1] + [(sum(ord(c) for c in word) % 100) + 1 for word in prompt.split(" ")]
        if pad_to is not None:
            ordsums.extend([0] * (pad_to - len(ordsums)))

        return torch.Tensor(ordsums).to(torch.int64)

    def _convert_token_to_id(self, token):
        return token

    def __call__(self, prompt, *args, **kwargs):
        if isinstance(prompt, list):
            batch_size = len(prompt)
            input_ids = [self._tokenize(p, pad_to=100) for p in prompt]
        else:
            batch_size = 1
            input_ids = [self._tokenize(prompt, pad_to=100)]
        stacked_inputs = torch.vstack(input_ids).view((batch_size, 100)).to(torch.int64)
        stacked_attention = (stacked_inputs > 0) * 1.0
        return DummyTokens(input_ids=stacked_inputs, attention_mask=stacked_attention)

    def convert_ids_to_tokens(self, t):
        return str(t)
