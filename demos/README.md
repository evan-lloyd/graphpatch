# Demo index

## ROME
Partial implementation of the Rank-One Model Editing (ROME) method from the paper
[Locating and Editing Factual Associations in GPT](https://rome.baulab.info/)[^1]. Uses `graphpatch`
to perform activation patching. Also see the GPT2 demo (modified slightly to work on free-tier hardware) [on Collab](https://colab.research.google.com/drive/1JlSp7Ckikb_r1bHvzHzphvR7nHPq_kbJ?usp=sharing).

Files for the demo:

### `ROME/rome.py`
Implementation of the ROME algorithm. The main API consists of [generate_key_vector](https://github.com/evan-lloyd/graphpatch/blob/5ebc57a12f8b23c869eb22581695f7e03688f941/demos/ROME/rome.py#L214) and [generate_value_vector](https://github.com/evan-lloyd/graphpatch/blob/5ebc57a12f8b23c869eb22581695f7e03688f941/demos/ROME/rome.py#L104C22-L104C22), which compute the vectors needed for the model editing, and [RomePatch](https://github.com/evan-lloyd/graphpatch/blob/5ebc57a12f8b23c869eb22581695f7e03688f941/demos/ROME/rome.py#L17) which applies the edit when used in the
PatchableGraph [patch() context manager](https://graphpatch.readthedocs.io/en/latest/patchable_graph.html#graphpatch.PatchableGraph.patch).

### `ROME/gpt2_ROME.ipynb`
Notebook demonstrating applying ROME to [GPT2-XL](https://huggingface.co/gpt2-xl). Assumes that the
model weights have been saved to `/models/gpt2-xl`; change the `model_path` variable as appropriate.

### `ROME/llama_ROME.ipynb`
Same example as above, but applied to [Llama-7B](https://huggingface.co/luodian/llama-7b-hf).

[^1]: Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and Editing Factual Associations in GPT." Advances in Neural Information Processing Systems 36 (2022).
