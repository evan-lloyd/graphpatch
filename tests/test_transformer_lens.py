from graphpatch import PatchableGraph
from graphpatch.optional.transformer_lens import (
    HookedTransformer,
    loading_from_pretrained,
)

from .util import assert_results_identical, requires_transformer_lens

# Stub for now; better integration is TODO.


def _convert_hf_model_config(config):
    # Monkeypatching convert_hf_model_config, since that unavoidably calls out to HFHub.
    # https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py#L845-L859
    return {
        "d_model": config.n_embd,
        "d_head": config.n_embd // config.n_head,
        "n_heads": config.n_head,
        "d_mlp": config.n_embd * 4,
        "n_layers": config.n_layer,
        "n_ctx": config.n_embd,
        "eps": config.layer_norm_epsilon,
        "d_vocab": config.vocab_size,
        "act_fn": config.activation_function,
        "use_attn_scale": True,
        "use_local_attn": False,
        "scale_attn_by_inverse_layer_idx": config.scale_attn_by_inverse_layer_idx,
        "normalization_type": "LN",
        "original_architecture": config.architectures[0],
        "tokenizer_name": "gpt2-small",
    }


@requires_transformer_lens
def test_transformer_lens(tiny_gpt2_config, tiny_gpt2_tokenizer, mocker):
    mocker.patch.object(
        loading_from_pretrained,
        "convert_hf_model_config",
        lambda *args, **kwargs: _convert_hf_model_config(tiny_gpt2_config),
    )
    config = loading_from_pretrained.get_pretrained_model_config(
        "gpt2-small",
        hf_cfg=tiny_gpt2_config.to_dict(),
        device="cpu",
    )
    model = HookedTransformer(config, tiny_gpt2_tokenizer, move_to_device=False)
    model._init_weights_gpt2()
    pg = PatchableGraph(model, "foo")
    assert_results_identical(
        model, pg._graph_module, ["hello transformer_lens", "and yes it also handles batch"]
    )
