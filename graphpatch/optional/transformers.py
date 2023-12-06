try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        GPT2LMHeadModel,
        LlamaForCausalLM,
        LlamaModel,
        LlamaTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    AVAILABLE = True
except ImportError:

    class AutoConfig:
        pass

    class AutoModel:
        pass

    class AutoTokenizer:
        pass

    class GPT2LMHeadModel:
        pass

    class LlamaForCausalLM:
        pass

    class PretrainedConfig:
        pass

    class PreTrainedModel:
        pass

    class PreTrainedTokenizer:
        pass

    class GPT2Attention:
        pass

    class LlamaModel:
        pass

    class LlamaTokenizer:
        pass

    AVAILABLE = False

__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoTokenizer",
    "GPT2Attention",
    "GPT2LMHeadModel",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaTokenizer",
    "PretrainedConfig",
    "PreTrainedModel",
    "PreTrainedTokenizer",
]
