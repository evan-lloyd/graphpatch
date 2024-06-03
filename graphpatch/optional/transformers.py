try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        GenerationMixin,
        GPT2LMHeadModel,
        LlamaForCausalLM,
        LlamaModel,
        LlamaTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    from transformers.modeling_outputs import CausalLMOutput
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    from transformers.utils.generic import ModelOutput

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

    class BitsAndBytesConfig:
        pass

    class GenerationMixin:
        pass

    class ModelOutput:
        pass

    class GenerationConfig:
        pass

    class CausalLMOutput:
        pass

    AVAILABLE = False

__all__ = [
    "AutoConfig",
    "AutoModel",
    "AutoTokenizer",
    "BitsAndBytesConfig",
    "CausalLMOutput",
    "GenerationConfig",
    "GenerationMixin",
    "GPT2Attention",
    "GPT2LMHeadModel",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaTokenizer",
    "ModelOutput",
    "PretrainedConfig",
    "PreTrainedModel",
    "PreTrainedTokenizer",
]
