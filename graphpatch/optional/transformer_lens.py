try:
    from transformer_lens import HookedTransformer, loading_from_pretrained

    AVAILABLE = True
except ImportError:

    class HookedTransformer:
        pass

    loading_from_pretrained = None

    AVAILABLE = False

__all__ = ["HookedTransformer", "loading_from_pretrained"]
