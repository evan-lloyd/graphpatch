try:
    from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module

    AVAILABLE = True
except ImportError:

    class ModelHook:
        pass

    def add_hook_to_module(*args, **kwargs):
        pass

    def remove_hook_from_module(*args, **kwargs):
        pass

    AVAILABLE = False

__all__ = ["ModelHook", "add_hook_to_module", "remove_hook_from_module"]
