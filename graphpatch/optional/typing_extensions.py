import sys

if sys.version_info >= (3, 10):
    # Version 3.10: https://peps.python.org/pep-0647/
    from typing import TypeAlias, TypeGuard
else:
    try:
        from typing_extensions import TypeAlias, TypeGuard
    except ImportError:
        TypeGuard = None
        TypeAlias = None

__all__ = ["TypeAlias", "TypeGuard"]
