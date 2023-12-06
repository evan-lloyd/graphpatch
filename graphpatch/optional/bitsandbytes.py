import contextlib
import warnings

try:
    with contextlib.redirect_stdout(None), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bitsandbytes.nn import Linear8bitLt
    AVAILABLE = True
except ImportError:

    class Linear8bitLt:
        pass

    AVAILABLE = False

__all__ = ["Linear8bitLt"]
