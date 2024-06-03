import contextlib
import warnings

try:
    with contextlib.redirect_stdout(None), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from bitsandbytes import MatmulLtState, matmul
        from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
        from bitsandbytes.nn import Linear8bitLt
    AVAILABLE = True
except ImportError:

    class Linear8bitLt:
        pass

    class MatmulLtState:
        pass

    def _undef(*args, **kwargs):
        raise NotImplementedError(
            "You seem to have encountered an error with graphpatch's optional dependency logic."
            " To work around, you can try making sure that the module bitsandbytes is available."
        )

    get_tile_inds = _undef
    undo_layout = _undef
    matmul = _undef

    AVAILABLE = False

__all__ = ["Linear8bitLt", "get_tile_inds", "undo_layout", "MatmulLtState", "matmul"]
