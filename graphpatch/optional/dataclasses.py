import sys
from dataclasses import field

if sys.version_info >= (3, 10):
    # kw_only added in 3.10
    from dataclasses import dataclass
else:
    # kw_only implementation adapted from
    # https://stackoverflow.com/questions/49908182/how-to-make-keyword-only-fields-with-dataclasses#comment129098154_49911616
    import dataclasses
    from contextlib import contextmanager
    from dataclasses import MISSING, dataclass as _orig_dataclass

    # Re-order fields so non-defaults come first. Would ordinarily be much more complicated than this,
    # but all our dataclasses are kw_only anyway, so we don't care about changing the order.
    @contextmanager
    def monkeypatch_init_fn():
        orig_init_fn = dataclasses._init_fn

        def patched_init_fn(fields, *args, **kwargs):
            fields = sorted(
                fields,
                key=lambda f: f.default is MISSING and f.default_factory is MISSING,
                reverse=True,
            )
            return orig_init_fn(fields, *args, **kwargs)

        dataclasses._init_fn = patched_init_fn
        try:
            yield
        finally:
            dataclasses._init_fn = orig_init_fn

    # Backport good-enough kw_only behavior
    def dataclass(cls=None, *, kw_only=False, **dc_kwargs):
        def make_dataclass(cls):
            with monkeypatch_init_fn():
                dc = _orig_dataclass(cls, **dc_kwargs)
            if not kw_only:
                return dc
            _orig_init = dc.__init__

            def kw_only_init(self, **kwargs):
                _orig_init(self, **kwargs)

            dc.__init__ = kw_only_init
            return dc

        if cls is None:
            return make_dataclass
        return make_dataclass(cls)


__all__ = ["dataclass", "field"]
