import pytest
import torch

# TODO: add metadata to the request defs somehow to avoid needing to hardcode this?
REQUIRES_GPU = (
    "patchable_mixed_cpu_pretrained_module",
    "patchable_accelerate_pretrained_module",
    "patchable_quantized_pretrained_module",
)
REQUIRES_BITSANDBYTES = ("patchable_quantized_pretrained_module",)
REQUIRES_TRANSFORMERS = (
    "patchable_pretrained_module",
    "patchable_mixed_cpu_pretrained_module",
    "patchable_disk_offload_pretrained_module",
    "patchable_accelerate_pretrained_module",
    "patchable_quantized_pretrained_module",
)
REQUIRES_ACCELERATE = (
    "patchable_mixed_cpu_pretrained_module",
    "patchable_disk_offload_pretrained_module",
    "patchable_accelerate_pretrained_module",
    "patchable_quantized_pretrained_module",
)
# We don't yet have transformer_lens fixtures.
REQUIRES_TRANSFORMER_LENS = ("",)


def _filter_by_test_env(name):
    from graphpatch.optional import (
        accelerate,
        bitsandbytes,
        transformer_lens,
        transformers,
    )

    has_gpu = torch.cuda.device_count() >= 1
    return all(
        [
            name not in REQUIRES_GPU or has_gpu,
            name not in REQUIRES_BITSANDBYTES or bitsandbytes.AVAILABLE,
            name not in REQUIRES_TRANSFORMERS or transformers.AVAILABLE,
            name not in REQUIRES_ACCELERATE or accelerate.AVAILABLE,
            name not in REQUIRES_TRANSFORMER_LENS or transformer_lens.AVAILABLE,
        ]
    )


@pytest.fixture
def all_patchable_graphs(request):
    session = request.session
    fixture_defs = session._fixturemanager._arg2fixturedefs
    return {
        k: v[0].execute(request)
        for k, v in fixture_defs.items()
        if k.startswith("patchable") and _filter_by_test_env(k)
    }
