import torch
from pytest import fixture

from graphpatch import ExtractionOptions, PatchableGraph
from graphpatch.optional.transformers import AutoModel

from .pretrained.test_model_tokenizer import DummyTokenizer


@fixture
def pretrained_tokenizer(pretrained_module_path):
    return DummyTokenizer.from_pretrained(pretrained_module_path)


@fixture
def pretrained_module(pretrained_module_path):
    return AutoModel.from_pretrained(pretrained_module_path)


@fixture
def pretrained_module_inputs(pretrained_tokenizer):
    return pretrained_tokenizer("foo").input_ids


@fixture
def patchable_pretrained_module(request, pretrained_module, pretrained_module_inputs):
    return PatchableGraph(
        pretrained_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        pretrained_module_inputs,
    )


@fixture
def accelerate_pretrained_module(pretrained_module_path):
    return AutoModel.from_pretrained(pretrained_module_path, device_map="auto")


@fixture
def accelerate_pretrained_module_inputs(pretrained_module_inputs):
    return pretrained_module_inputs.to(device="cuda")


@fixture
def patchable_accelerate_pretrained_module(
    request, accelerate_pretrained_module, accelerate_pretrained_module_inputs
):
    return PatchableGraph(
        accelerate_pretrained_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        accelerate_pretrained_module_inputs,
    )


@fixture
def quantized_pretrained_module(pretrained_module_path):
    return AutoModel.from_pretrained(
        pretrained_module_path,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )


@fixture
def quantized_pretrained_module_inputs(pretrained_module_inputs):
    return pretrained_module_inputs.to(device="cuda", dtype=torch.float16)


@fixture
def patchable_quantized_pretrained_module(
    request, quantized_pretrained_module, quantized_pretrained_module_inputs
):
    return PatchableGraph(
        quantized_pretrained_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        quantized_pretrained_module_inputs,
    )
