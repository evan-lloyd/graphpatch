import torch
from pytest import fixture

from graphpatch import ExtractionOptions, PatchableGraph
from graphpatch.optional.transformers import AutoModel, BitsAndBytesConfig

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
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.float16,
    )


@fixture
def quantized_pretrained_module_inputs(pretrained_module_inputs):
    return pretrained_module_inputs.to(device="cuda", dtype=torch.float16)


@fixture
def mixed_cpu_pretrained_module(pretrained_module_path):
    return AutoModel.from_pretrained(
        pretrained_module_path,
        device_map={
            "model.child_a.a_linear": "cpu",
            "model.child_a.grandchildren_b.0": 0,
            "model.child_a.grandchildren_b.1": "cpu",
            "model.child_a.grandchildren_b.2": 0,
            "model.root_linear": 0,
        },
    )


@fixture
def mixed_cpu_pretrained_module_inputs(pretrained_module_inputs):
    return pretrained_module_inputs


@fixture
def disk_offload_pretrained_module(pretrained_module_path, tmp_path_factory):
    return AutoModel.from_pretrained(
        pretrained_module_path,
        device_map={
            "model.child_a": "disk",
            "model.root_linear": "cpu",
        },
        offload_folder=tmp_path_factory.mktemp("disk_offload"),
    )


@fixture
def disk_offload_pretrained_module_inputs(pretrained_module_inputs):
    return pretrained_module_inputs


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


@fixture
def patchable_mixed_cpu_pretrained_module(
    request, mixed_cpu_pretrained_module, mixed_cpu_pretrained_module_inputs
):
    return PatchableGraph(
        mixed_cpu_pretrained_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        mixed_cpu_pretrained_module_inputs,
    )


@fixture
def patchable_disk_offload_pretrained_module(
    request, disk_offload_pretrained_module, disk_offload_pretrained_module_inputs
):
    return PatchableGraph(
        disk_offload_pretrained_module,
        ExtractionOptions(
            skip_compilation=getattr(request, "param", None) == "opaque",
            error_on_compilation_failure=True,
        ),
        disk_offload_pretrained_module_inputs,
    )
