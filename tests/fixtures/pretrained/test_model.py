import torch
from pytest import fixture

from graphpatch.optional.transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
)

from ..nested_module import NestedModule
from .test_model_config import TestModelConfig
from .test_model_tokenizer import DummyTokenizer


class TestModel(PreTrainedModel):
    config_class = TestModelConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        self.model = NestedModule()

    def forward(self, input_ids, attention_mask=None):
        # Make a fake "embedding" of the input ids
        embedding = (
            input_ids.view((input_ids.shape[0], 1, 100))
            .repeat((1, 100, 1))
            .to(self.config.torch_dtype)
        )
        return self.model(embedding)


@fixture(scope="session")
def pretrained_module_path(tmp_path_factory):
    # TODO: pyfakefs might offer speedups, if we can get it working with Safetensors' rust
    # implementation (or easily swap for real fs for tests that need it)
    config = TestModelConfig()
    model = TestModel(config)

    # Constrain weight magnitude so quantization tests are less flaky
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            torch.nn.init.uniform_(module.bias, -0.01, 0.01)

    model.apply(init_weights)
    save_path = tmp_path_factory.mktemp("models") / "test_model"
    config.save_pretrained(save_path)
    model.save_pretrained(save_path)
    AutoConfig.register("test_model", TestModelConfig)
    AutoModel.register(TestModelConfig, TestModel)
    AutoTokenizer.register(TestModelConfig, DummyTokenizer)
    open(save_path / "dummy.model", "w").write("dummy")
    return save_path
