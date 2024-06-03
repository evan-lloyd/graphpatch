from graphpatch.optional.transformers import PretrainedConfig


class TestModelConfig(PretrainedConfig):
    model_type = "test_model"

    def __init__(self, **kwargs):
        kwargs["max_length"] = 110
        super().__init__(**kwargs)
