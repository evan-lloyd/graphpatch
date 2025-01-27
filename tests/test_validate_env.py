import os
import sys

import pytest
import torch

"""Due to having a single lock-file controlling the dependencies that get set up for each test,
a misconfiguration could cause us to end up with an environment different from the one we were
expecting. We should fail the test suite in that case, since we want to guarantee that we are
truly testing every combination of factors."""


def test_validate_tox_factors():
    # From tox factor, formatted as torchXX
    torch_from_env = os.environ["GP_TOX_FACTOR_TORCH"]
    assert torch_from_env[5:] == "".join(torch.__version__.split(".")[:2])

    # From tox factor, formatted as pyXX
    py_from_env = os.environ["GP_TOX_FACTOR_PY"]
    assert py_from_env[2:] == "".join(map(str, sys.version_info[:2]))

    if os.environ["GP_TOX_FACTOR_EXTRA"] == "extranone":
        with pytest.raises(ImportError):
            import transformers
    elif os.environ["GP_TOX_FACTOR_EXTRA"] == "extraall":
        import transformers

        assert transformers is not None
    else:
        assert False, "Invalid tox factor for extra"
