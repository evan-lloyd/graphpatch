import os

import pytest

# No tests in the fixtures directory.
collect_ignore = ["fixtures"]

# Walk fixtures directory to import all fixtures automatically.
pytest_plugins = []
for root, dirs, files in os.walk("tests/fixtures"):
    for dir in dirs:
        if dir.startswith("_"):
            dirs.remove(dir)
    for file in files:
        if file.endswith(".py") and not file.startswith("_"):
            pytest_plugins.append(os.path.join(root, file).replace("/", ".")[:-3])

pytest.register_assert_rewrite("tests.util")
