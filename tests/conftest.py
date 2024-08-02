import os
import re
import site
import sys
from pdb import Pdb

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


def pytest_addoption(parser, pluginmanager):
    # Convenience argument to add breakpoints based on parsed copy+pasted "remote file URLs" from,
    # eg, pytorch's python source. Super convenient for developing hacks.py!
    parser.addoption("--gpbp", dest="graphpatch_breakpoints", action="append")


# https://stackoverflow.com/a/54564137
class RunningTrace:
    def set_running_trace(self):
        frame = sys._getframe().f_back
        self.botframe = None
        self.setup(frame, None)
        while frame:
            frame.f_trace = self.trace_dispatch
            self.botframe = frame
            frame = frame.f_back
        self.set_continue()
        self.quitting = False
        sys.settrace(self.trace_dispatch)


class ProgrammaticPdb(Pdb, RunningTrace):
    pass


debugger = ProgrammaticPdb()


def pytest_configure(config):
    # https://github.com/pytorch/pytorch/blob/dcfa7702c3ecd8754e8a66bc49142de00c8474ee/torch/_dynamo/source.py#L375
    site_packages_dir = site.getsitepackages()[0]
    for bp in config.option.graphpatch_breakpoints or []:
        match = re.match(
            r"^https://github.com/.+?/blob/.+?/(.+?)#L(\d+),?(.+?)?$",
            bp,
        )
        library_source = f"{site_packages_dir}/{match.group(1)}"
        line_number = int(match.group(2))
        condition = match.group(3)
        debugger.set_break(library_source, line_number, cond=condition)
        print(
            f"Set breakpoint in {library_source}, line {line_number}"
            f" {f'cond={condition}' if condition else ''}"
        )

    if config.option.graphpatch_breakpoints:
        debugger.set_running_trace()
