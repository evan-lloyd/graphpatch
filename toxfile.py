"""Add a couple of useful tweaks to tox. Firstly, support a TOX_RUN_COMMANDS_PRE environment variable,
so that the user can initialize tox environments without running the tests with the combination
TOX_RUN_COMMANDS_PRE tox run -e {toxenv} --notest
This is needed because we set up poetry in our commands_pre step, which is ordinarily skipped with
--notest.
Secondly, use a vendored tox-ignore-env-name-mismatch to allow tox envs to share the same virtual
env and save a TON of space and time.
"""

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple

from tox.execute.request import StdinSource
from tox.plugin import impl
from tox.tox_env.api import ToxEnv
from tox.tox_env.info import Info
from tox.tox_env.python.virtual_env.runner import VirtualEnvRunner
from tox.tox_env.register import ToxEnvRegister
from tox.config.of_type import ConfigConstantDefinition


@impl
def tox_add_env_config(env_conf, state):
    env_conf.add_config("env_dir_replace", str, None, "")
    if env_conf["env_dir_replace"] is None:
        return
    replacements = env_conf["env_dir_replace"].split("\n")
    assert len(replacements) % 2 == 0
    for target, replacement in zip(replacements[:-1], replacements[1:]):
        new_value = Path(re.sub(target, replacement, str(env_conf["env_dir"])))
        env_conf._defined["env_dir"] = ConfigConstantDefinition(("env_dir"), "", new_value)


@impl
def tox_on_install(tox_env, arguments, section, of_type):
    if os.getenv("TOX_RUN_COMMANDS_PRE") == "1":
        for cmd in tox_env.conf["commands_pre"]:
            tox_env.execute(cmd=cmd.args, stdin=StdinSource.user_only(), show=True, cwd=Path.cwd())


"""
tox-ignore-env-name-mismatch, adapted from:
https://github.com/masenf/tox-ignore-env-name-mismatch/blob/ebed15982e30840767d6393913ad507c5d8f5642/src/tox_ignore_env_name_mismatch.py

MIT License
Copyright (c) 2023 Masen Furer
"""


class FilteredInfo(Info):
    """Subclass of Info that optionally filters specific keys during compare()."""

    def __init__(
        self,
        *args: Any,
        filter_keys: Optional[Sequence[str]] = None,
        filter_section: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        :param filter_keys: key names to pop from value
        :param filter_section: if specified, only pop filter_keys when the compared section matches

        All other args and kwargs are passed to super().__init__
        """
        self.filter_keys = filter_keys
        self.filter_section = filter_section
        super().__init__(*args, **kwargs)

    @contextmanager
    def compare(
        self,
        value: Any,
        section: str,
        sub_section: Optional[str] = None,
    ) -> Iterator[Tuple[bool, Optional[Any]]]:
        """Perform comparison and update cached info after filtering `value`."""
        if self.filter_section is None or section == self.filter_section:
            try:
                value = value.copy()
            except AttributeError:  # pragma: no cover
                pass
            else:
                for fkey in self.filter_keys or []:
                    value.pop(fkey, None)
        with super().compare(value, section, sub_section) as rv:
            yield rv


class IgnoreEnvNameMismatchVirtualEnvRunner(VirtualEnvRunner):
    """EnvRunner that does NOT save the env name as part of the cached info."""

    @staticmethod
    def id() -> str:
        return "ignore_env_name_mismatch"

    @property
    def cache(self) -> Info:
        """Return a modified Info class that does NOT pass "name" key to `Info.compare`."""
        return FilteredInfo(
            self.env_dir,
            filter_keys=["name", "type"],
            filter_section=ToxEnv.__name__,
        )


@impl
def tox_register_tox_env(register: ToxEnvRegister) -> None:
    """tox4 entry point: add IgnoreEnvNameMismatchVirtualEnvRunner to registry."""
    register.add_run_env(IgnoreEnvNameMismatchVirtualEnvRunner)
