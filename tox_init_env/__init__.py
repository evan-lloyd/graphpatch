from tox.plugin import impl
from tox.execute.request import StdinSource
from pathlib import Path
import os


@impl
def tox_on_install(tox_env, arguments, section, of_type):
    if os.getenv("TOX_RUN_COMMANDS_PRE") == "1":
        for cmd in tox_env.conf["commands_pre"]:
            tox_env.execute(cmd=cmd.args, stdin=StdinSource.user_only(), show=True, cwd=Path.cwd())
