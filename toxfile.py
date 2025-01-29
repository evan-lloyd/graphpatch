from argparse import ArgumentParser

from tox.execute import Outcome
from tox.plugin import impl
from tox.tox_env.register import ToxEnvRegister
from tox_uv._run_lock import UvVenvLockRunner
from tox import run


"""A couple of customizations to tox are added here:
1) tox has no mechanism to dynamically choose the environment runner, so hack something together
that will noop when given the "noop" config, but otherwise acts like uv-venv-lock-runner
2) tox forbids combining "-f" factor command-line options with labels, so add a custom option
"-af" which first computes whatever vanilla tox would have selected with the given options, *then*
adds the given factors on top. For example,
    `tox -m test-20 -af coverage`
adds the "coverage" factor to the default torch 2.0 test environment.
"""


class GPRunner(UvVenvLockRunner):
    """Runner that acts like tox-uv lock runner, but can be configured to noop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _allow_externals(self) -> list[str]:
        if self.conf["noop"]:
            result = super()._allow_externals
            result.append("echo")
            return result
        return super()._allow_externals

    def register_config(self) -> None:
        self.conf.add_config(
            keys=["noop"],
            of_type=bool,
            default=False,
            desc="Should we skip env creation and commands?",
        )
        self.conf.add_config(
            keys=["noop_message"],
            of_type=str,
            default="Skipping environment {envname}",
            desc="Message to print if skipping.",
        )
        super().register_config()

    @staticmethod
    def id() -> str:
        return "gp-runner"

    def create_python_env(self) -> None:
        if self.conf["noop"]:
            return
        super().create_python_env()

    def _setup_env(self) -> None:
        if self.conf["noop"]:
            return
        super()._setup_env()

    def execute(self, cmd, stdin, show=None, cwd=None, run_id="", executor=None) -> Outcome:
        if self.conf["noop"]:
            cmd = ["echo", self.conf["noop_message"].format(envname=self.name)]
        return super().execute(cmd, stdin, show, cwd, run_id, executor)


@impl
def tox_register_tox_env(register: ToxEnvRegister) -> None:
    register.add_run_env(GPRunner)


@impl
def tox_add_option(parser: ArgumentParser) -> None:
    parser.add_argument("-af", dest="add_factor", action="append")


orig_init = run.State.__init__


def state_init(self, options, args):
    orig_init(self, options, args)
    # Add additional factors, if specified, by activating envs with the factor appended, and
    # deactivating the ones originally selected.
    for factor in options.parsed.add_factor or []:
        # This has the side-effect of finalizing all configs, which breaks subsequent operations...
        # so undo that.
        envs = self.envs._defined_envs
        for tox_env in envs.values():
            tox_env.env.conf._final = False
        self.conf.core._final = False
        envs_to_modify = [env for env in self.envs.iter() if f"{env}-{factor}" in envs]
        for env in envs_to_modify:
            self.envs._defined_envs[env].is_active = False
            self.envs._defined_envs[f"{env}-{factor}"].is_active = True


run.State.__init__ = state_init
