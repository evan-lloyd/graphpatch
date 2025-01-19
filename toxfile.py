from tox.tox_env.api import ToxEnv


def _write_execute_log(*args, **kwargs):
    return


ToxEnv._write_execute_log = _write_execute_log
