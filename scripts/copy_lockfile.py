import sys
import re

[root_dir, env_name] = sys.argv[1:]
torch_version = re.match(r"(lock|update)-(torch\d+)", env_name).group(2)

with open(f"{root_dir}/.poetry-lockfiles/lock-{torch_version}.lock", "w") as out_file:
    with open("poetry.lock", "r") as in_file:
        out_file.write(in_file.read())
