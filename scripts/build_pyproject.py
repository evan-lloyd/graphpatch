import sys
import re

[root_dir, env_name, out_name] = sys.argv[1:]
torch_version = re.match(r"(lock|update)-(torch\d+)", env_name).group(2)

with open(out_name, "w") as out_file:
    for filename in [
        f"{root_dir}/.poetry-lockfiles/pyproject.in",
        f"{root_dir}/.poetry-lockfiles/{torch_version}.in",
    ]:
        with open(filename, "r") as in_file:
            out_file.write(in_file.read())
