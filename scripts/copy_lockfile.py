import sys
import re

[root_dir, env_name, direction] = sys.argv[1:]
torch_version = re.match(r"(lock|update)-(torch\d+)", env_name).group(2)

versioned_filename = f"{root_dir}/.poetry-lockfiles/lock-{torch_version}.lock"
static_filename = "poetry.lock"

if direction == "in":
    in_file_name = versioned_filename
    out_file_name = static_filename
else:
    in_file_name = static_filename
    out_file_name = versioned_filename

with open(out_file_name, "w") as out_file:
    with open(in_file_name, "r") as in_file:
        out_file.write(in_file.read())
