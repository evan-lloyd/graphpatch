import re
import subprocess
import sys

result = subprocess.run(
    "poetry export -f requirements.txt --all-extras", check=True, shell=True, capture_output=True
)

PACKAGE_NAMES = (
    "accelerate",
    "bitsandbytes",
    "transformers",
    "numpy",
    "sentencepiece",
    "transformer-lens",
)

wanted_lines = [
    (line, f"\t{match.group(0)}")
    for line in result.stdout.decode().split("\n")
    if (match := re.match(f"({'|'.join(PACKAGE_NAMES)}).+?(?= ;)", line))
]
out_lines = []
for line in wanted_lines:
    if "numpy" in line[1]:
        if 'python_version < "3.9"' in line[0]:
            out_lines.append(f"{line[1]} (Python 3.8)")
        elif 'python_version < "3.10"' in line[0]:
            out_lines.append(f"{line[1]} (Python 3.9)")
        else:
            out_lines.append(f"{line[1]} (later Python versions)")
    else:
        out_lines.append(line[1])

result = ".. code::\n\n" + "\n".join(out_lines) + "\n\n"
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(result)
else:
    print(result)
