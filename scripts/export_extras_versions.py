import re
import subprocess
import sys

try:
    result = subprocess.run(
        "uv export --all-extras",
        check=True,
        shell=True,
        capture_output=True,
    )
except subprocess.CalledProcessError as exc:
    print(exc.stdout, exc.stderr)
    raise

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
    if (match := re.match(f"({'|'.join(PACKAGE_NAMES)}).+?(?= ;?)", line))
]
out_lines = []
for line in wanted_lines:
    version_requirement_match = re.search(r"python_full_version ([!|<|>|=]+) ('.+?')", line[0])
    sys_platform_match = re.search(r"sys_platform ([!|<|>|=]+) '(.+?)'", line[0])
    out_line = line[1]
    if version_requirement_match:
        out_line += f" ; {version_requirement_match.group(0)}"
    if sys_platform_match:
        out_line += f" {'and' if version_requirement_match else ';'} {sys_platform_match.group(0)}"
    out_lines.append(out_line)


result = ".. code::\n\n" + "\n".join(out_lines) + "\n\n"
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(result)
else:
    print(result)
