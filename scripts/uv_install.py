import subprocess
import sys

export_command = sys.argv[1]
uv_command = sys.argv[2]

requirements = subprocess.run(export_command, check=True, shell=True, capture_output=True)

subprocess.run(
    uv_command,
    stdout=subprocess.PIPE,
    input=requirements.stdout,
    check=True,
    shell=True,
    capture_output=False,
)
