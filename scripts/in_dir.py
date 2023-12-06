# Workaround for not being able to "cd foo && bar" in tox commands
import subprocess
import sys

subprocess.run(" ".join(sys.argv[2:]), check=True, shell=True, cwd=sys.argv[1])
