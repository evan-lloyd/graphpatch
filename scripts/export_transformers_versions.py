import subprocess

result = subprocess.run("poetry export -f requirements.txt -E transformers", check=True, shell=True)

print(result.stdout.decode())
