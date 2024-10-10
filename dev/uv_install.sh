POETRY_WARNINGS_EXPORT=false poetry export --all-groups --without-hashes --all-extras -f requirements.txt \
  | uv pip install --link-mode symlink $1 -r -
