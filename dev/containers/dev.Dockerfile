FROM graphpatch-base
WORKDIR /graphpatch

# Add dev packages for remote development/testing
RUN --mount=type=bind,source=dev/external_requirements.txt,target=external_requirements.txt \
  uv tool install tox --overrides external_requirements.txt --with tox-uv
COPY pytest.ini mypy.ini tox.ini .black .flake8 .isort.cfg .taplo.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --all-extras --group testenv-lint --group testenv-format \
  --group testenv-typecheck --group testenv-test --group torch25 --group base --group dev

RUN wget -O - https://tailscale.com/install.sh | sh

COPY --chmod=777 dev/containers/init/ /init/

COPY tests/ tests/
COPY demos/ demos/

ENTRYPOINT [ "/init/tailscale_init.sh" ]
