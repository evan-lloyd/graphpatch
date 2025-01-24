FROM graphpatch-base
WORKDIR /graphpatch

RUN wget -O - https://tailscale.com/install.sh | sh
COPY --chmod=777 dev/containers/init/ /init/

RUN uv python install

# Add dev packages for remote development/testing
RUN --mount=type=bind,source=dev/external_requirements.txt,target=external_requirements.txt \
  uv tool install tox --overrides external_requirements.txt --with tox-uv
COPY pytest.ini mypy.ini tox.ini .black .flake8 .isort.cfg .taplo.toml ./
RUN uv sync --frozen --all-extras --group testenv-lint --group testenv-format \
  --group testenv-typecheck --group testenv-test --group torch25 --group base --group dev

COPY tests/ tests/
COPY demos/ demos/

ENTRYPOINT [ "/init/tailscale_init.sh" ]
