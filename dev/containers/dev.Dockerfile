FROM graphpatch-base
WORKDIR /graphpatch

ENV POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

# Add dev packages for remote development/testing
COPY pyproject.toml poetry.lock pytest.ini mypy.ini tox.ini .black .flake8 .isort.cfg ./
RUN --mount=type=cache,target=$POETRY_CACHE_DIR touch README.md && pip install poetry==1.6.1 \
  && poetry install --no-root --all-extras

RUN wget -O - https://tailscale.com/install.sh | sh

RUN mkdir /init && echo "MODEL_DIR=/models" >> /etc/environment
COPY --chmod=777 dev/containers/init/ /init/

COPY tests/ tests/
COPY demos/ demos/

ENTRYPOINT [ "/init/runpod_init.sh" ]
