FROM graphpatch-base
WORKDIR /graphpatch

ENV POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache

# Needed for testenv-debug
RUN DEBIAN_FRONTEND=noninteractive apt install -y cmake libcairo2-dev python3-gi python3-gi-cairo gir1.2-gtk-4.0 libgirepository1.0-dev

# Add dev packages for remote development/testing
COPY pyproject.toml poetry.lock pytest.ini mypy.ini tox.ini toxfile.py .isort.cfg LICENSE README.md .taplo.toml .python-version .black .flake8 .isort.cfg ./

RUN wget -O - https://tailscale.com/install.sh | sh

RUN mkdir /init && echo "MODEL_DIR=/models" >> /etc/environment
COPY --chmod=777 dev/containers/init/ /init/

COPY tests/ tests/
COPY demos/ demos/
COPY scripts/ scripts/
COPY dev/ dev/
COPY .poetry-lockfiles/ .poetry-lockfiles/

RUN ./dev/uv_install.sh --system

ENTRYPOINT [ "/init/tailscale_init.sh" ]
