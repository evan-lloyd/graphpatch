FROM graphpatch-base
WORKDIR /graphpatch

ENV POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_IN_PROJECT=1 \
  POETRY_VIRTUALENVS_CREATE=1 \
  POETRY_CACHE_DIR=/tmp/poetry_cache \
  PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo "PYENV_ROOT=$PYENV_ROOT" >> /etc/environment \
  && echo "PATH=$PATH" >> /etc/environment \
  && echo "GP_MODEL_DIR=/models" >> /etc/environment

RUN mkdir /init 
RUN curl -fsSL https://tailscale.com/install.sh | sh
RUN wget -O - https://pyenv.run | bash
RUN pyenv install -s

# Add dev packages for remote development/testing
COPY pyproject.toml poetry.lock pytest.ini mypy.ini tox.ini .black .flake8 .isort.cfg ./
RUN --mount=type=cache,target=$POETRY_CACHE_DIR pip install poetry==1.8.3 \
  && poetry install --no-root --all-extras

COPY --chmod=777 dev/containers/init/ /init/
COPY tests/ tests/
COPY demos/ demos/

ENTRYPOINT [ "/init/tailscale_init.sh" ]
