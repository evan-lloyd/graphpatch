# Cached poetry dependency installation adapted from
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

FROM python:3.10.11-buster as builder

RUN pip install poetry==1.6.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    MODEL_DIR=/models

# There is an outstanding issue where this isn't actually respected,
# https://github.com/python-poetry/poetry/issues/5906
# POETRY_VIRTUALENVS_OPTIONS_ALWAYS_COPY=1

WORKDIR /graphpatch

COPY pyproject.toml poetry.lock ./

# TODO: Cleaner to set this by env, replace when issue is resolved
RUN poetry config virtualenvs.options.always-copy true
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --only main --no-root

FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as runtime

# TODO: this is probably bloating the image size, can we build a "portable" pinned Python in another
# intermediate layer and just copy it over? Or just download a pre-built binary?
RUN apt update -y && apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget build-essential checkinstall \
    libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev \
    libc6-dev libbz2-dev libffi-dev zlib1g-dev openssh-server curl git liblzma-dev lzma tmux && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
    tar xzf Python-3.10.11.tgz && \
    cd Python-3.10.11 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    make bininstall && \
    rm /usr/src/Python-3.10.11.tgz && rm -rf /usr/src/Python-3.10.11 && \
    apt clean

WORKDIR /graphpatch
ENV VIRTUAL_ENV=/graphpatch/.venv \
    PATH="/graphpatch/.venv/bin:$PATH"

# Bake in env vars so they'll be present when we SSH into a remote container
RUN env > /etc/environment && mkdir /models && mkdir /graphpatch/.pytest_cache
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
RUN cp /usr/local/bin/python3.10 /graphpatch/.venv/bin/python
RUN echo "cd /graphpatch" >> "/root/.bashrc"

COPY graphpatch ./graphpatch
