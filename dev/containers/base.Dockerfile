FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

WORKDIR /graphpatch

# TODO: this is probably bloating the image size, can we build a "portable" pinned Python in another
# intermediate layer and just copy it over? Or just download a pre-built binary?
RUN apt update -y && apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget build-essential checkinstall \
    libreadline-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev \
    libc6-dev libbz2-dev libffi-dev zlib1g-dev openssh-server curl git liblzma-dev lzma tmux && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz && \
    tar xzf Python-3.10.11.tgz && \
    cd Python-3.10.11 && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make && make install && \
    rm /usr/src/Python-3.10.11.tgz && rm -rf /usr/src/Python-3.10.11 && \
    apt clean

RUN ln -s /usr/local/bin/python3 /usr/local/bin/python

# # TODO: this will probably be actually fixed in poetry 2.0.0, go back to a standard install then
RUN pip3 install git+https://github.com/evan-lloyd/poetry-dep-walk-fix@lock-markers-and-groups3a \
    git+https://github.com/python-poetry/poetry-plugin-export@9016c83b8609844890620def6c6c96e4c1b90ed5 \
    uv==0.4.20

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    MODEL_DIR=/models

# Bake in env vars so they'll be present when we SSH into a remote container
RUN env > /etc/environment && mkdir /models && mkdir /graphpatch/.pytest_cache
RUN echo "cd /graphpatch" >> "/root/.bashrc"

COPY pyproject.toml poetry.lock ./

RUN POETRY_WARNINGS_EXPORT=false poetry export --only main,dev,testenv-test,torch --without-hashes --all-extras -f requirements.txt \
    | uv pip install --link-mode symlink --system -r -

COPY graphpatch ./graphpatch
COPY .python-version README.md ./
