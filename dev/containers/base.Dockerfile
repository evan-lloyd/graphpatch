FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /graphpatch
COPY .python-version ./

RUN apt update -y && apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt install -y wget openssh-server curl git tmux

ADD https://astral.sh/uv/0.5.23/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH=/root/.local/bin:/graphpatch/.venv/bin:$PATH \
    GP_MODEL_DIR=/models \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=symlink \
    TERMINFO_DIRS=/etc/terminfo:/lib/terminfo:/usr/share/terminfo
RUN uv python install `head -n 1 .python-version`

# Bake in env vars so they'll be present when we SSH into a remote container
RUN env > /etc/environment && mkdir /models && mkdir /graphpatch/.pytest_cache
RUN echo "cd /graphpatch" >> "/root/.bashrc"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY graphpatch ./graphpatch
RUN uv sync --frozen --group base --group torch25 --all-extras
