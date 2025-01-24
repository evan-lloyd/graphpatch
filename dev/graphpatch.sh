#!/bin/bash

function docker_binds() {
    echo "--mount type=bind,src=$(pwd)/graphpatch,dst=/graphpatch/graphpatch \
    --mount type=bind,src=$(pwd)/tests,dst=/graphpatch/tests \
    --mount type=bind,src=$(pwd)/demos,dst=/graphpatch/demos \
    --mount type=bind,src=$(pwd)/dev,dst=/graphpatch/dev \
    --mount type=bind,src=$(pwd)/pyproject.toml,dst=/graphpatch/pyproject.toml \
    --mount type=bind,src=$(pwd)/uv.lock,dst=/graphpatch/uv.lock \
    --mount type=bind,src=$(pwd)/README.md,dst=/graphpatch/README.md \
    --mount type=bind,src=$(pwd)/LICENSE,dst=/graphpatch/LICENSE \
    --mount type=bind,src=$(pwd)/.black,dst=/graphpatch/.black \
    --mount type=bind,src=$(pwd)/.flake8,dst=/graphpatch/.flake8 \
    --mount type=bind,src=$(pwd)/.isort.cfg,dst=/graphpatch/.isort.cfg \
    --mount type=bind,src=$(pwd)/.taplo.toml,dst=/graphpatch/.taplo.toml \
    --mount type=bind,src=$(pwd)/tox.ini,dst=/graphpatch/tox.ini \
    --mount type=bind,src=$(pwd)/pytest.ini,dst=/graphpatch/pytest.ini \
    --mount type=bind,src=$(pwd)/.python-version,dst=/graphpatch/.python-version"
}

function docker_run() {
    docker run --gpus all $(docker_binds) "$@" || docker run $(docker_binds) "$@"
}

function tailscale_env() {
    echo "-e TAILSCALE_AUTH_KEY=$TAILSCALE_AUTH_KEY -e TAILSCALE_HOST_NAME=graphpatch"
}
