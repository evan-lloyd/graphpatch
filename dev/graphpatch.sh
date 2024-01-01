#!/bin/bash

function docker_volumes() {
    echo "-v $(pwd):/graphpatch -v /graphpatch/.venv -v /graphpatch/.tox -v /graphpatch/.pytest_cache"
}

function docker_run() {
    docker run --gpus all $(docker_volumes) "$@" || docker run $(docker_volumes) "$@"
}

function tailscale_env() {
    echo "-e TAILSCALE_AUTH_KEY=$TAILSCALE_AUTH_KEY -e TAILSCALE_HOST_NAME=graphpatch"
}
