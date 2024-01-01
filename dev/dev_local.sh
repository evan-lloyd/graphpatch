#!/bin/bash
. ./dev/graphpatch.sh
docker_run -it --rm $(tailscale_env) graphpatch-dev
