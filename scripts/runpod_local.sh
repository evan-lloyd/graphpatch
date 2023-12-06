#!/bin/bash
docker run -it --gpus all --rm -e TAILSCALE_AUTH_KEY=$TAILSCALE_AUTH_KEY \
  -e TAILSCALE_HOST_NAME=graphpatch graphpatch-runpod
