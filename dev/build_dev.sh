#!/bin/bash
docker build . -f dev/containers/dev.Dockerfile --tag graphpatch-dev --platform linux/amd64
