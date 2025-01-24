#!/bin/bash
docker build . -f dev/containers/base.Dockerfile --tag graphpatch-base --platform linux/amd64
