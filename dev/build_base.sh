#!/bin/bash
docker build . -f dev/containers/base.Dockerfile --tag graphpatch-base --target runtime
