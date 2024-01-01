#!/bin/bash
docker build . -f dev/containers/test.Dockerfile --tag graphpatch-test --platform linux/amd64
