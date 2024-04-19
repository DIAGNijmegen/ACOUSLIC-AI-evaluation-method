#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build "$SCRIPTPATH" \
    -t acouslicai_evaluation:v0.2 \
    -t acouslicai_evaluation:latest