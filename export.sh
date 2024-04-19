#!/usr/bin/env bash

./build.sh

docker save acouslicai_evaluation:latest | gzip -c > acouslicai_evaluation.tar.gz