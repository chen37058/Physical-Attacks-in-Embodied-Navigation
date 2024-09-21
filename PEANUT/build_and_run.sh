#!/usr/bin/env bash

DOCKER_NAME="peanut_attack"

DOCKER_BUILDKIT=1 docker build . --build-arg INCUBATOR_VER=$(date +%Y%m%d-%H%M%S) --file peanut.Dockerfile -t ${DOCKER_NAME}

docker run \
    -v $(pwd):/PEANUT \
    -v /home/disk1/DATASET/Habitat/habitat-challenge-data:/PEANUT/habitat-challenge-data \
    --gpus='all' \
    --ipc=host \
    ${DOCKER_NAME}