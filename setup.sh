#!/bin/bash

set -x

# install system dependencies
ehco "Installing system dependencies"
pip install --upgrade pip
pip install bs4 \
    nvidia-ml-py \
    accelerate \
    datasets \
    transformers==4.46.2 \
    librosa \
    soundfile \
    jiwer \
    wandb \
    backoff \
    fire \
    peft \
    rich \
    tensorboardX \
    tensorboard

MAX_JOBS=20 pip install flash-attn --no-build-isolation
pip install -e . --no-deps

echo "Move data from Azure Blob Storage to local directory"
# move data
region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data"
local_dir="/root/data"

bbb sync --delete --concurrency 32 $remote_dir $local_dir

echo "Data moved successfully to $local_dir"


