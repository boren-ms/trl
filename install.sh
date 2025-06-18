#!/bin/bash

set -x
# install system dependencies
echo "Installing system dependencies"
pip uninstall -y torch torchvision torchaudio transformers flash-attn
pip install --upgrade pip
pip install bs4 \
    nvidia-ml-py \
    accelerate \
    datasets \
    torch==2.5.0 \
    torchvision==0.20.0 \
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
    tensorboard \
    accelerate

MAX_JOBS=20 pip install flash-attn --no-build-isolation
pip install -e /root/code/trl --no-deps
