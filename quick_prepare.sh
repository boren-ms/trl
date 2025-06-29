#!/bin/bash

set -x

REMOTE_DIR="az://orngwus2cresco/data/boren/data"
LOCAL_DIR="/home/boren/data/"


rel_paths=(
    "ckp/Qwen/models--Qwen--Qwen2.5-0.5B-Instruct"   
    "gsm8k"
)

for rel_path in "${rel_paths[@]}"; do
    bbb sync "$REMOTE_DIR/$rel_path" "$LOCAL_DIR/$rel_path"
done