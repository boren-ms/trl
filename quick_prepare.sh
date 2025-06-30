#!/bin/bash

set -x

REMOTE_DIR="az://orngwus2cresco/data/boren/data"
LOCAL_DIR="/home/boren/data/"


rel_dirs=(
    "ckp/Qwen/models--Qwen--Qwen2.5-0.5B-Instruct"   
    "gsm8k",
    "ckp/hf_models/phi4_mm_bias_merged",
    "ckp/hf_models/phi4_mm_bias",
    "LibriSpeech/train-clean-360/115/122944/115-122944"
)

for rel_dir in "${rel_dirs[@]}"; do
    bbb sync "$REMOTE_DIR/$rel_dir" "$LOCAL_DIR/$rel_dir"
done

rel_files=(
   LibriSpeech/ls_30k_shuf.tsv
   LibriSpeech/debug.tsv
)
for rel_file in "${rel_files[@]}"; do
    bbb cp "$REMOTE_DIR/$rel_file" "$LOCAL_DIR/$rel_file"
done