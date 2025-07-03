#!/bin/bash

set -x
export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_map
region_map=(
    ["southcentralus"]="scus"
    ["westus2"]="wus2"
    ["uksouth"]="uks"
)   
export REGION_CODE=${region_map[$CLUSTER_REGION]}

REMOTE_DIR="az://orng${REGION_CODE}cresco/data/boren/data"
# REMOTE_DIR="az://orngwus2cresco/data/boren/data"
LOCAL_DIR="${HOME}/data"


rel_dirs=(
    "gsm8k"
    "ckp/hf_models/Qwen2.5-0.5B-Instruct"
    "ckp/hf_models/phi4_mm_bias_merged"
    "ckp/hf_models/phi4_mm_bias"
    "LibriSpeech/test-clean"
    "LibriSpeech/train-clean-360/115/122944"
)

for rel_dir in "${rel_dirs[@]}"; do
    bbb sync --concurrency 64  "$REMOTE_DIR/$rel_dir" "$LOCAL_DIR/$rel_dir"
done

rel_files=(
   LibriSpeech/ls_30k_shuf.tsv
   LibriSpeech/debug.tsv
)
for rel_file in "${rel_files[@]}"; do
    bbb cp "$REMOTE_DIR/$rel_file" "$LOCAL_DIR/$rel_file"
done