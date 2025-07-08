#!/bin/bash

# set -x
export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_storages
region_storages=(
    ["southcentralus"]="orngscuscresco"
    ["westus2"]="orngwus2cresco"
    ["uksouth"]="orngukscresco"
)   
export DATA_STORAGE=${region_storages[$CLUSTER_REGION]}

REMOTE_DIR="az://${DATA_STORAGE}/data/boren/data"
# REMOTE_DIR="az://orngwus2cresco/data/boren/data"
# REMOTE_DIR="az://orngscuscresco/data/boren/data"
LOCAL_DIR="${HOME}/data"


rel_dirs=(
    # "gsm8k"
    # "ckp/hf_models/Qwen2.5-0.5B-Instruct"
    "ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf_merged"
    "ckp/hf_models/phi4_mm_bias_merged"
    # "ckp/hf_models/phi4_mm_bias"
    "librispeech_biasing/words"
    "librispeech_biasing/ref"
    "LibriSpeech/test-clean"
    "LibriSpeech/train-clean-360/115/122944"
)

for rel_dir in "${rel_dirs[@]}"; do
    echo "Syncing directory: $rel_dir"
    bbb sync --concurrency 64  "$REMOTE_DIR/$rel_dir" "$LOCAL_DIR/$rel_dir"
done

rel_files=(
   LibriSpeech/ls_30k_shuf.tsv
   LibriSpeech/debug.tsv
)
for rel_file in "${rel_files[@]}"; do
    echo "Syncing file: $rel_file"
    bbb cp "$REMOTE_DIR/$rel_file" "$LOCAL_DIR/$rel_file"
done