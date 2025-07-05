#! /bin/bash
# run_accelerate.sh
set -euo pipefail
# set -x
config_file=${1}
if [[ -z "${config_file}" ]]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

config_file=$(realpath "${config_file}")
export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_storages
region_storages=(
    ["southcentralus"]="orngscuscresco"
    ["westus2"]="orngwus2cresco"
    ["uksouth"]="orngukscresco"
)
# export REGION_CODE=${region_storages[$CLUSTER_REGION]}
export DATA_STORAGE=${region_storages[$CLUSTER_REGION]}
#!/bin/bash
set -x

# model_path=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged/
# echo "Running trl-vllm with ${model_path} with ${VLLM_GPU_NUM} GPUs"
# # CUDA_VISIBLE_DEVICES=4,5,6,7
VLLM_GPU_NUM=1
# trl vllm-serve --model ${model_path} --data-parallel-size ${VLLM_GPU_NUM} --trust-remote-code # for 4 GPUs
export VLLM_SERVER_HOST="localhost"


MAIN_NODE=${RCALL_JOB_NAME}-0
# update the ENV variables in the config file
new_config_file=${config_file}.tmp
envsubst < ${config_file} > ${new_config_file}

HOST=$(hostname)
echo "syncing config: ${MAIN_NODE} -> ${HOST}"
rsync -avz ${MAIN_NODE}:${new_config_file} ${new_config_file}

echo "Working Dir: ${PWD}"

CONFIG_NAME=$(basename "$config_file" | sed 's/\.[^.]*$//')
OUTPUT_DIR=${HOME}/outputs/${CONFIG_NAME}
mkdir -p ${OUTPUT_DIR}

GPUs=$(seq ${VLLM_GPU_NUM} 7 | paste -sd, -)
export CUDA_VISIBLE_DEVICES=${GPUs}
cmd="accelerate launch --num_processes 1 \
trl/scripts/grpo_bias.py --config ${new_config_file} --output-dir ${OUTPUT_DIR}
"
export RANK=0
mkdir -p ${RCALL_LOGDIR}
RANK_LOG_FILE=${RCALL_LOGDIR}/${CONFIG_NAME}_rank_${RANK}.log
echo "Logging to ${RANK_LOG_FILE}"
echo "vLLM Server Host: ${VLLM_SERVER_HOST}" > ${RANK_LOG_FILE}
echo "Running $cmd" >> $RANK_LOG_FILE
# printenv >> $RANK_LOG_FILE
# export WANDB_MODE=offline
# echo "WANDB MODE: ${WANDB_MODE}"
$cmd >> $RANK_LOG_FILE 2>&1
# echo "Sync wandb: ${WANDB_DIR}/wandb/offline-run* "
# wandb sync ${WANDB_DIR}/wandb/offline-run* |tee -a $RANK_LOG_FILE
