#!/bin/bash

set -x

export JOB_NAME=${RCALL_JOB_NAME}
export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}

export CODE_DIR=/root/code/trl
export DATA_DIR=/root/data
export USE_ACCELERATE=false
export WANDB_MODE=offline

# export EXP_CONFIG=${CODE_DIR}/orng_conf/grpo_bias_debug.yaml
# export EXP_NAME=grpo_bias_debug

export EXP_NAME=grpo_bias_ls_mix_30k

#update the ENV variables in the config file
export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_map
region_map=(
    ["southcentralus"]="scus"
    ["westus2"]="wus2"
    ["uksouth"]="uks"
)   
export REGION_CODE=${region_map[$CLUSTER_REGION]}
envsubst < ${CODE_DIR}/orng_conf/${EXP_NAME}.yaml > ${CODE_DIR}/orng_conf/${EXP_NAME}_tmp.yaml
export EXP_CONFIG=${CODE_DIR}/orng_conf/${EXP_NAME}_tmp.yaml

echo "sync ${EXP_CONFIG}"
for i in $(seq 1 $((NUM_NODE-1))); do
    rsync -avz ${EXP_CONFIG} ${JOB_NAME}-${i}:${EXP_CONFIG}
done
export OUTPUT_DIR=${RCALL_LOGDIR}/${EXP_NAME}
# bash prepare_orng.sh --force
bash prepare_orng.sh

bash run_mpi.sh trl/scripts/grpo_bias.py --output_dir ${OUTPUT_DIR}  --config ${EXP_CONFIG}