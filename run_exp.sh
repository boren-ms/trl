#!/bin/bash

set -x

export JOB_NAME=${RCALL_JOB_NAME}
export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}

export CODE_DIR=/root/code/trl
export DATA_DIR=/root/data
export USE_ACCELERATE=false
export WANDB_DISABLED=true

# export EXP_CONFIG=${CODE_DIR}/orng_conf/grpo_bias_debug.yaml
# export EXP_NAME=grpo_bias_debug

export EXP_CONFIG=${CODE_DIR}/orng_conf/grpo_bias_librispeech_b100k_v1.yaml
export EXP_NAME=grpo_bias_v0_n12_b20k



echo "sync ${EXP_CONFIG}"
for i in $(seq 1 $((NUM_NODE-1))); do
    rsync -avz ${EXP_CONFIG} ${JOB_NAME}-${i}:${EXP_CONFIG}
done

export OUTPUT_DIR=${RCALL_LOGDIR}/${EXP_NAME}
mkdir -p ${OUTPUT_DIR}

# bash prepare_orng.sh --force
bash prepare_orng.sh

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${OUTPUT_DIR}  --config ${EXP_CONFIG}