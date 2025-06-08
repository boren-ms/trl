#!/bin/bash

set -x

JOB_NAME=${RCALL_JOB_NAME}
export NUM_NODE=${RCALL_INSTANCE_COUNT}
export WANDB_DISABLED="true" 
code_dir=/root/code/trl



config_file=${code_dir}/orng_conf/grpo_bias_debug.yaml
name=grpo_bias_debug


# config_file=${code_dir}/orng_conf/grpo_bias_librispeech_v0.yaml
# name=grpo_bias_v0_n12_b20k

echo "sync ${config_file}"
for i in $(seq 1 $((NUM_NODE-1))); do
    rsync -avz ${config_file} ${JOB_NAME}-${i}:${config_file}
done


export USE_ACCELERATE=false
OUTPUT_DIR=${RCALL_LOGDIR}/${name}

echo "
export OUTPUT_DIR=${OUTPUT_DIR}
" >>~/.bashrc

# bash prepare_orng.sh --force
bash prepare_orng.sh

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${OUTPUT_DIR}  --config ${config_file}