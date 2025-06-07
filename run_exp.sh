#!/bin/bash

set -x

# login to wandb
# export WANDB_API_KEY=cefd6cab73652e59732d3cfc4f776050f88f08f8
# export WANDB_ORGANIZATION=https://msaip.wandb.io
export WANDB_DISABLED="true" 
# wandb login --relogin --host=https://msaip.wandb.io

conf=sft_grpo_debug
OUTPUT_DIR=${RCALL_LOGDIR}/${conf}
export WANDB_DIR=${OUTPUT_DIR}/wandb

# bash prepare_orng.sh --force
bash prepare_orng.sh

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${OUTPUT_DIR} \
     --config orng_conf/${conf}.yaml