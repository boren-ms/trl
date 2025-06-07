#!/bin/bash

set -x

# login to wandb
export WANDB_API_KEY=cefd6cab73652e59732d3cfc4f776050f88f08f8
export WANDB_ORGANIZATION=https://msaip.wandb.io
# wandb login --relogin --host=https://msaip.wandb.io

conf=sft_grpo_debug
output_dir=${RCALL_LOGDIR}/${conf}


bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${output_dir} \
     --config orng_conf/${conf}.yaml