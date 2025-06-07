#!/bin/bash

set -x

# login to wandb
# wandb login --relogin --host=https://msaip.wandb.io

conf=sft_grpo_debug

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ./output/${conf} \
     --config orng_conf/${conf}.yaml