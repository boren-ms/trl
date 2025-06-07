#!/bin/bash

set -x

# login to wandb
# wandb login --relogin --host=https://msaip.wandb.io

host=$(hostname)
task=${host%-*}
output_dir=/root/results/${task}/${conf}

conf=sft_grpo_debug

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${output_dir} \
     --config orng_conf/${conf}.yaml