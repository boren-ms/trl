#!/bin/bash

set -x

# login to wandb
# wandb login --relogin --host=https://msaip.wandb.io


bash launch.sh trl/scripts/grpo_bias.py \
     --output_dir ./output \
     --config orng_conf/sft_grpo_ls_train_n12_err_lr_v0.yaml