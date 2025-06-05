#!/bin/bash

set -x

cmd="trl/scripts/grpo_bias.py --config exp_conf/ls_train_biasing.yaml --output_dir ./output_bias"
python $cmd
# torchrun --nproc_per_node=8 $cmd

