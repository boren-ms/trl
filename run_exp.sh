#!/bin/bash

set -x

# login to wandb
# export WANDB_API_KEY=cefd6cab73652e59732d3cfc4f776050f88f08f8
# export WANDB_ORGANIZATION=https://msaip.wandb.io
export WANDB_DISABLED="true" 
# wandb login --relogin --host=https://msaip.wandb.io


config_file=orng_conf/grpo_bias_debug.yaml
name=grpo_bias_debug

# config_file=orng_conf/grpo_bias_librispeech_v0.yaml
# name=grpo_bias_v0_n12_b20k




OUTPUT_DIR=${RCALL_LOGDIR}/${name}
echo "
export OUTPUT_DIR=${OUTPUT_DIR}
" >>~/.bashrc

# bash prepare_orng.sh --force
bash prepare_orng.sh

bash run_mpi.sh trl/scripts/grpo_bias.py \
     --output_dir ${OUTPUT_DIR} \
     --config ${config_file}