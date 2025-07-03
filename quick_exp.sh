#!/bin/bash 
set -x 

model_path=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged/

echo "Running grpo_bias with ${model_path} on 4 GPUs"
# CUDA_VISIBLE_DEVICES=4,5,6,7 
# wandb offline
# wandb login --relogin --host=https://msaip.wandb.io
# CUDA_VISIBLE_DEVICES=4 accelerate launch trl/scripts/grpo_bias.py --config orng_conf/grpo_bias_debug.yaml
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch trl/scripts/grpo_bias.py --config orng_conf/grpo_bias_debug.yaml