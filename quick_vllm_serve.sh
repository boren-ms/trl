#!/bin/bash 
set -x 

model_path= ${HOME}/data/ckp/hf_models/phi4_mm_bias_merged/

echo "Running trl-vllm with ${model_path} with 4 GPUs"
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve --model ${model_path} --data-parallel-size 4 --trust-remote-code # for 4 GPUs
