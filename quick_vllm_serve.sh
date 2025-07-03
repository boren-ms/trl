#!/bin/bash 
set -x 

model_path=${HOME}/data/ckp/hf_models/phi4_mm_bias_merged/
N=8
echo "Running trl-vllm with ${model_path} with ${N} GPUs"
# CUDA_VISIBLE_DEVICES=4,5,6,7 
trl vllm-serve --model ${model_path} --data-parallel-size ${N} --trust-remote-code # for 4 GPUs
