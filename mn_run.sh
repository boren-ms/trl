#!/bin/bash

set -x

code_dir=/root/code/trl
config_file=${code_dir}/orng_conf/sft_grpo_ls_train_n12_err_lr_v0.yaml
cmd="${code_dir}/trl/scripts/grpo_bias.py --config ${config_file} --output_dir ./output"
# python $cmd
# torchrun --nproc_per_node=8 $cmd

host=$(hostname)
target=${host::-1}
echo "Hostname: ${host}"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"

export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}


for i in $(seq 0 $((NUM_NODE-1))); do
    ssh ${target}${i} \
    MASTER_ADDR=${host} MASTER_PORT=2235 NODE_RANK=${i} NNODES=${NUM_NODE} NPROC_PER_NODE=8 \
    nohup /root/.pyenv/versions/3.11.8/bin/torchrun \
    --nproc_per_node=8 \
    --nnodes=${NUM_NODE} \
    --node_rank=${i} \
    --master_addr=${host} \
    --master_port=2235 \
    $cmd > torchrun${i}.log 2>&1 &
done 

tail -n 20 -f torchrun*.log
