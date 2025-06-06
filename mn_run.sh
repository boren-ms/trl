#!/bin/bash

set -x

code_dir=/root/code/trl
config_file=${code_dir}/orng_conf/sft_grpo_ls_train_n12_err_lr_v0.yaml
cmd="${code_dir}/trl/scripts/grpo_bias.py --config ${config_file} --output_dir ./output"
# python $cmd

# login to wandb
# export WANDB_API_KEY=your_wandb_api_key_here
wandb login --relogin --host=https://msaip.wandb.io
# setup accelerate
accelerate config
accelerate test

accelerate launch $cmd 2>&1 | tee accelerate.log
exit 0

host=$(hostname)
target=${host::-1}
num_nodes=1
N=0 #last node

for i in $(seq 0 $N); do
    ssh ${target}${i} \
    MASTER_ADDR=${host} MASTER_PORT=1235 NODE_RANK=${i} NNODES=${num_nodes} NPROC_PER_NODE=8 \
    nohup /root/.pyenv/versions/3.11.8/bin/torchrun \
    --nproc_per_node=8 \
    --nnodes=${num_nodes} \
    --node_rank=${i} \
    --master_addr=${host} \
    --master_port=1235 \
    $cmd > torchrun${i}.log 2>&1 &
done 

tail -n 20 -f torchrun*.log

# ps aux|grep bias
# ps aux|grep bias|awk '{print $2}'|xargs kill -9 