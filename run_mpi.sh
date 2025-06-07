#!/bin/bash

set -x

JOB_NAME=${RCALL_JOB_NAME}
echo "Job: ${JOB_NAME}"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"

export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}
export MASTER_ADDR=${RCALL_HOSTNAME}
export MASTER_PORT=12345

# generate hostfile
for i in $(seq 0 $((NUM_NODE-1))); do echo "${JOB_NAME}-$i"; done > "hostfile"

# mpirun launch
mpirun --f "hostfile" -np ${NUM_NODE} \
    bash run_dist.sh $@

# Copy the necessary scripts to each node
# for i in $(seq 1 $((NUM_NODE-1))); do
#     scp /root/code/trl/run_dist.sh ${JOB_NAME}-$i:/root/code/trl/run_dist.sh
#     scp /root/code/trl/trl/scripts/grpo_bias.py ${JOB_NAME}-$i:/root/code/trl/trl/scripts/grpo_bias.py
# done
