#!/bin/bash

set -x

TARGET=$(hostname| sed 's/.$//')
echo "Hostname: ${TARGET}0"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"
export NUM_NODE=${RCALL_INSTANCE_COUNT}

hostfile= "hosts"
if [ ! -f ${hostfile} ]; then
    for i in $(seq 0 $((NUM_NODE-1))); do echo "${TARGET}$i"; done > ${hostfile}
fi

export NUM_GPU=${RCALL_NUM_GPU}
export MASTER_ADDR=${TARGET}0
export MASTER_PORT=12345

mpirun \
    --hostfile ${hostfile} \
    -np ${NUM_NODE} \
    bash run_dist.sh $@

# Copy the necessary scripts to each node
# for i in $(seq 1 $((NUM_NODE-1))); do
#     scp /root/code/trl/run_dist.sh ${TARGET}$i:/root/code/trl/run_dist.sh
#     scp /root/code/trl/trl/scripts/grpo_bias.py ${TARGET}$i:/root/code/trl/trl/scripts/grpo_bias.py
# done
