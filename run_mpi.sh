#!/bin/bash

set -x

JOB_NAME=${RCALL_JOB_NAME}
echo "Job: ${JOB_NAME}"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"

export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}
export MASTER_ADDR=${RCALL_HOSTNAME}
export MASTER_PORT=1235

# generate hostfile
for i in $(seq 0 $((NUM_NODE-1))); do echo "${JOB_NAME}-$i"; done > "hostfile"

# mpirun launch
mpirun -f "hostfile" -np ${NUM_NODE} \
    bash run_dist.sh $@

# export PMI_SIZE=$NUM_NODE
# for i in $(seq 0 $((NUM_NODE-1))); do
#     export PMI_RANK=${i}
#     ssh ${JOB_NAME}-${i} bash run_dist.sh $@
# done


# for i in $(seq 0 $((NUM_NODE-1))); do     ssh "${JOB_NAME}-${i}" sudo hostname;pgrep -fa bias; done