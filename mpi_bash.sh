#!/bin/bash

set -x

echo "Job: ${JOB_NAME}"
echo "Node:${NUM_NODE}, GPU:${NUM_GPU}"

# generate hostfile
for i in $(seq 0 $((NUM_NODE-1))); do echo "${JOB_NAME}-$i"; done > "hostfile"

# mpirun launch
mpirun -f "hostfile" -np ${NUM_NODE} bash -c "$@"
# export PMI_SIZE=$NUM_NODE
# for i in $(seq 0 $((NUM_NODE-1))); do
#     export PMI_RANK=${i}
#     ssh ${JOB_NAME}-${i} bash run_dist.sh $@
# done


# for i in $(seq 0 $((NUM_NODE-1))); do     ssh "${JOB_NAME}-${i}" sudo hostname;pgrep -fa bias; done