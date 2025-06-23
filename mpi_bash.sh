#!/bin/bash

# set -x

N=${RCALL_INSTANCE_COUNT}
# generate hostfile
for i in $(seq 0 $((N-1))); do echo "${RCALL_JOB_NAME}-$i"; done > "hostfile"

# mpirun launch
mpirun -l -f "hostfile" bash -c "$*" 