#!/bin/bash

# set -x
echo "Tail all logs : ${RCALL_LOGDIR}"
bash mpi_bash.sh 'tail -n 20 -f ${RCALL_LOGDIR}/*.log'