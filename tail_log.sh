#!/bin/bash

# set -x
echo "Tail all logs : ${RCALL_LOGDIR}"
python mpi_run.py 'tail -n 100 -f ${RCALL_LOGDIR}/*.log'