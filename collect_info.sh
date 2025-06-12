#!/bin/bash


echo "Collecting process information..."
bash ./mpi_bash pgrep -fa grpo_bias | tee proc.txt

echo "Collecting GPUs information..."
bash ./mpi_bash nvidia-smi|grep Default | tee gpus.txt

echo "Collecting logs from all ranks..."
echo "Log directory: ${RCALL_LOGDIR}"
bash ./mpi_bash "tail -n 100 ${RCALL_LOGDIR}/*/rank_*.log" | tee proc.txt
