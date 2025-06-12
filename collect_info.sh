#!/bin/bash


echo "Collecting process information..."
bash ./mpi_bash.sh pgrep -fa grpo_bias | tee proc.txt

echo "Collecting GPUs information..."
bash ./mpi_bash.sh nvidia-smi|grep Default | tee gpus.txt

echo "Collecting logs from all ranks..."
echo "Log directory: ${RCALL_LOGDIR}"
bash ./mpi_bash.sh "tail -n 100 ${RCALL_LOGDIR}/*/rank_*.log" | tee proc.txt
