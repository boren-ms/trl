#!/bin/bash


echo "Collecting process information..."
bash ./mpi_bash.sh pgrep -fa grpo_bias | tee procs.txt

echo "Collecting GPUs information..."
bash ./mpi_bash.sh nvidia-smi|grep Default | tee gpus.txt

echo "Collecting logs from all ranks..."
echo "Log directory: ${OUTPUT_DIR}"
bash ./mpi_bash.sh "tail -n 100  ${OUTPUT_DIR}/rank_*.log" |tee rank_logs.txt

# kill all grpo_bias process from all nodes
# pgrep -f grpo_bias | grep -v mpirun | xargs kill 