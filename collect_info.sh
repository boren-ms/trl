#!/bin/bash


echo "Collecting process information..."
bash ./mpi_bash.sh pgrep -fa grpo_bias | tee procs.txt

echo "Collecting GPUs information..."
bash ./mpi_bash.sh nvidia-smi|grep Default | tee gpus.txt

echo "Collecting logs from all ranks..."
echo "Log directory: ${RCALL_LOGDIR}"

echo 'Writing log' > ranks.txt
for i in $(seq 0 $((RCALL_INSTANCE_COUNT-1))); do
    
    echo "Logging to ${RCALL_JOB_NAME}-${i}" >> ranks.txt
    ssh ${RCALL_JOB_NAME}-${i} "tail -n 30  ${RCALL_LOGDIR}/*/rank_${i}.log" >> ranks.txt
    echo "" >> ranks.txt
done