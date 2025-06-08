#! /bin/bash
# dist_train.sh
set -euo pipefail

wandb offline
# printenv

cmd="
torchrun \
    --nnodes=${PMI_SIZE} \
    --node_rank ${PMI_RANK} \
    --nproc-per-node=${NUM_GPU} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    $@
"
# cmd="
# accelerate launch \
#     --num_processes $(($PMI_SIZE*$NUM_GPU)) \
#     --num_machines ${PMI_SIZE} \
#     --machine_rank ${PMI_RANK} \
#     --main_process_ip ${MASTER_ADDR} \
#     --main_process_port ${MASTER_PORT} \
#     $@
# "
log_file=rank_${PMI_RANK}.log
echo "Running $cmd" > $log_file
# printenv >> $log_file
$cmd >> $log_file 2>&1 