#! /bin/bash
# dist_train.sh
set -euo pipefail

# printenv
USE_ACCELERATE=${USE_ACCELERATE:-false}

if [ "$USE_ACCELERATE" = true ]; then
    cmd="
    accelerate launch \
        --num_processes $(($PMI_SIZE*$NUM_GPU)) \
        --num_machines ${PMI_SIZE} \
        --machine_rank ${PMI_RANK} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        $@
    "
else
    cmd="
    torchrun \
        --nnodes=${PMI_SIZE} \
        --node_rank ${PMI_RANK} \
        --nproc-per-node=${NUM_GPU} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        $@
    "
fi

mkdir -p ${OUTPUT_DIR}
RANK_LOG_FILE=${OUTPUT_DIR}/rank_${PMI_RANK}.log
echo "Logging to ${RANK_LOG_FILE}"
echo "Running $cmd" > $RANK_LOG_FILE
# printenv >> $RANK_LOG_FILE
$cmd >> $RANK_LOG_FILE 2>&1 