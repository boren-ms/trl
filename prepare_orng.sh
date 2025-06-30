#%!/bin/bash
set -x 

export JOB_NAME=${RCALL_JOB_NAME}
export NUM_NODE=${RCALL_INSTANCE_COUNT}
export NUM_GPU=${RCALL_NUM_GPU}

export CODE_DIR=/root/code/trl
export DATA_DIR=/root/data
echo "Preparing ENV and DATA for ORNG job..."
echo "Job: ${JOB_NAME}"
echo "Node:${NUM_NODE}, GPU:${NUM_GPU}"

FORCE="false"
if [ "$1" == "--force" ]; then
    FORCE="true"
fi

if [ "${PREPARED_ENV}" != "true" ] || [ "${FORCE}" == "true" ]; then
    echo "Preparing trl environment"
    # for i in $(seq 0 $((NUM_NODE-1))); do
    #     echo "Remotely run ${CODE_DIR}/install.sh on ${JOB_NAME}-${i}"
    #     ssh ${JOB_NAME}-${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${CODE_DIR}/install.sh > ${CODE_DIR}/install.log 2>&1 & "
    # done
    bash mpi_bash.sh "rsync -avz  ${JOB_NAME}-0:${CODE_DIR}/* ${CODE_DIR}/"
    bash mpi_bash.sh "bash ${CODE_DIR}/install.sh"
    export PREPARED_ENV="true"
    echo "export PREPARED_ENV=true" >> ~/.bashrc
fi
if [ "${PREPARED_DATA}" != "true" ] || [ "${FORCE}" == "true" ]; then
    echo "Preparing data"
    remote_dir="az://orng${REGION_CODE}cresco/data/boren/data"
    # sync remote output dir 
    for i in $(seq 0 2); do
        echo "[${i}]th Syncing remote ${RCALL_BLOB_LOGDIR}"
        bbb sync --concurrency 128 ${RCALL_BLOB_LOGDIR} $RCALL_LOGDIR
        bbb sync --concurrency 128 $remote_dir/ckp/phi4_mm_bias $DATA_DIR/ckp/phi4_mm_bias
    done
    # bbb sync --concurrency 128 $remote_dir/LibriSpeech/${REGION_CODE}_tsv $DATA_DIR/LibriSpeech/${REGION_CODE}_tsv
    echo "Data moved successfully to $DATA_DIR"

    bash mpi_bash.sh " rsync -avz ${JOB_NAME}-0:$DATA_DIR/* $DATA_DIR/"
    bash mpi_bash.sh " rsync -avz ${JOB_NAME}-0:$RCALL_LOGDIR/* $RCALL_LOGDIR/"
    # for i in $(seq 1 $((NUM_NODE-1))); do
    #     echo "Move data to ${JOB_NAME}-${i}"
    #     rsync -avz $DATA_DIR ${JOB_NAME}-${i}:$(dirname $DATA_DIR) > rsync_data_${i}.log 2>&1 
    #     rsync -avz $RCALL_LOGDIR ${JOB_NAME}-${i}:$(dirname $RCALL_LOGDIR) > rsync_output_${i}.log 2>&1 
    # done

    export PREPARED_DATA="true"
    echo "export PREPARED_DATA=true" >> ~/.bashrc
fi

echo "Preparation complete. Starting ORNG job..."
