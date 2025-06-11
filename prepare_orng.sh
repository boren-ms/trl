#%!/bin/bash
set -x 
echo "Preparing ENV and DATA for ORNG job..."
echo "Job: ${JOB_NAME}"
echo "Node:${NUM_NODE}, GPU:${NUM_GPU}"

region="wus2"

FORCE="false"
if [ "$1" == "--force" ]; then
    FORCE="true"
fi

if [ "${PREPARED_ENV}" != "true" ] || [ "${FORCE}" == "true" ]; then
    echo "Preparing trl environment"
    for i in $(seq 0 $((NUM_NODE-1))); do
        echo "Remotely run ${CODE_DIR}/install.sh on ${JOB_NAME}-${i}"
        ssh ${JOB_NAME}-${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${CODE_DIR}/install.sh > ${CODE_DIR}/install.log 2>&1 & "
    done

    export PREPARED_ENV="true"
    echo "export PREPARED_ENV=true" >> ~/.bashrc
fi
if [ "${PREPARED_DATA}" != "true" ] || [ "${FORCE}" == "true" ]; then
    echo "Preparing data"
    remote_dir="az://orng${region}cresco/data/boren/data"

    bbb sync --delete --concurrency 32 $remote_dir/LibriSpeech $DATA_DIR/LibriSpeech
    bbb sync --delete --concurrency 32 $remote_dir/cpk/phi4_mm_bias $DATA_DIR/cpk/phi4_mm_bias
    echo "Data moved successfully to $DATA_DIR"
    for i in $(seq 1 $((NUM_NODE-1))); do
        echo "Move data to ${JOB_NAME}-${i}"
        nohup rsync -avz $DATA_DIR ${JOB_NAME}-${i}:$(dirname $DATA_DIR) > rsync${i}.log 2>&1 &
    done

    export PREPARED_DATA="true"
    echo "export PREPARED_DATA=true" >> ~/.bashrc
fi

echo "Preparation complete. Starting ORNG job..."
