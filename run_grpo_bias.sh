#! /bin/bash
# run_accelerate.sh
set -euo pipefail

config_file=${1}
if [[ -z "${config_file}" ]]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_map
region_map=(
    ["southcentralus"]="scus"
    ["westus2"]="wus2"
    ["uksouth"]="uks"
)   
export REGION_CODE=${region_map[$CLUSTER_REGION]}
# update the ENV variables in the config file
envsubst < ${config_file} > ${config_file}.yaml


RANK=${RCALL_INSTANCE_INDEX}
NNODES=${RCALL_INSTANCE_COUNT}
NGPUS=$(($RCALL_NUM_GPU*$NNODES))

cmd="
accelerate launch \
    --num_processes ${NGPUS} \
    --num_machines ${NNODES} \
    --machine_rank ${RANK} \
    --main_process_ip ${RCALL_JOB_NAME}-0 \
    --main_process_port 12345 \
    trl/scripts/grpo_bias.py --config ${config_file} 
"

mkdir -p ${RCALL_LOGDIR}
RANK_LOG_FILE=${RCALL_LOGDIR}/rank_${RANK}.log
echo "Logging to ${RANK_LOG_FILE}"
echo "Running $cmd" > $RANK_LOG_FILE
# printenv >> $RANK_LOG_FILE
$cmd >> $RANK_LOG_FILE 2>&1 