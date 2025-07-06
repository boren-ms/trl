#! /bin/bash
# run_accelerate.sh
set -euo pipefail
# set -x 
config_file=${1}
if [[ -z "${config_file}" ]]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

config_file=$(realpath "${config_file}")
export CLUSTER_REGION=$(echo "$RCALL_KUBE_CLUSTER" | cut -d'-' -f2)
declare -A region_storages
region_storages=(
    ["southcentralus"]="orngscuscresco"
    ["westus2"]="orngwus2cresco"
    ["uksouth"]="orngukscresco"
)   
export DATA_STORAGE=${region_storages[$CLUSTER_REGION]}


MAIN_NODE=${RCALL_JOB_NAME}-0
# update the ENV variables in the config file
new_config_file=${config_file}.tmp
envsubst < ${config_file} > ${new_config_file}

HOST=$(hostname)
echo "syncing config: ${MAIN_NODE} -> ${HOST}"
rsync -avz ${MAIN_NODE}:${new_config_file} ${new_config_file}

echo "Working Dir: ${PWD}"

CONFIG_NAME=$(basename "$config_file" | sed 's/\.[^.]*$//')
OUTPUT_DIR=${HOME}/outputs/${CONFIG_NAME}
mkdir -p ${OUTPUT_DIR}

RANK=${PMI_RANK:-0}
RANK_SIZE=${PMI_SIZE:-1}
cmd="
accelerate launch \
    --num_processes $((RCALL_NUM_GPU*RANK_SIZE)) \
    --num_machines ${RANK_SIZE} \
    --machine_rank ${RANK} \
    --main_process_ip ${MAIN_NODE} \
    --main_process_port 12345 \
    trl/scripts/grpo_bias.py --config ${new_config_file} --output-dir ${OUTPUT_DIR}
"

mkdir -p ${RCALL_LOGDIR}
RANK_LOG_FILE=${RCALL_LOGDIR}/${CONFIG_NAME}_rank_${RANK}.log
echo "Logging to ${RANK_LOG_FILE}"
echo "Running $cmd" > $RANK_LOG_FILE
# printenv >> $RANK_LOG_FILE
$cmd >> $RANK_LOG_FILE 2>&1 
