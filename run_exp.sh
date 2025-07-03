#!/bin/bash

set -x

config_file=${1}
prepare=${2:-false}
if [[ -z "${config_file}" ]]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi


if [[ "${prepare}" == "true" ]]; then
    echo "Preparing the environment..."
    # Install dependencies
    bash mpi_bash.sh bash quick_install.sh
    # Sync data
    bash mpi_bash.sh bash quick_data.sh
else
    echo "Skipping preparation steps."
fi

echo "Please run the following command to start the vllm server in a separate node for vllm generation:"
echo "bash quick_vllm_serve.sh"
# kick off the vllm server in a separate node for vllm generation.
# bash quick_vllm_serve.sh

# Run the GRPO bias experiment
echo "Running multi-nodes jobs with ${config_file}"
bash mpi_bash.sh bash run_grpo_bias.sh ${config_file}