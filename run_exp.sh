#!/bin/bash

# set -x

config_file=${1}
prepare=${2:-false}
if [[ -z "${config_file}" ]]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

config_file=$(realpath "${config_file}")

if [[ "${prepare}" == "true" ]]; then
    echo "Preparing the environment..."
    # Install dependencies
    bash mpi_bash.sh bash quick_install.sh
    # Sync data
    bash mpi_bash.sh bash quick_data.sh
else
    echo "Skipping preparation steps."
fi


# Run the GRPO bias experiment
echo "Running multi-nodes jobs with ${config_file}"

echo bash mpi_bash.sh bash run_grpo_bias.sh ${config_file}
bash mpi_bash.sh bash run_grpo_bias.sh ${config_file}