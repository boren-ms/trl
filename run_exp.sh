#!/bin/bash

set -x

# Install dependencies
bash mpi_bash.sh quick_install.sh
# Sync data
bash mpi_bash.sh quick_data.sh

# kick off the vllm server in a separate node for vllm generation.
# bash quick_vllm_serve.sh

# Run the GRPO bias experiment
bash mpi_bash.sh run_grpo_bias.sh orng_conf/libri_grpo_v1_n12_err_m300_p8_bp8_mp4_5k.yaml