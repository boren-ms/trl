#!/bin/bash

set -x

python trl/scripts/grpo_bias.py --config exp_conf/ls_train_biasing.yaml --output_dir ./output_bias
