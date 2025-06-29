#!/bin/bash

set -x


echo "Move data from Azure Blob Storage to local directory"
# move data
region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data/LibriSpeech/train-clean-360"
local_dir="/home/boren/data/LibriSpeech/train-clean-360"

bbb sync --delete --concurrency 32  $local_dir $remote_dir

echo "Data moved successfully to $remote_dir"

# upload tsv az://orng${region}cresco/data/boren/data/LibriSpeech/ 
bbb cp /home/boren/data/LibriSpeech/ls_clean10k_other10k.tsv az://orngwus2cresco/data/boren/data/LibriSpeech/ls_clean10k_other10k.tsv