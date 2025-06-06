
#!/bin/bash
set -x 
region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data"
local_dir="/root/data"
# download data from Azure Blob Storage to local directory
# bbb sync --delete --concurrency 32 $remote_dir $local_dir

# upload data to remote server
local_dir=/home/boren/data/LibriSpeech
bbb sync --delete --concurrency 32 $local_dir $remote_dir