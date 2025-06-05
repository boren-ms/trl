
#!/bin/bash
set -x 
region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data"
local_dir="/root/data"

bbb sync --delete --concurrency 32 $remote_dir $local_dir
