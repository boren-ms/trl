#%!/bin/bash
target=$(hostname | sed 's/.$//')

data_dir="/root/data"

host=$(hostname)
target=${host::-1}
echo "Hostname: ${host}"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"

export N=${RCALL_INSTANCE_COUNT}
region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data"
bbb sync --delete --concurrency 32 $remote_dir $data_dir
echo "Data moved successfully to $data_dir"
for i in $(seq 1 $((N-1))); do
    echo "Move data to ${target}${i}"
    nohup rsync -avz $data_dir ${target}${i}:$(dirname $data_dir) > rsync${i}.log 2>&1 &
done

