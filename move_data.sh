#%!/bin/bash
target=$(hostname | sed 's/.$//')

data_dir="/root/data"
N=7 #last node

region="wus2"
remote_dir="az://orng${region}cresco/data/boren/data"
bbb sync --delete --concurrency 32 $remote_dir $data_dir
echo "Data moved successfully to $data_dir"
for i in $(seq 1 $N); do
    echo "Move data to ${target}${i}"
    scp -r $data_dir ${target}${i}:$(dirname $data_dir)
done

