#%!/bin/bash

host=$(hostname)
target=${host::-1}
echo "Hostname: ${host}"
echo "Node:${RCALL_INSTANCE_COUNT}, GPU:${RCALL_NUM_GPU}"

export N=${RCALL_INSTANCE_COUNT}

code_dir=/root/code/trl/
for i in $(seq 0 $((N-1))); do
    echo "Remotely run ${code_dir}/install.sh on ${target}${i}"
    ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${code_dir}/install.sh > ${code_dir}/install.log 2>&1 & "
done