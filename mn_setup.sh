#%!/bin/bash

target=$(hostname | sed 's/.$//')

N=7 #last node
code_dir=/root/code/trl/
for i in $(seq 0 $N); do
    echo "Remotely run ${code_dir}/install.sh on ${target}${i}"
    ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${code_dir}/install.sh > ${code_dir}/install.log 2>&1 & "
done