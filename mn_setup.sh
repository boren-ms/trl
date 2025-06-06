#%!/bin/bash

target=$(hostname | sed 's/.$//')

install_package="true"
copy_code="false"
copy_data="true"

code_dir=/root/code/trl/
data_dir="/root/data"
N=7 #last node
setup_script=/root/code/trl/install.sh
if [ "$install_package" = "true" ]; then
    bash ${setup_script}
    for i in $(seq 1 $N); do
        echo "Remotely run ${script} on ${target}${i}"
        scp  ${setup_script} ${target}${i}:${setup_script}
        ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${setup_script}"
        # ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; nohup bash /root/code/openai/personal/shuowa/verl/recipe/phi_green/install.sh > install${i}.log 2>&1 & "
    done
fi

if [ "$copy_code" = "true" ]; then
    for i in $(seq 1 $N); do
        echo "move ${code_dir} to ${target}${i}"
        scp -r ${code_dir} ${target}${i}:$(dirname $code_dir)
    done 
fi

region="wus2"
if [ "$copy_data" = "true" ]; then
    remote_dir="az://orng${region}cresco/data/boren/data"
    bbb sync --delete --concurrency 32 $remote_dir $data_dir
    echo "Data moved successfully to $data_dir"

    for i in $(seq 1 $N); do
        echo "Move data to ${target}${i}"
        scp -r $data_dir ${target}${i}:$(dirname $data_dir)
    done
fi

