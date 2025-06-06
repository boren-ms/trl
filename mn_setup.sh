#%!/bin/bash

target=$(hostname | sed 's/.$//')

setup_script=/root/code/trl/install.sh
install_package="false"
if [ "$install_package" = "true" ]; then
    bash ${setup_script}

    for i in $(seq 1 3); do
        echo "Remotely run ${script} on ${target}${i}"
        scp  ${setup_script} ${target}${i}:${setup_script}
        ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; bash ${setup_script}"
        # ssh ${target}${i} "export PATH=/root/.pyenv/versions/3.11.8/bin/:\$PATH; nohup bash /root/code/openai/personal/shuowa/verl/recipe/phi_green/install.sh > install${i}.log 2>&1 & "
    done
fi

copy_code="false"
if [ "$copy_code" = "true" ]; then
    code_dir=/root/code/trl/
    for i in $(seq 1 3); do
        echo "move ${code_dir} to ${target}${i}"
        scp -r ${code_dir} ${target}${i}:$(dirname $code_dir)
    done 
fi

copy_data="false"
if [ "$copy_data" = "true" ]; then
    region="wus2"
    remote_dir="az://orng${region}cresco/data/boren/data"
    local_dir="/root/data"
    bbb sync --delete --concurrency 32 $remote_dir $local_dir
    echo "Data moved successfully to $local_dir"

    for i in $(seq 1 3); do
        echo "Move data to ${target}${i}"
        scp -r $local_dir ${target}${i}:$(dirname $local_dir)
    done
fi

