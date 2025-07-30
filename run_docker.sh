# az login
# az account show
# az account set --subscription "Accoustic Modeling - NonProd"
# az acr list --resource-group devboxes --output table
# az acr login --resource-group devboxes -n sramdevregistry
# run as uid:gid in container

# podman stop boren_phi_dev && podman rm boren_phi_dev
# podman run --rm -d --user boren --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_phi_dev sramdevregistry.azurecr.io/boren_dev:a100_dev tail -f /dev/null
# podman exec -it boren_phi_dev bash

name=boren_trl
# image=nvcr.io/nvidia/pytorch:25.01-py3
# image=vastai/vllm:v0.8.5-cuda-12.4-pytorch-2.6.0-py312
image=vllm/vllm-openai:latest
podman run --rm -d --gpus all --ipc host -v /datablob1:/datablob1 -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_trl ${image} tail -f /dev/null
podman exec -it ${name} bash

# podman stop ${name} && podman rm ${name}
# podman run --rm -d --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_trl sramdevregistry.azurecr.io/boren_dev:trl tail -f /dev/null
# podman run --rm -d --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_trl sramdevregistry.azurecr.io/boren_dev:a100_dev tail -f /dev/null
