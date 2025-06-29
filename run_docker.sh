# az login
# az account show
# az account set --subscription "Accoustic Modeling - NonProd"
# az acr list --resource-group devboxes --output table
# az acr login --resource-group devboxes -n sramdevregistry
# run as uid:gid in container

# docker stop boren_phi_dev && docker rm boren_phi_dev
# docker run --rm -d --user boren --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_phi_dev sramdevregistry.azurecr.io/boren_dev:a100_dev tail -f /dev/null
# docker exec -it boren_phi_dev bash

docker stop boren_trl && docker rm boren_trl
docker run --rm -d --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_trl sramdevregistry.azurecr.io/boren_dev:trl tail -f /dev/null
# docker run --rm -d --gpus all --ipc host -v /home/boren:/home/boren -v /mnt:/mnt -v /mnt2:/mnt2 --name boren_trl sramdevregistry.azurecr.io/boren_dev:a100_dev tail -f /dev/null
docker exec -it boren_trl bash
