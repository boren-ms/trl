# az login
# az acr login -n sramdevregistry -g devboxes
# set -x
# docker build --build-arg WANDB_API_KEY="${WANDB_API_KEY}" -t sramdevregistry.azurecr.io/boren_dev:trl -f Dockerfile.trl .
# docker push sramdevregistry.azurecr.io/boren_dev:trl
# set -x

tag=boren_trl
docker build --build-arg WANDB_API_KEY="${WANDB_API_KEY}" -t sramdevregistry.azurecr.io/boren_dev:${tag} -f Dockerfile.${tag} .
docker push sramdevregistry.azurecr.io/boren_dev:${tag}

# sudo nvidia-docker run --net host --ipc host -v /home/boren:/home/boren --memory 416G --name boren_dev speechpipelineregistry01.azurecr.io/cascades:official
# sudo nvidia-docker exec -it cascades bash
