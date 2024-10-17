#!/bin/bash

# Registry details
REGISTRY_NAME=registry
REGISTRY_URL=localhost:5000
IMAGE_NAME=cuda-networks-docs

# Stop the registry before deleting images
docker stop $REGISTRY_NAME

# Get the list of tags for the image
IMAGE_TAGS=$(curl -s -X GET http://$REGISTRY_URL/v2/$IMAGE_NAME/tags/list | jq -r '.tags[]' 2>/dev/null)

# Loop over the tags and delete each one
for tag in $IMAGE_TAGS; do
  # Get the digest for the image:tag
  DIGEST=$(curl -I -s -X GET http://$REGISTRY_URL/v2/$IMAGE_NAME/manifests/$tag | grep Docker-Content-Digest | awk '{print $2}' | tr -d $'\r')
  # Delete the image:tag
  curl -X DELETE http://$REGISTRY_URL/v2/$IMAGE_NAME/manifests/$DIGEST
done

# Garbage collect the deleted image data from the registry
docker exec $REGISTRY_NAME bin/registry garbage-collect /etc/docker/registry/config.yml

# Start the registry back up
docker start $REGISTRY_NAME

# Delete the old docker image
sudo docker rmi $REGISTRY_URL/$IMAGE_NAME:latest

# Build the docker image
sudo docker build -t $REGISTRY_URL/$IMAGE_NAME:latest .

# Push the docker image to the local registry
sudo docker push $REGISTRY_URL/$IMAGE_NAME:latest

# Delete the old deployment
microk8s kubectl delete deployment cuda-networks-docs-deployment

# Delete the old service
microk8s kubectl delete service cuda-networks-docs-service

# Apply the Kubernetes deployment
microk8s kubectl apply -f deployment.yaml

# Apply the Kubernetes service
microk8s kubectl apply -f service.yaml

# Get SVC
microk8s kubectl get svc

echo "Deployment completed. The documentation should be accessible at http://your-server-ip:30008"
