#!/bin/bash

# Exit on any error except for the groupadd error when the group already exists
set -e

# Install Nvidia Container Toolkit to allow Docker to access GPUs
echo "Installing Nvidia Container Toolkit..."

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Add the current user to the docker group
echo "Setting up Docker permissions..."
if getent group docker > /dev/null 2>&1; then
    echo "Group 'docker' already exists, skipping group creation."
else
    sudo groupadd docker
fi

sudo usermod -aG docker $USER

# Inform the user to log out and log back in
echo "Docker has been installed and the current user has been added to the 'docker' group."
echo "Please log out and log back in, or start a new terminal session."
echo "Once done, you can run './run-docker.sh' to build and run the Docker image."
