#!/bin/bash

# Exit on any error
set -e

# Test Docker installation
echo "Testing Docker installation with hello-world container..."
docker run hello-world

# Build Docker image
echo "Building Docker image..."
docker build -t llama3-8b-cuda-inference .

# Run the Docker image
echo "Running Docker image..."
docker run --gpus all -it llama3-8b-cuda-inference /bin/bash
