# Nvidia CUDA Base Image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    clang \
    docker.io \
    python3.10 \
    python3-pip && pip install -U "huggingface_hub[cli]"

# Create the workspace directory
RUN mkdir -p /llama3-workspace

# Set the working directory
WORKDIR /llama3-workspace

COPY . /llama3-workspace

# Default command to keep the container running and allow manual commands
CMD ["/bin/bash"]