!/bin/bash

# Docker automation script for STS solver
# Builds image and runs all experiments in containerized environment

set -e

IMAGE_NAME="sts-solver"
CONTAINER_NAME="sts-experiment"

echo "======================================"
echo "STS Docker Runner"
echo "======================================"

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "✓ Docker image built successfully"