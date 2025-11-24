#!/bin/bash

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

# Run the container with volume mounting for results
echo ""
echo "Running experiments in Docker container..."
echo "Results will be saved to ./res/"

docker run --rm \
    --name $CONTAINER_NAME \
    -v "$(pwd)/res:/app/res" \
    -v "$(pwd)/solution_checker.py:/app/solution_checker.py" \
    $IMAGE_NAME \
    /bin/bash -c "
        echo 'Starting automated benchmark...'
        chmod +x /app/run_all.sh
        /app/run_all.sh
        
        echo ''
        echo 'Validating results using official checker...'
        if [ -f /app/solution_checker.py ]; then
            uv run sts validate-all --official
        else
            echo 'Note: solution_checker.py not found, skipping validation'
            echo 'Place solution_checker.py in project root to enable validation'
        fi
        
        echo ''
        echo 'Experiments completed!'
        echo 'Results are available in ./res/ directory'
    "

echo ""
echo "======================================"
echo "Docker experiments completed!"
echo "======================================"
echo "Results are available in ./res/ directory"
echo "Check individual .json files for solver outputs"