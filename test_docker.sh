#!/bin/bash

# Quick Docker verification script
# Tests that Docker image works correctly with new CLI

set -e

IMAGE_NAME="sts-solver"

echo "======================================"
echo "Docker Setup Verification"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi
echo "✅ Docker is installed"

# Check if image exists
if docker images | grep -q "$IMAGE_NAME"; then
    echo "✅ Docker image '$IMAGE_NAME' exists"
else
    echo "⚠️  Docker image '$IMAGE_NAME' not found"
    echo "Building image..."
    docker build -t $IMAGE_NAME .
fi

echo ""
echo "======================================"
echo "Testing CLI Commands"
echo "======================================"

# Test 1: List models
echo ""
echo "Test 1: List available models"
if docker run --rm $IMAGE_NAME uv run sts list-models | grep -q "CP Models"; then
    echo "✅ list-models command works"
else
    echo "❌ list-models command failed"
    exit 1
fi

# Test 2: Solve small instance
echo ""
echo "Test 2: Solve small instance (n=6, CP, gecode)"
mkdir -p res/CP
if docker run --rm -v "$(pwd)/res:/app/res" $IMAGE_NAME \
    uv run sts solve 6 CP --solver gecode --timeout 30 | grep -q "Solution completed"; then
    echo "✅ Solve command works"
    
    # Check if result file was created
    if [ -f "res/CP/6.json" ]; then
        echo "✅ Result file created"
    else
        echo "❌ Result file not created"
        exit 1
    fi
else
    echo "❌ Solve command failed"
    exit 1
fi

# Test 3: Validate result
echo ""
echo "Test 3: Validate result"
if docker run --rm -v "$(pwd)/res:/app/res" $IMAGE_NAME \
    uv run sts validate res/CP/6.json | grep -q "valid"; then
    echo "✅ Validate command works"
else
    echo "❌ Validate command failed"
    exit 1
fi

# Test 4: Analyze results
echo ""
echo "Test 4: Analyze results"
if docker run --rm -v "$(pwd)/res:/app/res" $IMAGE_NAME \
    uv run sts analyze | grep -q "ANALYSIS"; then
    echo "✅ Analyze command works"
else
    echo "❌ Analyze command failed"
    exit 1
fi

echo ""
echo "======================================"
echo "All Tests Passed! ✅"
echo "======================================"
echo ""
echo "Docker setup is working correctly."
echo "You can now run:"
echo "  ./docker_run.sh          # Full automation"
echo "  docker run ... sts ...   # Individual commands"
