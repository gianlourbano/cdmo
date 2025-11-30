#!/bin/bash
# Usage: ./run.sh <n> <approach> [--model <model>] [--optimization] [--solver <solver>]
# Executes the solve command inside Docker, forwarding arguments.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <n> <approach> [--model <model>] [--optimization] [--solver <solver>]"
  exit 1
fi

N="$1"
APPROACH="$2"
shift 2

# Build docker image if not present
IMAGE_NAME="sts-solver"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "Building Docker image '$IMAGE_NAME'..."
  docker build -t "$IMAGE_NAME" .
fi

# Run solve in container, mounting workspace
docker run --rm \
  -v "$PWD":"/app" \
  -w "/app" \
  "$IMAGE_NAME" \
  uv run sts solve "$N" "$APPROACH" "$@"
