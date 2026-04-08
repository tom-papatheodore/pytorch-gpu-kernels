#!/bin/bash
# Clean up build artifacts from both gpu-mm and gpu-mm-optimized packages.
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

rm -rf "${REPO_DIR}/gpu-mm/build"
rm -rf "${REPO_DIR}/gpu-mm/src/gpu_mm.egg-info"
rm -f  "${REPO_DIR}/gpu-mm/src/gpu_mm/matmul.hip"

rm -rf "${REPO_DIR}/gpu-mm-optimized/build"
rm -rf "${REPO_DIR}/gpu-mm-optimized/src/gpu_mm.egg-info"

echo "Clean."
