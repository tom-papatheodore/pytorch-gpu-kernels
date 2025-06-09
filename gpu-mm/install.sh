#!/bin/bash
set -e

# Usage help
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [cuda|rocm]"
    exit 1
fi

MODE="$1"

if [[ "$MODE" == "cuda" ]]; then
    echo "Installing dependencies for CUDA..."
    pip install --no-build-isolation -r requirements_pypi.txt
    pip install --no-build-isolation -r requirements_pytorch.txt

elif [[ "$MODE" == "rocm" ]]; then
    echo "Installing dependencies for ROCm..."
    pip install --no-build-isolation -r requirements_pypi.txt
    pip install --no-build-isolation -r requirements_pytorch_rocm.txt

else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [cuda|rocm]"
    exit 1
fi

echo "Installing GPU extension package..."
pip install --no-build-isolation .

echo "Done."

