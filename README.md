# PyTorch GPU Kernels

This tutorial shows how to write custom CUDA kernels for use in PyTorch, how to port them to HIP for AMD GPUs using PyTorch's built-in auto-hipification, and how to maintain independent CUDA and HIP kernels in the same package. A companion slide deck is included as a PDF.

## Repository Structure

```
pytorch-gpu-kernels/
├── gpu-mm/                  # Basic example: CUDA kernel, auto-hipified for ROCm
│   ├── src/gpu_mm/
│   │   ├── matmul.cu        # Naive matrix multiply CUDA kernel
│   │   ├── bindings.cpp     # pybind11 bindings
│   │   └── __init__.py
│   ├── tests/test_mm.py     # Benchmark and correctness test
│   ├── setup.py
│   ├── install.sh
│   ├── requirements_pypi.txt
│   ├── requirements_pytorch.txt
│   └── requirements_pytorch_rocm.txt
│
├── gpu-mm-optimized/        # Advanced example: independent CUDA and HIP kernels
│   ├── src/gpu_mm/
│   │   ├── matmul.cu        # Naive CUDA kernel (used on NVIDIA)
│   │   ├── matmul_tiled.hip # Tiled shared-memory HIP kernel (used on AMD)
│   │   ├── bindings.cpp
│   │   └── __init__.py
│   ├── tests/test_mm.py
│   ├── setup.py
│   ├── install.sh
│   ├── requirements_pypi.txt
│   ├── requirements_pytorch.txt
│   └── requirements_pytorch_rocm.txt
│
├── port-custom-cuda-to-hip-pytorch.pdf   # Slide deck
└── test_on_mi300x.sh                     # Slurm test script for MI300X
```

### gpu-mm

Demonstrates writing a custom CUDA matrix multiply kernel and packaging it as a pip-installable PyTorch extension. On a ROCm system, PyTorch automatically hipifies the `.cu` file at build time -- no code changes are needed.

### gpu-mm-optimized

Extends the basic example by maintaining separate CUDA and HIP kernels. The `setup.py` detects the PyTorch backend at build time and selects the appropriate source file. The HIP kernel uses tiled shared memory for improved performance on AMD GPUs.

## Prerequisites

- Python 3.12+
- A GPU-equipped system with either:
  - NVIDIA GPU with CUDA toolkit, or
  - AMD GPU with ROCm (tested with ROCm 7.2)
- `pip` (used for all installs)

## Quick Start

Each package includes an `install.sh` that creates the environment and builds the extension. From either the `gpu-mm/` or `gpu-mm-optimized/` directory:

```bash
# Create and activate a virtual environment
python3.12 -m venv --upgrade-deps venv
source venv/bin/activate

# Install (choose one)
bash install.sh rocm    # for AMD GPUs
bash install.sh cuda    # for NVIDIA GPUs
```

Or install manually:

```bash
pip install --no-build-isolation -r requirements_pypi.txt
pip install --no-build-isolation -r requirements_pytorch_rocm.txt   # or requirements_pytorch.txt
pip install --no-build-isolation .
```

## Running the Benchmark

```bash
python3 tests/test_mm.py
```

This runs both the custom kernel and PyTorch's built-in `torch.matmul`, prints average times over 50 iterations, and reports the max absolute error.

## Slide Deck

The file `port-custom-cuda-to-hip-pytorch.pdf` contains the companion presentation covering:

- ROCm software stack overview
- LLM training and inference examples on AMD GPUs
- Step-by-step walkthrough of CUDA-to-HIP porting in PyTorch
- Maintaining independent CUDA and HIP kernels
- Helpful resources and documentation links
