import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME

is_rocm_build = (torch.version.hip is not None) and (ROCM_HOME is not None)

common_src  = ["src/gpu_mm/bindings.cpp"]
cuda_src    = ["src/gpu_mm/matmul.cu"]
hip_src     = ["src/gpu_mm/matmul_tiled.hip"]

sources = common_src + (hip_src if is_rocm_build else cuda_src)

setup(
    name="gpu_mm",
    description="A simple GPU matrix multiply extension using PyTorch",
    packages=["gpu_mm"],
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="gpu_mm._matmul",
            sources=sources,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
    ],
    zip_safe=False,
)
