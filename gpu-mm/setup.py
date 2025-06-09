from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cpp_src  = ["src/gpu_mm/bindings.cpp"]
cuda_src = ["src/gpu_mm/matmul.cu"]

sources = cpp_src + cuda_src

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
