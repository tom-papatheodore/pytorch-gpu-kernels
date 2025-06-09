#include <torch/extension.h>

void matmul_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_launcher, "Matrix Multiply CUDA");
}
