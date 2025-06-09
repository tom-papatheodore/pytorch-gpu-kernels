import time
import torch
import gpu_mm

# ---------------------------------------------------------------------
# Matrix sizes
# ---------------------------------------------------------------------
#M, K, N = 4096, 8192, 2048
M, K, N = 8192, 16384, 4096

# ---------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------
n_iters = 50          # Number of timed repetitions
device  = "cuda"      # Change to "hip" if needed for ROCm

# ---------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------
torch.manual_seed(0)               # Reproducible
A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)
C = torch.empty(M, N, device=device)

# ---------------------------------------------------------------------
# Warm-up (once for each kernel to prime JIT / caches)
# ---------------------------------------------------------------------
gpu_mm.matmul(A, B, C)
torch.matmul(A, B)

# ---------------------------------------------------------------------
# Custom kernel timing
# ---------------------------------------------------------------------
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    gpu_mm.matmul(A, B, C)
torch.cuda.synchronize()
custom_avg = (time.time() - t0) / n_iters
print(f"Custom kernel avg. time over {n_iters} runs: {custom_avg:.6f} s")

# ---------------------------------------------------------------------
# PyTorch matmul timing
# ---------------------------------------------------------------------
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iters):
    C_ref = torch.matmul(A, B)
torch.cuda.synchronize()
torch_avg = (time.time() - t0) / n_iters
print(f"PyTorch matmul avg. time over {n_iters} runs: {torch_avg:.6f} s")

# ---------------------------------------------------------------------
# Correctness check (using result from last iteration)
# ---------------------------------------------------------------------
max_error = (C - C_ref).abs().max().item()
print(f"Max error: {max_error:.3e}")

