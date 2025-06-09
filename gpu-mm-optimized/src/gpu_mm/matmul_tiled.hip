#include "hip/hip_runtime.h"
#include <torch/extension.h>

/*
 * Simple tiled GEMM (C = A × B) in row‑major layout.
 * ────────────────────────────────────────────────
 *  - One block ↔ one TILE×TILE output tile.
 *  - Threads cooperate via shared memory.
 *  - Works on any HIP‑capable GPU, no vendor‑specific intrinsics.
 */

#ifndef TILE
#define TILE 16  // Try 16 or 32
#endif

__global__ void matmul_kernel_tiled(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float*       __restrict__ C,
                                    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles of K dimension
    for (int tileIdx = 0; tileIdx < (K + TILE - 1) / TILE; ++tileIdx) {
        int tiledCol = tileIdx * TILE + threadIdx.x;
        int tiledRow = tileIdx * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

void matmul_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    hipLaunchKernelGGL(matmul_kernel_tiled, blocks, threads, 0, 0,
                       A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                       M, N, K);
}

