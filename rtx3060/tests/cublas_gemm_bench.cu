// cublas_gemm_bench.cu - what a well-tuned vendor GEMM reaches at the prefill shape.
// MMQ's int8 path measures ~38-41 TFLOPS on this card (report 20). If cuBLAS int8 reaches
// far more, the int8-expansion route has headroom; if it also lands near 40, that route is dead.
// Build: tests\build_cublas_gemm_bench.bat
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda error: %s\n", cudaGetErrorString(e_)); return 1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_) { printf("cublas error: %d\n", (int) s_); return 1; } } while (0)

using Clock = std::chrono::steady_clock;
static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// cuBLAS returns column-major int32: index = m + n * M, so the row scale is idx % M.
__global__ void scale_epilogue(const int32_t * __restrict__ c, float * __restrict__ out,
                               const float * __restrict__ row_scale, int M, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        out[idx] = (float) c[idx] * row_scale[idx % M];
    }
}

int main(int argc, char ** argv) {
    const int M = argc > 3 ? atoi(argv[3]) : 17408;  // output rows
    const int N = argc > 1 ? atoi(argv[1]) : 2048;  // tokens
    const int K = argc > 4 ? atoi(argv[4]) : 5120;  // reduction (argv: N reps M K)
    const int reps = argc > 2 ? atoi(argv[2]) : 6;
    const double flop = 2.0 * M * N * K;

    int8_t * dA = nullptr;
    int8_t * dB = nullptr;
    int32_t * dC = nullptr;
    __half * hA = nullptr;
    __half * hB = nullptr;
    __half * hC = nullptr;

    CK(cudaMalloc(&dA, (size_t) K * M));
    CK(cudaMalloc(&dB, (size_t) K * N));
    CK(cudaMalloc(&dC, (size_t) M * N * sizeof(int32_t)));
    CK(cudaMemset(dA, 1, (size_t) K * M));
    CK(cudaMemset(dB, 1, (size_t) K * N));
    CK(cudaMalloc(&hA, (size_t) K * M * 2));
    CK(cudaMalloc(&hB, (size_t) K * N * 2));
    CK(cudaMalloc(&hC, (size_t) M * N * 2));
    CK(cudaMemset(hA, 0, (size_t) K * M * 2));
    CK(cudaMemset(hB, 0, (size_t) K * N * 2));

    cublasHandle_t h;
    CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    printf("cuBLAS GEMM  M=%d N=%d K=%d  (%.1f GFLOP per call)\n", M, N, K, flop / 1e9);

    // int8 x int8 -> int32, the same arithmetic class MMQ uses
    {
        const int32_t alpha = 1;
        const int32_t beta  = 0;
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                            &alpha, dA, CUDA_R_8I, K, dB, CUDA_R_8I, K,
                            &beta, dC, CUDA_R_32I, M,
                            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaDeviceSynchronize());
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
            printf("  int8  rep%d %7.2f ms\n", r, dt * 1e3);
        }
        printf("  int8  -> %7.2f ms  %6.1f TOPS\n", best * 1e3, flop / best / 1e12);
    }

    // int8 GEMM plus the scaling epilogue any real int8 path needs: out = c * w_scale[row] * a_scale[token]
    {
        float * dRowScale = nullptr;
        float * dOut = nullptr;
        CK(cudaMalloc(&dRowScale, (size_t) M * sizeof(float)));
        CK(cudaMemset(dRowScale, 0, (size_t) M * sizeof(float)));
        CK(cudaMalloc(&dOut, (size_t) M * N * sizeof(float)));

        const int32_t alpha = 1;
        const int32_t beta  = 0;
        const int total  = M * N;
        const int blocks = (total + 255) / 256;

        double best_epi = 1e9, best_seq = 1e9;
        for (int r = 0; r < reps; ++r) {
            // epilogue alone
            double t0 = now_s();
            scale_epilogue<<<blocks, 256>>>(dC, dOut, dRowScale, M, N);
            CK(cudaDeviceSynchronize());
            const double de = now_s() - t0;
            if (de < best_epi) best_epi = de;

            // the full sequence: GEMM then scaled store
            t0 = now_s();
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                            &alpha, dA, CUDA_R_8I, K, dB, CUDA_R_8I, K,
                            &beta, dC, CUDA_R_32I, M,
                            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            scale_epilogue<<<blocks, 256>>>(dC, dOut, dRowScale, M, N);
            CK(cudaDeviceSynchronize());
            const double ds = now_s() - t0;
            if (ds < best_seq) best_seq = ds;
        }
        printf("  int8  epilogue alone %6.2f ms ; GEMM+epilogue %6.2f ms -> %6.1f TOPS end to end\n",
               best_epi * 1e3, best_seq * 1e3, flop / best_seq / 1e12);
        cudaFree(dRowScale); cudaFree(dOut);
    }

    // fp16 tensor cores, for reference
    {
        const __half alpha = __float2half(1.f);
        const __half beta  = __float2half(0.f);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
                            &alpha, hA, CUDA_R_16F, K, hB, CUDA_R_16F, K,
                            &beta, hC, CUDA_R_16F, M,
                            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaDeviceSynchronize());
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
            printf("  fp16  rep%d %7.2f ms\n", r, dt * 1e3);
        }
        printf("  fp16  -> %7.2f ms  %6.1f TFLOPS\n", best * 1e3, flop / best / 1e12);
    }

    cublasDestroy(h);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(hA); cudaFree(hB); cudaFree(hC);
    return 0;
}
