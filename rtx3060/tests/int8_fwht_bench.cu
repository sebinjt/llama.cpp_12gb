// int8_fwht_bench.cu - Q3 gate: one real ffn matrix, int8 tensor-core GEMM with per-row weight scales,
// per-token activation scales, and an optional FWHT block-1024 rotation to tame outliers.
//
// Gate (reviewer seq 79/83): accuracy of one matrix (rel err vs the f32 GEMM of the SAME weights, i.e. the
// additional error the int8 path introduces) and its speed at ub 512, against MMQ's recorded numbers
// (report 20.2: iq3_xxs 2048 tok 9.75 ms = 37.4 TFLOPS, 4096 tok 20.40 ms). Standalone: no fork edits,
// no production path, default untouched.
//
// Build: tests\build_int8_fwht_bench.bat
// Run:   llama_cpp_src\build-cuda\bin\int8_fwht_bench.exe results\w2\w_ffn_gate_blk9.f32 17408 5120 512 6
//        (argv: weights.f32 M K N reps; N is the token count, run once with 512 and once with 2048)
//
// W is row-major M x K (dequantized IQ3_S blk.9.ffn_gate). X is K x N, column-major for cuBLAS. The
// activations are synthetic but outlier-shaped like a real prefill (a few channels at 20-40x), which is the
// structure the rotation is meant to fix; this is a limitation, not a claim.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("cuda error: %s\n", cudaGetErrorString(e_)); return 1; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_) { printf("cublas error: %d\n", (int) s_); return 1; } } while (0)

using Clock = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

static const int FWHT_BLOCK = 1024;

// normalised fast Walsh-Hadamard transform over blocks of FWHT_BLOCK along the last axis
static void fwht_blocks(float * v, int rows, int cols) {
    const int nb = cols / FWHT_BLOCK;
    const float norm = 1.0f / std::sqrt((float) FWHT_BLOCK);
    for (int r = 0; r < rows; ++r) {
        float * p = v + (size_t) r * cols;
        for (int b = 0; b < nb; ++b) {
            float * x = p + b * FWHT_BLOCK;
            for (int len = 1; len < FWHT_BLOCK; len <<= 1) {
                for (int i = 0; i < FWHT_BLOCK; i += len << 1) {
                    for (int j = 0; j < len; ++j) {
                        const float a = x[i + j], c = x[i + j + len];
                        x[i + j]       = a + c;
                        x[i + j + len] = a - c;
                    }
                }
            }
            for (int i = 0; i < FWHT_BLOCK; ++i) x[i] *= norm;
        }
    }
}


// q8_1-style activation quantisation: one int8 scale per 32 consecutive K values of each token,
// dequantized back to f32. Used to measure MMQ's OWN activation-quantisation error (its weights are the
// exact dequantized IQ3_S values), which is the bar the int8 path must be judged against.
static void quant_per32_dequant(const float * x, int cols, int n, std::vector<float> & out) {
    out.assign((size_t) cols * n, 0.f);
    const int g = 32;
    for (int c = 0; c < n; ++c) {
        for (int k0 = 0; k0 < cols; k0 += g) {
            float amax = 0.f;
            for (int k = k0; k < std::min(cols, k0 + g); ++k) amax = std::max(amax, std::fabs(x[(size_t) k + (size_t) c * cols]));
            const float sc = amax / 127.0f;
            if (sc <= 0.f) continue;
            for (int k = k0; k < std::min(cols, k0 + g); ++k) {
                float v = std::nearbyint(x[(size_t) k + (size_t) c * cols] / sc);
                v = std::max(-127.f, std::min(127.f, v));
                out[(size_t) k + (size_t) c * cols] = v * sc;
            }
        }
    }
}

__global__ void scale_epilogue(const int32_t * __restrict__ c, float * __restrict__ out,
                               const float * __restrict__ w_scale, const float * __restrict__ x_scale,
                               int M, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        // cuBLAS column-major C: idx = row + col*M
        const int row = idx % M;
        const int col = idx / M;
        out[idx] = (float) c[idx] * w_scale[row] * x_scale[col];
    }
}

static void quant_rows_int8(const float * w, int rows, int cols, std::vector<int8_t> & q, std::vector<float> & s) {
    q.resize((size_t) rows * cols);
    s.resize(rows);
    for (int r = 0; r < rows; ++r) {
        const float * p = w + (size_t) r * cols;
        float amax = 0.f;
        for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(p[c]));
        const float sc = amax / 127.0f;
        s[r] = sc > 0 ? sc : 1.f;
        for (int c = 0; c < cols; ++c) {
            float v = std::nearbyint(p[c] / s[r]);
            v = std::max(-127.f, std::min(127.f, v));
            q[(size_t) r * cols + c] = (int8_t) v;
        }
    }
}

// X arrives row-major K x N (token c in row c) but cuBLAS needs column-major K x N with lda=K,
// i.e. element (k, c) at k + c*K; write that layout directly.
static void quant_cols_int8(const float * x, int cols, int n, std::vector<int8_t> & q, std::vector<float> & s) {
    q.resize((size_t) cols * n);
    s.resize(n);
    for (int c = 0; c < n; ++c) {
        float amax = 0.f;
        for (int r = 0; r < cols; ++r) amax = std::max(amax, std::fabs(x[(size_t) r + (size_t) c * cols]));
        const float sc = amax / 127.0f;
        s[c] = sc > 0 ? sc : 1.f;
        for (int r = 0; r < cols; ++r) {
            float v = std::nearbyint(x[(size_t) r + (size_t) c * cols] / s[c]);
            v = std::max(-127.f, std::min(127.f, v));
            q[(size_t) r + (size_t) c * cols] = (int8_t) v;
        }
    }
}

static double rel_err(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return std::sqrt(num / den);
}

int main(int argc, char ** argv) {
    const char * wpath = argc > 1 ? argv[1] : "results/w2/w_ffn_gate_blk9.f32";
    const int M    = argc > 2 ? atoi(argv[2]) : 17408;   // weight rows (output features)
    const int K    = argc > 3 ? atoi(argv[3]) : 5120;    // reduction
    const int N    = argc > 4 ? atoi(argv[4]) : 512;     // tokens
    const int reps = argc > 5 ? atoi(argv[5]) : 6;
    if (K % FWHT_BLOCK != 0) { printf("K must be a multiple of %d\n", FWHT_BLOCK); return 1; }

    std::vector<float> W((size_t) M * K);
    {
        FILE * f = fopen(wpath, "rb");
        if (!f) { printf("cannot open %s\n", wpath); return 1; }
        const size_t got = fread(W.data(), sizeof(float), W.size(), f);
        fclose(f);
        if (got != W.size()) { printf("short read: %zu of %zu\n", got, W.size()); return 1; }
    }
    printf("W: %d x %d from %s (|w| mean %.5f max %.5f)\n", M, K, wpath,
           [&]{ double s=0; for (float v : W) s += std::fabs(v); return s / W.size(); }(),
           [&]{ double m=0; for (float v : W) m = std::max(m, (double) std::fabs(v)); return m; }());

    // activation matrix K x N, column-major. Outlier-shaped: 0.4% of channels scaled 30x.
    std::vector<float> X((size_t) K * N);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> chan(K, 1.f);
        for (int k = 0; k < K; ++k) if ((rng() % 250) == 0) chan[k] = 30.f;
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) X[(size_t) k + (size_t) n * K] = nd(rng) * chan[k];
    }

    // f32 reference: Y = W * X, cuBLAS column-major: A = W^T (K x M), B = X (K x N)
    std::vector<float> Yref((size_t) M * N);
    float * dW = nullptr, * dX = nullptr, * dY = nullptr;
    int8_t * qW = nullptr, * qX = nullptr;
    int32_t * dC = nullptr;
    float * dWs = nullptr, * dXs = nullptr, * dOut = nullptr;
    CK(cudaMalloc(&dW, (size_t) M * K * sizeof(float)));
    CK(cudaMalloc(&dX, (size_t) K * N * sizeof(float)));
    CK(cudaMalloc(&dY, (size_t) M * N * sizeof(float)));
    CK(cudaMalloc(&qW, (size_t) M * K));
    CK(cudaMalloc(&qX, (size_t) K * N));
    CK(cudaMalloc(&dC, (size_t) M * N * sizeof(int32_t)));
    CK(cudaMalloc(&dWs, (size_t) M * sizeof(float)));
    CK(cudaMalloc(&dXs, (size_t) N * sizeof(float)));
    CK(cudaMalloc(&dOut, (size_t) M * N * sizeof(float)));

    cublasHandle_t h;
    CB(cublasCreate(&h));
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    CK(cudaMemcpy(dW, W.data(), (size_t) M * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), (size_t) K * N * sizeof(float), cudaMemcpyHostToDevice));
    const float alpha_f = 1.f, beta_f = 0.f;
    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha_f,
                    dW, CUDA_R_32F, K, dX, CUDA_R_32F, K, &beta_f, dY, CUDA_R_32F, M,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(Yref.data(), dY, (size_t) M * N * sizeof(float), cudaMemcpyDeviceToHost));

    const double flop = 2.0 * M * N * K;
    printf("shape M=%d K=%d N=%d (%.1f GFLOP per GEMM)\n", M, K, N, flop / 1e9);

    auto run_int8 = [&](const std::vector<float> & w_src, const std::vector<float> & x_src, const char * tag) {
        std::vector<int8_t> qi; std::vector<float> sw;
        quant_rows_int8(w_src.data(), M, K, qi, sw);
        std::vector<int8_t> qa; std::vector<float> sx;
        quant_cols_int8(x_src.data(), K, N, qa, sx);
        CK(cudaMemcpy(qW, qi.data(), qi.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(qX, qa.data(), qa.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dWs, sw.data(), M * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dXs, sx.data(), N * sizeof(float), cudaMemcpyHostToDevice));

        const int32_t alpha = 1, beta = 0;
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                            qW, CUDA_R_8I, K, qX, CUDA_R_8I, K, &beta, dC, CUDA_R_32I, M,
                            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            const int total = M * N, blocks = (total + 255) / 256;
            scale_epilogue<<<blocks, 256>>>(dC, dOut, dWs, dXs, M, N);
            CK(cudaDeviceSynchronize());
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        std::vector<float> y((size_t) M * N);
        CK(cudaMemcpy(y.data(), dOut, (size_t) M * N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  %-22s %7.2f ms  %6.1f TOPS   rel_err vs f32 GEMM = %.5f\n",
               tag, best * 1e3, flop / best / 1e12, rel_err(y, Yref));
    };

    // A: plain int8 (per-row weights, per-token activations)
    run_int8(W, X, "int8, no rotation");

    // B: int8 after a normalised FWHT block-1024 rotation applied identically to W rows and X columns
    std::vector<float> Wr = W, Xr = X;
    fwht_blocks(Wr.data(), M, K);
    {
        // X is K x N column-major; transform each column (stride N) by rotating a transposed copy
        std::vector<float> Xt((size_t) N * K);
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) Xt[(size_t) n * K + k] = X[(size_t) k + (size_t) n * K];
        fwht_blocks(Xt.data(), N, K);
        for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) Xr[(size_t) k + (size_t) n * K] = Xt[(size_t) n * K + k];
    }
    run_int8(Wr, Xr, "int8 + FWHT1024");

    // ---- accuracy paths: dequantize the chosen scheme, then a plain f32 GEMM, compare to Yref ----
    auto run_f32_scheme = [&](const std::vector<float> & w, const std::vector<float> & x, const char * tag) {
        CK(cudaMemcpy(dW, w.data(), (size_t) M * K * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dX, x.data(), (size_t) K * N * sizeof(float), cudaMemcpyHostToDevice));
        CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha_f,
                        dW, CUDA_R_32F, K, dX, CUDA_R_32F, K, &beta_f, dY, CUDA_R_32F, M,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t) M * N);
        CK(cudaMemcpy(y.data(), dY, (size_t) M * N * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  %-34s rel_err vs f32 GEMM = %.5f\n", tag, rel_err(y, Yref));
    };

    std::vector<float> Xq32;
    quant_per32_dequant(X.data(), K, N, Xq32);
    run_f32_scheme(W, Xq32, "MMQ-like: exact W, q8_1 X");

    // int8 weights (per-row) with q8_1 activations, no rotation
    {
        std::vector<int8_t> qi; std::vector<float> sw;
        quant_rows_int8(W.data(), M, K, qi, sw);
        std::vector<float> Wq((size_t) M * K);
        for (int r = 0; r < M; ++r) for (int c = 0; c < K; ++c) Wq[(size_t) r * K + c] = (float) qi[(size_t) r * K + c] * sw[r];
        run_f32_scheme(Wq, Xq32, "int8 W + q8_1 X, no rotation");
    }

    // same, after rotation (per-32 activation groups in the rotated domain)
    {
        std::vector<float> Xr32;
        quant_per32_dequant(Xr.data(), K, N, Xr32);
        std::vector<int8_t> qi; std::vector<float> sw;
        quant_rows_int8(Wr.data(), M, K, qi, sw);
        std::vector<float> Wq((size_t) M * K);
        for (int r = 0; r < M; ++r) for (int c = 0; c < K; ++c) Wq[(size_t) r * K + c] = (float) qi[(size_t) r * K + c] * sw[r];
        run_f32_scheme(Wq, Xr32, "int8 W + q8_1 X + FWHT1024 (emul)");
    }

    // fp16 tensor cores for the same shape (sanity reference for the speed scale)
    {
        __half *hW = nullptr, *hX = nullptr, *hY = nullptr;
        CK(cudaMalloc(&hW, (size_t) M * K * 2));
        CK(cudaMalloc(&hX, (size_t) K * N * 2));
        CK(cudaMalloc(&hY, (size_t) M * N * 2));
        std::vector<__half> w16(W.size()), x16(X.size());
        for (size_t i = 0; i < W.size(); ++i) w16[i] = __float2half(W[i]);
        for (size_t i = 0; i < X.size(); ++i) x16[i] = __float2half(X[i]);
        CK(cudaMemcpy(hW, w16.data(), w16.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(hX, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice));
        const __half a16 = __float2half(1.f), b16 = __float2half(0.f);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &a16,
                            hW, CUDA_R_16F, K, hX, CUDA_R_16F, K, &b16, hY, CUDA_R_16F, M,
                            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaDeviceSynchronize());
            const double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("  %-22s %7.2f ms  %6.1f TFLOPS\n", "fp16 (reference)", best * 1e3, flop / best / 1e12);
        cudaFree(hW); cudaFree(hX); cudaFree(hY);
    }

    cublasDestroy(h);
    cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(qW); cudaFree(qX);
    cudaFree(dC); cudaFree(dWs); cudaFree(dXs); cudaFree(dOut);
    return 0;
}

