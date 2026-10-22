import re
p = r'../ggml/src/ggml-cuda/ggml-cuda.cu'
s = open(p, encoding='utf-8', errors='replace').read()
anchor = "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"
assert s.count(anchor) == 1, s.count(anchor)

code = r'''
// ---------------------------------------------------------------------------------------------
// Q3 prototype (opt-in, prefill only): int8 tensor-core ffn matmuls.
// Env: LLAMA_CUDA_INT8_FFN=1, LLAMA_CUDA_INT8_FFN_LAYERS="9,17,..." (blk.N.ffn_down.weight),
//      LLAMA_CUDA_INT8_MIN_TOKENS=<int, default 64>.
// Weights: prepared once on first use - read the quantised bytes back, dequantise on the CPU with the type
// traits, normalised FWHT block-1024 along K, per-row absmax to int8. Activations: per-token absmax after
// the same rotation, on the GPU. GEMM: cuBLAS int8 with an int32 accumulator and a scale epilogue into the
// f32 dst. Decode (n_tokens < threshold) never enters here; with the env unset no instruction of this path
// runs and the default is unchanged.
// ---------------------------------------------------------------------------------------------
struct ggml_cuda_int8_ent {
    int8_t * q = nullptr;
    float  * s = nullptr;
    int64_t  M = 0, K = 0;
};
static std::vector<std::pair<const ggml_tensor *, ggml_cuda_int8_ent>> g_int8_ffn;
static int8_t * g_int8_act     = nullptr;
static size_t   g_int8_act_cap = 0;
static float  * g_int8_act_s   = nullptr;
static size_t   g_int8_act_s_cap = 0;

static bool ggml_cuda_int8_enabled(void) {
    static const bool on = getenv("LLAMA_CUDA_INT8_FFN") != nullptr;
    return on;
}

static bool ggml_cuda_int8_wanted(const char * name) {
    static const std::string layers = [] { const char * e = getenv("LLAMA_CUDA_INT8_FFN_LAYERS"); return e ? std::string(e) : std::string(); }();
    if (layers.empty() || name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    const char * p = name + 4;
    char * end = nullptr;
    const long il = strtol(p, &end, 10);
    if (end == p || strncmp(end, ".ffn_down.weight", 16) != 0) {
        return false;
    }
    size_t pos = 0;
    while (pos < layers.size()) {
        size_t komma = layers.find(',', pos);
        if (komma == std::string::npos) komma = layers.size();
        if (strtol(layers.c_str() + pos, nullptr, 10) == il) return true;
        pos = komma + 1;
    }
    return false;
}

// normalised FWHT over blocks of 1024 along a row (host side, weight preparation only)
static void ggml_cuda_int8_fwht_row(float * v, int64_t k) {
    const int nb = 1024;
    const float norm = 1.0f / sqrtf((float) nb);
    for (int64_t b = 0; b < k; b += nb) {
        float * x = v + b;
        for (int len = 1; len < nb; len <<= 1) {
            for (int i = 0; i < nb; i += len << 1) {
                for (int j = 0; j < len; ++j) {
                    const float a = x[i + j], c = x[i + j + len];
                    x[i + j] = a + c;
                    x[i + j + len] = a - c;
                }
            }
        }
        for (int i = 0; i < nb; ++i) x[i] *= norm;
    }
}

// rotate + per-token absmax + int8, one block per token; two passes over the K blocks
__global__ void ggml_cuda_int8_quant_act(const float * __restrict__ x, int8_t * __restrict__ q,
                                         float * __restrict__ scale, int64_t K, int64_t N) {
    extern __shared__ float sh[];
    const int64_t n = blockIdx.x;
    if (n >= N) return;
    const float * col = x + n * K;
    float * rot = sh;
    float amax = 0.0f;
    for (int64_t k0 = 0; k0 < K; k0 += 1024) {
        __syncthreads();
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) rot[i] = col[k0 + i];
        __syncthreads();
        for (int len = 1; len < 1024; len <<= 1) {
            for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
                const int half = (i / (len << 1)) * (len << 1) + (i % len);
                const float a = rot[half], c = rot[half + len];
                const float v = (i < half + len) ? (a + c) : (a - c);
                __syncthreads();
                rot[i] = v;
                __syncthreads();
            }
        }
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) amax = fmaxf(amax, fabsf(rot[i]));
    }
    __shared__ float red[32];
    float m = amax;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        const int nw = (int) ((blockDim.x + 31) / 32);
        for (int i = 0; i < nw; ++i) t = fmaxf(t, red[i]);
        scale[n] = (t > 0.f ? t : 1.f) / 127.0f;
    }
    __syncthreads();
    const float inv = 127.0f / fmaxf(1e-20f, scale[n] * 127.0f);
    const float norm = 1.0f / sqrtf(1024.0f);
    int8_t * qc = q + n * K;
    for (int64_t k0 = 0; k0 < K; k0 += 1024) {
        __syncthreads();
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) rot[i] = col[k0 + i] * norm;
        __syncthreads();
        for (int len = 1; len < 1024; len <<= 1) {
            for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
                const int half = (i / (len << 1)) * (len << 1) + (i % len);
                const float a = rot[half], c = rot[half + len];
                const float v = (i < half + len) ? (a + c) : (a - c);
                __syncthreads();
                rot[i] = v;
                __syncthreads();
            }
        }
        for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
            float v = nearbyintf(rot[i] * inv);
            v = fminf(127.f, fmaxf(-127.f, v));
            qc[k0 + i] = (int8_t) v;
        }
    }
}

__global__ void ggml_cuda_int8_epilogue(const int32_t * __restrict__ c, float * __restrict__ out,
                                        const float * __restrict__ ws, const float * __restrict__ xs,
                                        int64_t M, int64_t N) {
    const int64_t idx = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (idx < M * N) {
        const int64_t row = idx % M, col = idx / M;
        out[idx] = (float) c[idx] * ws[row] * xs[col];
    }
}

// returns true when it handled the mul_mat
static bool ggml_cuda_int8_ffn_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_int8_enabled() || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_cuda_int8_wanted(src0->name)) {
        return false;
    }
    static const int64_t min_tokens = [] { const char * e = getenv("LLAMA_CUDA_INT8_MIN_TOKENS"); return e ? (int64_t) atoll(e) : (int64_t) 64; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (N < min_tokens || K % 1024 != 0 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    ggml_cuda_int8_ent * e = nullptr;
    for (auto & kv : g_int8_ffn) if (kv.first == src0) { e = &kv.second; break; }
    if (e == nullptr) {
        std::vector<uint8_t> raw(ggml_nbytes(src0));
        CUDA_CHECK(cudaMemcpy(raw.data(), src0->data, raw.size(), cudaMemcpyDeviceToHost));
        std::vector<float> w((size_t) K * M);
        ggml_get_type_traits(src0->type)->to_float(raw.data(), w.data(), K * M);
        std::vector<int8_t> qh((size_t) K * M);
        std::vector<float>  sh((size_t) M);
        for (int64_t r = 0; r < M; ++r) {
            float * row = w.data() + r * K;
            ggml_cuda_int8_fwht_row(row, K);
            float amax = 0.f;
            for (int64_t k = 0; k < K; ++k) amax = std::max(amax, fabsf(row[k]));
            sh[r] = amax > 0 ? amax / 127.f : 1.f;
            for (int64_t k = 0; k < K; ++k) {
                float v = nearbyintf(row[k] / sh[r]);
                qh[r * K + k] = (int8_t) std::min(127.f, std::max(-127.f, v));
            }
        }
        ggml_cuda_int8_ent ne;
        ne.M = M; ne.K = K;
        CUDA_CHECK(cudaMalloc(&ne.q, (size_t) K * M));
        CUDA_CHECK(cudaMalloc(&ne.s, (size_t) M * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(ne.q, qh.data(), (size_t) K * M, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ne.s, sh.data(), (size_t) M * sizeof(float), cudaMemcpyHostToDevice));
        g_int8_ffn.push_back({src0, ne});
        e = &g_int8_ffn.back().second;
        fprintf(stderr, "INT8FFN prepared %s M=%lld K=%lld (%.1f MiB int8)\n", src0->name,
                (long long) M, (long long) K, (double) K * M / 1048576.0);
        fflush(stderr);
    }
    if ((size_t) K * N > g_int8_act_cap) {
        if (g_int8_act) cudaFree(g_int8_act);
        CUDA_CHECK(cudaMalloc(&g_int8_act, (size_t) K * N));
        g_int8_act_cap = (size_t) K * N;
    }
    if ((size_t) N > g_int8_act_s_cap) {
        if (g_int8_act_s) cudaFree(g_int8_act_s);
        CUDA_CHECK(cudaMalloc(&g_int8_act_s, (size_t) N * sizeof(float)));
        g_int8_act_s_cap = (size_t) N;
    }
    ggml_cuda_int8_quant_act<<<(int) N, 256, 1024 * sizeof(float), ctx.stream()>>>(
        (const float *) src1->data, g_int8_act, g_int8_act_s, K, N);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int32_t> c32(ctx.pool(), (size_t) M * N);
    const int32_t alpha = 1, beta = 0;
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, (int) M, (int) N, (int) K,
                              &alpha, e->q, CUDA_R_8I, (int) K, g_int8_act, CUDA_R_8I, (int) K,
                              &beta, c32.get(), CUDA_R_32I, (int) M,
                              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    const int64_t total = M * N;
    ggml_cuda_int8_epilogue<<<(int) ((total + 255) / 256), 256, 0, ctx.stream()>>>(
        c32.get(), (float *) dst->data, e->s, g_int8_act_s, M, N);
    CUDA_CHECK(cudaGetLastError());
    static int64_t n_calls = 0;
    if (n_calls++ < 8) {
        fprintf(stderr, "INT8FFN %s M=%lld K=%lld N=%lld\n", src0->name, (long long) M, (long long) K, (long long) N);
        fflush(stderr);
    }
    return true;
}

'''

s = s.replace(anchor, code + anchor, 1)

old = """    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {"""
new = """    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_int8_ffn_try(ctx, src0, src1, dst)) {
        return;
    }

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {"""
assert s.count(old) == 1
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8', errors='replace').write(s)
print('int8 ffn path inserted')
