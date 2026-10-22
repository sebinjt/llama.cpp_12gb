// fa_bench.cpp - FLASH_ATTN_EXT cost vs n_kv at the model's prefill shape.
// Q/K/V conversion cost is measured by timing q4_0 KV against F16 KV on the same shape:
// the delta is what the f16 scratch conversion adds, with no kernel instrumentation.
// Build: tests\build_fa_bench.bat
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Clock = std::chrono::steady_clock;
static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int D    = 256;   // head_dim
    const int NH   = argc > 1 ? atoi(argv[1]) : 24;
    const int NKV  = argc > 2 ? atoi(argv[2]) : 4;
    const int NQ   = argc > 3 ? atoi(argv[3]) : 2048;
    const int reps = argc > 4 ? atoi(argv[4]) : 3;

    const int64_t nkv_list[] = { 32768, 65536, 131072, 200704 };
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_F16 };

    ggml_backend_t cu = ggml_backend_cuda_init(0);
    if (!cu) {
        fprintf(stderr, "cuda backend init failed\n");
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();

    printf("flash_attn_ext D=%d nh=%d nkv=%d n_q=%d, causal, q block at end of kv\n", D, NH, NKV, NQ);
    printf("%-8s %8s %9s %9s %10s %12s %10s\n", "kv type", "n_kv", "ms/call", "ns/kvtok", "GFLOP/call", "TFLOPS", "device");

    for (int64_t n_kv : nkv_list) {
        for (ggml_type ty : types) {
            ggml_quantize_init(ty);
            struct ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(ip);

            struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, NQ, NH);
            struct ggml_tensor * k = ggml_new_tensor_3d(ctx, ty, D, n_kv, NKV);
            struct ggml_tensor * v = ggml_new_tensor_3d(ctx, ty, D, n_kv, NKV);
            struct ggml_tensor * m = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv, NQ, 1);
            struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(o, GGML_PREC_DEFAULT);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, o);

            ggml_backend_t be[2] = { cu, cpu };
            ggml_backend_sched_t sched = ggml_backend_sched_new(be, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
            if (!ggml_backend_sched_reserve(sched, gf) || !ggml_backend_sched_alloc_graph(sched, gf)) {
                printf("%-8s %8lld allocation failed\n", ggml_type_name(ty), (long long) n_kv);
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                continue;
            }

            // q
            std::vector<float> qs((size_t) D * NQ * NH);
            for (size_t i = 0; i < qs.size(); ++i) {
                qs[i] = sinf((float) (i % 601) * 0.01f);
            }
            ggml_backend_tensor_set(q, qs.data(), 0, qs.size() * sizeof(float));

            // k, v rows
            const size_t row = ggml_row_size(ty, D);
            std::vector<float> kf((size_t) D * NKV);
            std::vector<uint8_t> kb(row * NKV * (size_t) n_kv), vb(row * NKV * (size_t) n_kv);
            for (int64_t t = 0; t < n_kv; ++t) {
                for (int h = 0; h < NKV; ++h) {
                    for (int d = 0; d < D; ++d) {
                        kf[(size_t) h * D + d] = sinf((float) ((t * 7 + h * 13 + d) % 500) * 0.02f) * 0.5f;
                    }
                }
                if (ty == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> h16((size_t) D * NKV);
                    for (size_t i = 0; i < h16.size(); ++i) {
                        h16[i] = ggml_fp32_to_fp16(kf[i]);
                    }
                    memcpy(kb.data() + t * row * NKV, h16.data(), row * NKV);
                    memcpy(vb.data() + t * row * NKV, h16.data(), row * NKV);
                } else {
                    std::vector<float> imat(D, 1.0f);
                    for (int h = 0; h < NKV; ++h) {
                        ggml_quantize_chunk(ty, kf.data() + (size_t) h * D, kb.data() + t * row * NKV + h * row,
                                            0, 1, D, imat.data());
                        ggml_quantize_chunk(ty, kf.data() + (size_t) h * D, vb.data() + t * row * NKV + h * row,
                                            0, 1, D, imat.data());
                    }
                }
            }
            ggml_backend_tensor_set(k, kb.data(), 0, kb.size());
            ggml_backend_tensor_set(v, vb.data(), 0, vb.size());

            // causal mask, query block occupies the last NQ positions
            std::vector<ggml_fp16_t> ms((size_t) n_kv * NQ);
            for (int64_t i = 0; i < NQ; ++i) {
                const int64_t pos = n_kv - NQ + i;
                for (int64_t j = 0; j < n_kv; ++j) {
                    ms[(size_t) i * n_kv + j] = ggml_fp32_to_fp16(j <= pos ? 0.0f : -INFINITY);
                }
            }
            ggml_backend_tensor_set(m, ms.data(), 0, ms.size() * sizeof(ggml_fp16_t));

            const char * on = ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, o));
            double best = 1e9;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now_s();
                ggml_backend_sched_graph_compute(sched, gf);
                const double t1 = now_s();
                if (t1 - t0 < best) {
                    best = t1 - t0;
                }
            }
            // 2 flops per (q, kv) pair per dim, causal halves the pairs
            const double flop = 2.0 * (double) NQ * (double) n_kv * (double) D * (double) NH * 0.5;
            printf("%-8s %8lld %9.2f %9.2f %10.1f %12.1f %10s\n", ggml_type_name(ty), (long long) n_kv,
                   best * 1e3, best * 1e9 / (double) n_kv, flop / 1e9, flop / best / 1e12, on);

            ggml_backend_sched_free(sched);
            ggml_free(ctx);
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(cu);
    return 0;
}