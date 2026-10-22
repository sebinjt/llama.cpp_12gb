// mmq_types_bench.cpp - MMQ throughput at the prefill shape for every quant type this model uses.
// Copy of matmul_bench.cpp with the model's real type mix; used by _mmq_sweep.py.
// Answers one question: is IQ2_S/IQ3_XXS mul_mat slower per FLOP than Q8_0 or F16,
// and how much does an explicit dequantize pass cost?
// Build: tests\build_mmq_types_bench.bat
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
    const int n_tok   = argc > 1 ? atoi(argv[1]) : 2048;
    const int reps    = argc > 2 ? atoi(argv[2]) : 3;
    const int in_dim  = argc > 3 ? atoi(argv[3]) : 5120;
    const int out_dim = argc > 4 ? atoi(argv[4]) : 17408;

    ggml_backend_t cu = ggml_backend_cuda_init(0);
    if (!cu) {
        fprintf(stderr, "cuda backend init failed\n");
        return 1;
    }
    // the scheduler requires a CPU backend as the last one
    ggml_backend_t cpu = ggml_backend_cpu_init();

    const size_t n_w = (size_t) in_dim * out_dim;
    std::vector<float> wsrc(n_w);
    for (size_t i = 0; i < n_w; ++i) {
        wsrc[i] = sinf((float) (i % 1000) * 0.001f) * 0.1f;
    }
    std::vector<float> xsrc((size_t) in_dim * n_tok);
    for (size_t i = 0; i < xsrc.size(); ++i) {
        xsrc[i] = sinf((float) (i % 777) * 0.002f);
    }
    // per-column importance, shared by every row; the IQ quantizers assert on a null pointer
    std::vector<float> imat(in_dim, 1.0f);

    printf("mul_mat %d x %d, %d tokens, single matrix (not the model)\n", in_dim, out_dim, n_tok);
    printf("%-10s %10s %10s %10s  %s\n", "type", "w MiB", "ms", "TFLOPS", "device");

    const ggml_type types[] = {
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0,
    };

    for (ggml_type ty : types) {
        ggml_quantize_init(ty);
        const size_t row_size = ggml_row_size(ty, in_dim);
        std::vector<uint8_t> wq(row_size * out_dim);
        const size_t got = ggml_quantize_chunk(ty, wsrc.data(), wq.data(), 0, out_dim, in_dim, imat.data());
        if (got != wq.size()) {
            printf("%-10s quantize size mismatch: %zu vs %zu\n", ggml_type_name(ty), got, wq.size());
            continue;
        }

        struct ggml_init_params ip = { 512ull * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, ty, in_dim, out_dim);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tok);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        ggml_backend_t be[2] = { cu, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(be, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!ggml_backend_sched_reserve(sched, gf) || !ggml_backend_sched_alloc_graph(sched, gf)) {
            printf("%-10s allocation failed\n", ggml_type_name(ty));
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(x, xsrc.data(), 0, xsrc.size() * sizeof(float));
        // the scheduler may silently park an unsupported op on the CPU backend: name the device
        const char * on = ggml_backend_name(ggml_backend_sched_get_tensor_backend(sched, y));

        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            ggml_backend_sched_graph_compute(sched, gf);
            const double t1 = now_s();
            if (t1 - t0 < best) {
                best = t1 - t0;
            }
            printf("      rep%d %.2f ms\n", r, (t1 - t0) * 1e3);
        }
        const double flop = 2.0 * (double) n_w * n_tok;
        printf("%-10s %10.1f %10.2f %10.1f  %s\n", ggml_type_name(ty), wq.size() / 1048576.0,
               best * 1e3, flop / best / 1e12, on);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    ggml_backend_free(cpu);
    ggml_backend_free(cu);
    return 0;
}
