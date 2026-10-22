// mmvq_cols_bench.cpp - MMVQ cost vs number of columns (decode / speculative verify shapes), per weight type.
// Usage: mmvq_cols_bench [reps=50] [dump_prefix]
// For each type and shape it times a graph of 16 identical mul_mats (amortizes launch overhead) at ncols 1,2,4,5,8,
// and, with dump_prefix, writes the f32 outputs so that runs with different env flags can be compared bit for bit.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 50;
    const char * dump = argc > 2 ? argv[2] : nullptr;
    const int NMM = 16;
    ggml_backend_t cu  = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    const ggml_type types[] = { GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K,
                                GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    // {in, out}: both FFN shapes plus the real attention qkv (5120 -> 10240) and output (6144 -> 5120) shapes
    const int shapes[][2] = { {5120, 17408}, {17408, 5120}, {5120, 10240}, {6144, 5120}, {5120, 1024} };
    const int cols[] = { 1, 2, 4, 5, 8 };
    printf("%-8s %12s %5s %10s %12s\n", "type", "shape", "ncols", "us/matmul", "us/col");
    for (ggml_type ty : types) {
        for (auto & sh : shapes) {
            const int in = sh[0], out = sh[1];
            std::vector<float> wsrc((size_t) in * out);
            for (size_t i = 0; i < wsrc.size(); ++i) wsrc[i] = sinf((float) (i % 1013) * 0.0071f) * 0.1f;
            std::vector<float> imat(in, 1.0f);
            ggml_quantize_init(ty);
            std::vector<uint8_t> wq(ggml_row_size(ty, in) * out);
            ggml_quantize_chunk(ty, wsrc.data(), wq.data(), 0, out, in, imat.data());
            for (int nc : cols) {
                struct ggml_init_params ip = { 64ull * 1024 * 1024, nullptr, true };
                struct ggml_context * ctx = ggml_init(ip);
                struct ggml_tensor * w = ggml_new_tensor_2d(ctx, ty, in, out);
                struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, nc);
                struct ggml_cgraph * gf = ggml_new_graph(ctx);
                struct ggml_tensor * y0 = nullptr;
                for (int k = 0; k < NMM; ++k) {
                    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
                    if (!y0) y0 = y;
                    ggml_build_forward_expand(gf, y);
                }
                ggml_backend_t be[2] = { cu, cpu };
                ggml_backend_sched_t sched = ggml_backend_sched_new(be, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
                ggml_backend_sched_reserve(sched, gf);
                ggml_backend_sched_alloc_graph(sched, gf);
                ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
                std::vector<float> xs((size_t) in * nc);
                // column j of the input does not depend on nc, so outputs can be compared across nc
                for (int j = 0; j < nc; ++j)
                    for (int i = 0; i < in; ++i) xs[(size_t) j * in + i] = sinf((float) ((i * 7 + j * 131) % 997) * 0.003f);
                ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
                ggml_backend_sched_graph_compute(sched, gf);
                std::vector<double> t;
                for (int r = 0; r < reps; ++r) {
                    const double t0 = now_s();
                    ggml_backend_sched_graph_compute(sched, gf);
                    ggml_backend_synchronize(cu);
                    t.push_back((now_s() - t0) * 1e6 / NMM);
                }
                std::sort(t.begin(), t.end());
                printf("%-8s %5dx%-6d %5d %10.1f %12.1f\n", ggml_type_name(ty), in, out, nc, t[t.size() / 2], t[t.size() / 2] / nc);
                if (dump) {
                    std::vector<float> o((size_t) out * nc);
                    ggml_backend_tensor_get(y0, o.data(), 0, o.size() * sizeof(float));
                    char p[512];
                    snprintf(p, sizeof(p), "%s_%s_%d_%d_c%d.f32", dump, ggml_type_name(ty), in, out, nc);
                    FILE * f = fopen(p, "wb");
                    if (f) { fwrite(o.data(), sizeof(float), o.size(), f); fclose(f); }
                }
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(cpu);
    ggml_backend_free(cu);
    return 0;
}
