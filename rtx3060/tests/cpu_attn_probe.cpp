// cpu_attn_probe.cpp - time ggml's production CPU flash-attention kernel on the exact
// shape a CPU assist would need: q4_0 K/V, 8 KV heads, 32 query heads, 200K positions.
// Build: tests\build_cpu_attn_probe.bat
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

using Clock = std::chrono::steady_clock;
static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int n_kv     = argc > 1 ? atoi(argv[1]) : 204800;
    const int n_threads = argc > 2 ? atoi(argv[2]) : 6;
    const int n_head_q  = 32;
    const int n_head_kv = 8;
    const int d         = 128;

    printf("shape: q[%d,1,%d,1]  k/v q4_0[%d,%d,%d,1]  threads=%d\n",
           d, n_head_q, d, n_kv, n_head_kv, n_threads);

    // ---- payload: 8 KB of tensor metadata is plenty
    struct ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,  d, 1, n_head_q, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, n_kv, n_head_kv, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_Q4_0, d, n_kv, n_head_kv, 1);
    struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                 1.0f / sqrtf((float) d), 0.0f, 0.0f);
    if (!o) { fprintf(stderr, "ggml_flash_attn_ext returned null (unsupported type?)\n"); return 1; }
    ggml_set_name(o, "attn_out");

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "no cpu backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(cpu, n_threads);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    // ---- fill: q4_0 rows need a sane scale (f16) so the math does not blow up
    std::vector<float> qh((size_t) d * n_head_q, 0.0f);
    for (int h = 0; h < n_head_q; h++) {
        for (int i = 0; i < d; i++) qh[(size_t) h * d + i] = (float) ((i % 5) - 2) * 0.02f;
    }
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));

    const size_t row_bytes = (size_t) d / 32 * 18;          // 72 B per 128 q4_0 values
    std::vector<uint8_t> kv((size_t) n_kv * n_head_kv * row_bytes);
    for (size_t i = 0; i < kv.size(); i += 18) {
        kv[i]     = 0x1C;   // f16 0.01-ish
        kv[i + 1] = 0x21;
        for (int j = 2; j < 18; j++) kv[i + j] = (uint8_t) ((i * 7 + j * 31) & 0xFF);
    }
    ggml_backend_tensor_set(k, kv.data(), 0, kv.size());
    ggml_backend_tensor_set(v, kv.data(), 0, kv.size());

    // ---- graph + warmup
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n"); return 1;
    }
    std::vector<float> out((size_t) d * n_head_q, 0.0f);
    ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
    double checksum = 0;
    bool finite = true;
    for (float f : out) { checksum += f; if (!std::isfinite(f)) finite = false; }
    printf("sanity: out[0..3] = %.4f %.4f %.4f %.4f  sum=%.4f  finite=%s\n",
           out[0], out[1], out[2], out[3], checksum, finite ? "yes" : "NO");

    // ---- timed runs
    const double kv_bytes = 2.0 * (double) kv.size();
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        const double t0 = now_s();
        if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) return 1;
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }

    printf("\nCPU attention over %d positions (one layer) : %.2f ms\n", n_kv, best * 1e3);
    printf("KV read   : %.1f MiB => %.2f GB/s\n", kv_bytes / 1048576.0, kv_bytes / best / 1e9);
    printf("vs GPU    : 5.30 ms per layer at 200K (vec kernel)\n");
    printf("ratio     : CPU is %.1fx slower per layer\n", best * 1e3 / 5.30);

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return 0;
}
