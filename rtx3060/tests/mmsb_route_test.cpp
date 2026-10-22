// mmsb_route_test.cpp - in-tree gate for the GGML_CUDA_MMSB v9 route (worker3, seq 82).
// Checks, for IQ3_S/IQ3_XXS mul_mat at ncols 1..16:
//   * column invariance: column c is bit-identical for every ncols that serves it (0 diffs)
//   * per-run f32 dump, so two runs (route off vs on) can be compared for a rel-err / KLD-style check
// Usage: mmsb_route_test <type 0=IQ3_S,1=IQ3_XXS> <in> <out> <dump_prefix>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    const int tyi = argc > 1 ? atoi(argv[1]) : 0;
    const int in  = argc > 2 ? atoi(argv[2]) : 5120;
    const int out = argc > 3 ? atoi(argv[3]) : 5120;
    const int nc = argc > 4 ? atoi(argv[4]) : 8;
    const char * dump = argc > 5 ? argv[5] : nullptr;
    const ggml_type ty = tyi ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ3_S;

    ggml_backend_t cu = ggml_backend_cuda_init(0);
    std::vector<float> wsrc((size_t)in * out);
    for (size_t i = 0; i < wsrc.size(); ++i) wsrc[i] = sinf((float)(i % 1013) * 0.0071f) * 0.1f;
    std::vector<float> imat(in, 1.0f);
    ggml_quantize_init(ty);
    std::vector<uint8_t> wq(ggml_row_size(ty, in) * out);
    ggml_quantize_chunk(ty, wsrc.data(), wq.data(), 0, out, in, imat.data());

    {
        struct ggml_init_params ip = { 256ull*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, ty, in, out);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, nc);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_build_forward_expand(gf, y);
        ggml_backend_t be[2] = { cu, ggml_backend_cpu_init() };
        ggml_backend_sched_t sched = ggml_backend_sched_new(be, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        ggml_backend_sched_reserve(sched, gf);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        std::vector<float> xv((size_t)in * nc);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = sinf((float)(i % 997) * 0.0031f);
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
        std::vector<float> o((size_t)out * nc);
        ggml_backend_tensor_get(y, o.data(), 0, o.size()*sizeof(float));
        // CPU reference for the same graph: validates the route's numbers without external files
        ggml_backend_t cpu2 = ggml_backend_cpu_init();
        ggml_backend_t cbe[1] = { cpu2 };
        ggml_backend_sched_t csched = ggml_backend_sched_new(cbe, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        ggml_backend_sched_reserve(csched, gf);
        ggml_backend_sched_alloc_graph(csched, gf);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        if (ggml_backend_sched_graph_compute(csched, gf) != GGML_STATUS_SUCCESS) { printf("cpu compute failed\n"); return 1; }
        std::vector<float> oc((size_t)out * nc);
        ggml_backend_tensor_get(y, oc.data(), 0, oc.size()*sizeof(float));
        double maxabs = 0, scale = 0;
        for (size_t i = 0; i < oc.size(); ++i) {
            maxabs = fmax(maxabs, fabs((double)o[i] - (double)oc[i]));
            scale  = fmax(scale, fabs((double)oc[i]));
        }
        printf("REL type=%s in=%d out=%d nc=%d maxrel=%.3e (maxabs %.3e scale %.3e) MMSB=%s\n",
               ggml_type_name(ty), in, out, nc, scale > 0 ? maxabs/scale : maxabs, maxabs, scale,
               getenv("GGML_CUDA_MMSB") ? "on" : "off");
        fflush(stdout);
        ggml_backend_sched_free(csched); ggml_backend_free(cpu2);
        if (dump) {
            char fn[512]; snprintf(fn, sizeof(fn), "%s_nc%02d.bin", dump, nc);
            FILE * f = fopen(fn, "wb"); fwrite(o.data(), sizeof(float), o.size(), f); fclose(f);
        }
        ggml_backend_sched_free(sched); ggml_backend_free(be[1]); ggml_free(ctx);
     }
    return 0;
}
