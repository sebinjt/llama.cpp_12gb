// dec_fa_bench.cpp - decode-shaped FLASH_ATTN_EXT timing and output dumps: D=256, 24 Q / 4 KV heads.
// Extended copy of the reviewer's scratchpad bench.
// Usage: dec_fa_bench [n_kv=204800] [reps=20] [dump_prefix|-] [kv_type=q4_0|f16] [nq_list=1,2,4] [data=flat|real] [valid_len]
//   data=flat: the original data (fixed scale 0.02, uniform Q, zero mask).
//   data=real: random per-block scales, peaky logits (a few "hot" tokens), and a per-row masked tail, so that the
//              softmax is far from uniform and the mask matters.
// Kernel selection comes from the environment (LLAMA_FA_DECODE_Q4, LLAMA_FA_VEC_QUANT_KV, LLAMA_FA_VEC_GQA, ...).
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
#include <random>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int     D    = 256;
    const int     nh   = 24;
    const int     nkvh = 4;
    const int64_t n_kv = argc > 1 ? atoll(argv[1]) : 204800;
    const int     reps = argc > 2 ? atoi(argv[2]) : 20;
    const char *  dump = argc > 3 && strcmp(argv[3], "-") != 0 ? argv[3] : nullptr;
    const bool    f16  = argc > 4 && strcmp(argv[4], "f16") == 0;
    std::vector<int> nqs;
    {
        std::string l = argc > 5 ? argv[5] : "1,2,4";
        size_t p = 0;
        while (p < l.size()) {
            nqs.push_back(atoi(l.c_str() + p));
            const size_t c = l.find(',', p);
            if (c == std::string::npos) break;
            p = c + 1;
        }
    }
    const bool real = argc > 6 && strcmp(argv[6], "real") == 0;
    const int64_t valid_len = argc > 7 ? atoll(argv[7]) : 0; // real data: causal mask over the first valid_len tokens
    const ggml_type kvt = f16 ? GGML_TYPE_F16 : GGML_TYPE_Q4_0;

    ggml_backend_t cu  = ggml_backend_cuda_init(0);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cu) { fprintf(stderr, "cuda init failed\n"); return 1; }

    printf("n_kv=%lld D=%d %s K/V, %d/%d heads, data=%s, reps=%d\n", (long long) n_kv, D, ggml_type_name(kvt), nh, nkvh,
           real ? "real" : "flat", reps);
    printf("%4s %4s %3s %9s %9s %12s %12s\n", "nh", "nkv", "nq", "min ms", "med ms", "KV MiB/call", "GB/s (KV once)");

    for (const int nq : nqs) {
        std::mt19937 rng(42);
        struct ggml_init_params ip = { 64ull * 1024 * 1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(ip);
        struct ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, nq, nh);
        struct ggml_tensor * k = ggml_new_tensor_3d(ctx, kvt, D, n_kv, nkvh);
        struct ggml_tensor * v = ggml_new_tensor_3d(ctx, kvt, D, n_kv, nkvh);
        struct ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, nq);
        struct ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf((float) D), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_DEFAULT);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, o);

        ggml_backend_t be[2] = { cu, cpu };
        ggml_backend_sched_t sched = ggml_backend_sched_new(be, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
        if (!ggml_backend_sched_reserve(sched, gf) || !ggml_backend_sched_alloc_graph(sched, gf)) {
            printf("%4d %4d %3d allocation failed\n", nh, nkvh, nq);
            ggml_backend_sched_free(sched); ggml_free(ctx); continue;
        }

        std::vector<float> qs((size_t) D * nq * nh);
        std::uniform_real_distribution<float> uq(-1.0f, 1.0f);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float & x : qs) x = real ? 2.0f * nd(rng) : uq(rng);
        // DEC_FA_INV: query j of head h does not depend on nq (batch-invariance test; compare dumps across nq)
        const bool inv = getenv("DEC_FA_INV") != nullptr;
        if (inv) {
            for (int h = 0; h < nh; ++h) {
                for (int j = 0; j < nq; ++j) {
                    std::mt19937 r2(1000 + 100 * h + j);
                    for (int d = 0; d < D; ++d) qs[((size_t) h * nq + j) * D + d] = 2.0f * nd(r2);
                }
            }
            rng.seed(4242); // K/V must not depend on nq either
        }
        ggml_backend_tensor_set(q, qs.data(), 0, qs.size() * sizeof(float));

        // hot tokens (real data): K row = sign pattern of a Q head's vector, so its logit is large for that head
        std::vector<int64_t> hot;
        if (real) {
            for (int i = 0; i < 64; ++i) hot.push_back((int64_t) (rng() % n_kv));
        }

        const size_t nbytes = ggml_nbytes(k);
        std::vector<uint8_t> kb(nbytes), vb(nbytes);
        if (f16) {
            ggml_fp16_t * K16 = (ggml_fp16_t *) kb.data();
            ggml_fp16_t * V16 = (ggml_fp16_t *) vb.data();
            const size_t n = nbytes / 2;
            for (size_t i = 0; i < n; ++i) {
                K16[i] = ggml_fp32_to_fp16(real ? 0.15f * nd(rng) : 0.02f * (float) ((int) (rng() % 16) - 8));
                V16[i] = ggml_fp32_to_fp16(real ? 0.15f * nd(rng) : 0.02f * (float) ((int) (rng() % 16) - 8));
            }
        } else {
            // q4_0 block: fp16 scale + 16 bytes of nibbles
            const ggml_fp16_t d16 = ggml_fp32_to_fp16(0.02f);
            std::uniform_real_distribution<float> ud(0.005f, 0.06f);
            for (size_t b = 0; b + 18 <= nbytes; b += 18) {
                const ggml_fp16_t dk = real ? ggml_fp32_to_fp16(ud(rng)) : d16;
                const ggml_fp16_t dv = real ? ggml_fp32_to_fp16(ud(rng)) : d16;
                memcpy(&kb[b], &dk, 2);
                memcpy(&vb[b], &dv, 2);
                for (int i = 2; i < 18; ++i) { kb[b + i] = (uint8_t) rng(); vb[b + i] = (uint8_t) rng(); }
            }
            if (!real) vb = kb; // the original bench used identical K and V
        }
        if (real) {
            // K rows of hot tokens: head kh, token tt aligned with Q head kh*6 + (i % 6) of query 0
            const size_t row = (size_t) D / (f16 ? 1 : 32) * (f16 ? 2 : 18);
            for (size_t i = 0; i < hot.size(); ++i) {
                const int kh = (int) (i % nkvh);
                const int hq = kh * 6 + (int) ((i / nkvh) % 6);
                const float * qv = qs.data() + (size_t) hq * nq * D; // query 0 of head hq
                uint8_t * r = kb.data() + ((size_t) kh * n_kv + hot[i]) * row;
                if (f16) {
                    for (int d = 0; d < D; ++d) ((ggml_fp16_t *) r)[d] = ggml_fp32_to_fp16(qv[d] > 0 ? 0.25f : -0.25f);
                } else {
                    const ggml_fp16_t d16 = ggml_fp32_to_fp16(0.05f);
                    for (int blk = 0; blk < D / 32; ++blk) {
                        uint8_t * bp = r + blk * 18;
                        memcpy(bp, &d16, 2);
                        for (int j = 0; j < 16; ++j) {
                            const int lo = qv[blk * 32 + j]      > 0 ? 13 : 3;
                            const int hi = qv[blk * 32 + j + 16] > 0 ? 13 : 3;
                            bp[2 + j] = (uint8_t) (lo | (hi << 4));
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(k, kb.data(), 0, nbytes);
        ggml_backend_tensor_set(v, vb.data(), 0, nbytes);

        std::vector<ggml_fp16_t> ms((size_t) n_kv * nq, ggml_fp32_to_fp16(0.0f));
        if (real) {
            // query token j sees tokens < n_kv - 100 + 7*j (masked tail), plus a small random bias elsewhere
            std::uniform_real_distribution<float> ub(-0.5f, 0.0f);
            for (int j = 0; j < nq; ++j) {
                // causal verify rows: row j sees valid_len - nq + j + 1 positions
                const int64_t lim = inv ? n_kv - 50 - 3 * j : valid_len > 0 ? valid_len - nq + j + 1 : n_kv - 100 + 7 * j;
                std::mt19937 r3(7 + j);
                for (int64_t tt = 0; tt < n_kv; ++tt) {
                    ms[(size_t) j * n_kv + tt] = tt >= lim ? ggml_fp32_to_fp16(-INFINITY) : ggml_fp32_to_fp16(ub(inv ? r3 : rng));
                }
            }
        }
        ggml_backend_tensor_set(m, ms.data(), 0, ms.size() * sizeof(ggml_fp16_t));

        ggml_backend_sched_graph_compute(sched, gf); // warm-up
        std::vector<double> t;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_s();
            ggml_backend_sched_graph_compute(sched, gf);
            ggml_backend_synchronize(cu);
            t.push_back((now_s() - t0) * 1e3);
        }
        if (getenv("DEC_FA_REF") && n_kv <= 65536) {
            // double-precision CPU reference on the exact dequantized K/V; out layout [D, nh, nq]
            std::vector<float> out(ggml_nelements(o));
            ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
            auto deq = [&](const std::vector<uint8_t> & buf, int kh, int64_t tt, std::vector<double> & r) {
                if (f16) {
                    const ggml_fp16_t * p = (const ggml_fp16_t *) buf.data() + ((size_t) kh * n_kv + tt) * D;
                    for (int d = 0; d < D; ++d) r[d] = ggml_fp16_to_fp32(p[d]);
                } else {
                    const uint8_t * p = buf.data() + ((size_t) kh * n_kv + tt) * 144;
                    for (int blk = 0; blk < D / 32; ++blk) {
                        ggml_fp16_t d16; memcpy(&d16, p + blk * 18, 2);
                        const double dd = ggml_fp16_to_fp32(d16);
                        for (int j = 0; j < 16; ++j) {
                            r[blk * 32 + j]      = ((p[blk * 18 + 2 + j] & 15) - 8) * dd;
                            r[blk * 32 + j + 16] = ((p[blk * 18 + 2 + j] >> 4) - 8) * dd;
                        }
                    }
                }
            };
            double maxd = 0.0, maxa = 0.0;
            std::vector<double> kr(D), vr(D), s(n_kv), acc(D);
            for (int hq = 0; hq < nh; ++hq) {
                const int kh = hq / (nh / nkvh);
                for (int j = 0; j < nq; ++j) {
                    const float * qv = qs.data() + ((size_t) hq * nq + j) * D;
                    double mx = -INFINITY;
                    for (int64_t tt = 0; tt < n_kv; ++tt) {
                        deq(kb, kh, tt, kr);
                        double dot = 0.0;
                        for (int d = 0; d < D; ++d) dot += qv[d] * kr[d];
                        s[tt] = dot / 16.0 + ggml_fp16_to_fp32(ms[(size_t) j * n_kv + tt]);
                        mx = std::max(mx, s[tt]);
                    }
                    double den = 0.0;
                    std::fill(acc.begin(), acc.end(), 0.0);
                    for (int64_t tt = 0; tt < n_kv; ++tt) {
                        const double p = std::exp(s[tt] - mx);
                        if (p == 0.0) continue;
                        den += p;
                        deq(vb, kh, tt, vr);
                        for (int d = 0; d < D; ++d) acc[d] += p * vr[d];
                    }
                    for (int d = 0; d < D; ++d) {
                        const double ref = acc[d] / den;
                        maxd = std::max(maxd, std::fabs(ref - out[((size_t) j * nh + hq) * D + d]));
                        maxa = std::max(maxa, std::fabs(ref));
                    }
                }
            }
            printf("     nq=%d vs fp64 reference: max|diff|=%.3e max|ref|=%.3e rel=%.3e\n", nq, maxd, maxa, maxd / maxa);
        }
        if (dump) {
            std::vector<float> out(ggml_nelements(o));
            ggml_backend_tensor_get(o, out.data(), 0, out.size() * sizeof(float));
            char path[1024];
            snprintf(path, sizeof(path), "%s_%d_%d_%d.f32", dump, nh, nkvh, nq);
            FILE * f = fopen(path, "wb");
            if (!f) { fprintf(stderr, "cannot write %s\n", path); return 1; }
            fwrite(out.data(), sizeof(float), out.size(), f);
            fclose(f);
        }
        std::sort(t.begin(), t.end());
        const double kv_mib = 2.0 * nbytes / (1024.0 * 1024.0);
        printf("%4d %4d %3d %9.3f %9.3f %12.1f %12.1f\n", nh, nkvh, nq, t[0], t[t.size() / 2], kv_mib,
               2.0 * nbytes / (t[0] * 1e-3) / 1e9);
        fflush(stdout);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    ggml_backend_free(cpu);
    ggml_backend_free(cu);
    return 0;
}
