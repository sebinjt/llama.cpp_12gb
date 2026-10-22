#include "fattn-decode-q4.cuh"
#include "cp-async.cuh"

#include <cstdlib>

// Split-K flash decoding over 64-token chunks for q4_0 K and V, D = 256, GQA 6, 1-8 query tokens. CUDA/Ampere port of
// the design in amd_ref/llama.cpp/ggml/src/ggml-cuda/fattn-paged-decode-kernel.cuh (contiguous K/V, no page table).
//
// One block = one KV head x a range of chunks, 8 warps. Rows are (query token, GQA head) pairs packed into 1, 2 or 4
// 16-row mma tiles (n_q * 6 <= 64), so each K/V row is read from DRAM once for all 6 Q heads and all query tokens.
// Per chunk:
//   1. raw q4_0 K and V rows (and the mask columns) go to shared memory with cp.async, NSTAGES chunks in flight;
//   2. S = Q K^T with mma.sync m16n8k16 (f16 x f16 -> f32), K dequantized to f16 while building the B fragments;
//   3. online softmax by row through shared memory (P stored as f16, per-row rescale factor);
//   4. O = O * alpha + P V with mma.sync, V dequantized while building the B fragments; warp w owns 32 output dims.
// A second kernel combines the splits.
//
// Dimension permutations (they cancel because both mma operands use the same one):
//   S: within q4_0 block b, k-step 2b takes the low nibbles and 2b+1 the high nibbles. Lane (g, t) needs logical k
//      2t, 2t+1, 2t+8, 2t+9; these map to dims 4t..4t+3 (+16 for the high nibbles), i.e. one aligned-ish 32-bit word of
//      the block's quants per lane. Q is loaded with the same mapping.
//   PV: warp w owns q4_0 block w of V (dims 32w..32w+31). Lane g reads the 16-bit quant pair at bytes 2g, 2g+1; n-tile
//      0..3 column g is dim 2g, 2g+1, 16+2g, 17+2g. The output write undoes it.

#define DQ4_D       256
#define DQ4_GQA     6
#define DQ4_CHUNK   64
#define DQ4_NWARPS  8
#define DQ4_ROW     144                                     // q4_0 bytes per 256-dim row
#define DQ4_PIECES  (DQ4_ROW / 16)                          // 9 x 16 B per row
#define DQ4_MAXQ    8                                       // decode: query tokens in one launch tile (1, 2 or 4 16-row tiles)
#define DQ4_TILE_TOK 10                                     // n_q > 8 (prefill): query tokens per row tile on grid z (60 rows)
#define DQ4_MAXQ_ALL 512                                    // query tokens per op
#define DQ4_KV_B    (DQ4_CHUNK * DQ4_ROW)                   // 9216 B of K (or V) per chunk
#define DQ4_STAGE_B (2 * DQ4_KV_B + DQ4_TILE_TOK * DQ4_CHUNK * 2) // K, V, mask rows of one row tile
#define DQ4_SS      72                                      // S row stride (floats)
#define DQ4_PS      72                                      // P row stride (halves)

static __host__ __device__ constexpr int dq4_smem_bytes(const int m_tiles, const int nstages) {
    return nstages * DQ4_STAGE_B + m_tiles * 16 * DQ4_SS * 4 + m_tiles * 16 * DQ4_PS * 2 + m_tiles * 16 * 4;
}

static __device__ __forceinline__ void dq4_mma(float * c, const uint32_t * a, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED(c); GGML_UNUSED(a); GGML_UNUSED(b0); GGML_UNUSED(b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void dq4_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template <int n>
static __device__ __forceinline__ void dq4_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ uint32_t dq4_h2u(const half2 h) {
    return *(const uint32_t *) &h;
}

// Bytes 0x0N in x -> half(1024 + N) in the 16-bit lanes selected by sel, then (N - 8) * d.
static __device__ __forceinline__ uint32_t dq4_deq(const uint32_t x, const uint32_t sel, const half2 d) {
    const uint32_t u = __byte_perm(x, 0x64646464u, sel);
    const half2 v = __hsub2(*(const half2 *) &u, __half2half2(__float2half(1032.0f)));
    return dq4_h2u(__hmul2(v, d));
}

template <int M_TILES, int NSTAGES>
__launch_bounds__(DQ4_NWARPS * 32, 1)
static __global__ void dq4_partial(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const char * __restrict__ K, const int64_t k_nb1, const int64_t k_nb2,
        const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const char * __restrict__ mask, const int64_t mask_nb1,
        const float scale, const int nb_all, const int n_chunks, const int chunks_per_split,
        float * __restrict__ O_part, float2 * __restrict__ ML_part) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    // row tile on grid z: query tokens [tok0, tok0 + nb); a decode (n_q <= 8) is a single tile with tok0 = 0
    const int tok0 = blockIdx.z * DQ4_TILE_TOK;
    const int nb   = gridDim.z == 1 ? nb_all : min(DQ4_TILE_TOK, nb_all - tok0);
    Q    += tok0 * q_nb1;
    mask += tok0 * mask_nb1;
    constexpr int R   = M_TILES * 16;           // rows
    constexpr int TPR = DQ4_NWARPS * 32 / R;    // softmax threads per row
    constexpr int TOK = DQ4_CHUNK / TPR;        // tokens per softmax thread
    constexpr int WPM  = DQ4_NWARPS / M_TILES;  // S phase: warps per m-tile
    constexpr int TPW  = DQ4_CHUNK / WPM;       // S phase: tokens per warp
    constexpr int NT_S = TPW / 8;               // S phase: 8-token n-tiles per warp
    static_assert(M_TILES == 1 || M_TILES == 2 || M_TILES == 4, "M_TILES");

    extern __shared__ __align__(16) char dq4_smem[];
    float * Ss      = (float *) (dq4_smem + NSTAGES * DQ4_STAGE_B);
    half  * Ps      = (half  *) (Ss + R * DQ4_SS);
    float * alpha_s = (float *) (Ps + R * DQ4_PS);

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g    = lane / 4;
    const int t    = lane % 4;
    const int head  = blockIdx.y;
    const int split = blockIdx.x;
    const int c0 = split * chunks_per_split;
    const int c1 = min(n_chunks, c0 + chunks_per_split);
    const int n_rows = nb * DQ4_GQA;

    const char * Kh = K + head * k_nb2;
    const char * Vh = V + head * v_nb2;

    auto fetch = [&](const int c, const int s) {
        char * st = dq4_smem + s * DQ4_STAGE_B;
        const unsigned int st_s = ggml_cuda_cvta_generic_to_shared(st);
        const int64_t t0 = (int64_t) c * DQ4_CHUNK;
#pragma unroll
        for (int j = 0; j < (DQ4_CHUNK * DQ4_PIECES + DQ4_NWARPS * 32 - 1) / (DQ4_NWARPS * 32); ++j) {
            const int i = tid + j * DQ4_NWARPS * 32;
            if (i < DQ4_CHUNK * DQ4_PIECES) {
                const int r = i / DQ4_PIECES, col = i % DQ4_PIECES;
                cp_async_cg_16<0>(st_s + i * 16,            Kh + (t0 + r) * k_nb1 + col * 16);
                cp_async_cg_16<0>(st_s + DQ4_KV_B + i * 16, Vh + (t0 + r) * v_nb1 + col * 16);
            }
        }
        if (tid < nb * (DQ4_CHUNK * 2 / 16)) {
            const int q = tid / 8, col = tid % 8;
            cp_async_cg_16<0>(st_s + 2 * DQ4_KV_B + tid * 16, mask + q * mask_nb1 + t0 * 2 + col * 16);
        }
    };

    // Q fragments of this warp's m-tile (S phase), scaled, in the permuted dim order.
    const int s_mt = warp / WPM;
    uint32_t Qa[DQ4_D / 16][4];
    {
        const int rA = s_mt * 16 + g;
        const int rB = rA + 8;
        const float4 * qA = rA < n_rows ? (const float4 *) (Q + (rA / DQ4_GQA) * q_nb1 + (head * DQ4_GQA + rA % DQ4_GQA) * q_nb2) : nullptr;
        const float4 * qB = rB < n_rows ? (const float4 *) (Q + (rB / DQ4_GQA) * q_nb1 + (head * DQ4_GQA + rB % DQ4_GQA) * q_nb2) : nullptr;
#pragma unroll
        for (int ks = 0; ks < DQ4_D / 16; ++ks) {
            const int d0 = 32 * (ks / 2) + 16 * (ks % 2) + 4 * t;
            const float4 a = qA ? qA[d0 / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 b = qB ? qB[d0 / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            Qa[ks][0] = dq4_h2u(make_half2(a.x * scale, a.y * scale));
            Qa[ks][1] = dq4_h2u(make_half2(b.x * scale, b.y * scale));
            Qa[ks][2] = dq4_h2u(make_half2(a.z * scale, a.w * scale));
            Qa[ks][3] = dq4_h2u(make_half2(b.z * scale, b.w * scale));
        }
    }

    const int sm_row = tid / TPR;
    const int sm_sub = tid % TPR;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    float O[M_TILES][4][4];
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                O[mt][nt][l] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) {
        if (c0 + s < c1) {
            fetch(c0 + s, s);
        }
        dq4_commit();
    }

    for (int c = c0; c < c1; ++c) {
        const int stage = (c - c0) % NSTAGES;
        {
            // the stage being refilled was last read in iteration c - 1, which ended with a barrier
            const int cn = c + NSTAGES - 1;
            if (cn < c1) {
                fetch(cn, (cn - c0) % NSTAGES);
            }
            dq4_commit();
        }
        dq4_wait<NSTAGES - 1>();
        __syncthreads();

        const char * Ksm = dq4_smem + stage * DQ4_STAGE_B;
        const char * Vsm = Ksm + DQ4_KV_B;
        const half * Msm = (const half *) (Ksm + 2 * DQ4_KV_B);

        // 2. S = Q K^T.
#pragma unroll
        for (int nt = 0; nt < NT_S; ++nt) {
            const int n0 = (warp % WPM) * TPW + nt * 8;
            const char * krow = Ksm + (n0 + g) * DQ4_ROW;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int b = 0; b < DQ4_D / 32; ++b) {
                const int off = 18 * b + 2 + 4 * t;
                uint32_t w;
                if (b % 2 == 0) { // 2 mod 4
                    const uint32_t lo = *(const uint32_t *) (krow + off - 2);
                    const uint32_t hi = *(const uint32_t *) (krow + off + 2);
                    w = __funnelshift_r(lo, hi, 16);
                } else {
                    w = *(const uint32_t *) (krow + off);
                }
                const half2 d2 = __half2half2(*(const half *) (krow + 18 * b));
                const uint32_t lo4 = w & 0x0F0F0F0Fu;
                const uint32_t hi4 = (w >> 4) & 0x0F0F0F0Fu;
                dq4_mma(acc, Qa[2 * b + 0], dq4_deq(lo4, 0x4140, d2), dq4_deq(lo4, 0x4342, d2));
                dq4_mma(acc, Qa[2 * b + 1], dq4_deq(hi4, 0x4140, d2), dq4_deq(hi4, 0x4342, d2));
            }
            const int n = n0 + 2 * t;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = s_mt * 16 + g + 8 * h;
                float2 v;
                if (r < n_rows) {
                    const half2 m2 = *(const half2 *) (Msm + (r / DQ4_GQA) * DQ4_CHUNK + n);
                    v = make_float2(acc[2 * h + 0] + __low2float(m2), acc[2 * h + 1] + __high2float(m2));
                } else {
                    v = make_float2(-INFINITY, -INFINITY);
                }
                *(float2 *) (Ss + r * DQ4_SS + n) = v;
            }
        }
        __syncthreads();

        // 3. Online softmax by row. The row sum uses one fixed order for every M_TILES (batch invariance: a row's
        //    result does not depend on how many query tokens share the launch): 16 groups of 4 consecutive tokens,
        //    each summed in order, then a butterfly that pairs group G with G ^ 8, ^ 4, ^ 2, ^ 1. Thread sm_sub owns
        //    groups sm_sub + i * TPR, so the first levels are local and the rest are shuffles.
        {
            constexpr int GPT = TOK / 4; // groups per thread
            const float * srow = Ss + sm_row * DQ4_SS;
            half * prow = Ps + sm_row * DQ4_PS;
            float sv[GPT][4];
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < GPT; ++i) {
                const int t0 = 4 * (sm_sub + i * TPR);
                const float4 v = *(const float4 *) (srow + t0);
                sv[i][0] = v.x; sv[i][1] = v.y; sv[i][2] = v.z; sv[i][3] = v.w;
                mx = fmaxf(mx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, o, 32));
            }
            const float m_new = fmaxf(m_run, mx);
            float gs[GPT];
#pragma unroll
            for (int i = 0; i < GPT; ++i) {
                float p[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    p[j] = m_new == -INFINITY ? 0.0f : __expf(sv[i][j] - m_new);
                }
                gs[i] = ((p[0] + p[1]) + p[2]) + p[3];
                const int t0 = 4 * (sm_sub + i * TPR);
                *(half2 *) (prow + t0 + 0) = make_half2(p[0], p[1]);
                *(half2 *) (prow + t0 + 2) = make_half2(p[2], p[3]);
            }
#pragma unroll
            for (int off = 8; off >= TPR; off /= 2) { // local levels: group i pairs with i + off / TPR
#pragma unroll
                for (int i = 0; i < off / TPR; ++i) {
                    gs[i] += gs[i + off / TPR];
                }
            }
            float sum = gs[0];
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, o, 32);
            }
            const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_run - m_new);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if (sm_sub == 0) {
                alpha_s[sm_row] = alpha;
            }
        }
        __syncthreads();

        // 4. O = O * alpha + P V for this warp's 32 dims (q4_0 block `warp` of each V row).
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const float aA = alpha_s[mt * 16 + g];
            const float aB = alpha_s[mt * 16 + g + 8];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                O[mt][nt][0] *= aA;
                O[mt][nt][1] *= aA;
                O[mt][nt][2] *= aB;
                O[mt][nt][3] *= aB;
            }
        }
#pragma unroll
        for (int ks = 0; ks < DQ4_CHUNK / 16; ++ks) {
            const int k0 = ks * 16;
            uint32_t bfr[4][2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int tk = k0 + 2 * t + 8 * h; // tokens tk, tk + 1
                const char * v0 = Vsm + (tk + 0) * DQ4_ROW + 18 * warp;
                const char * v1 = Vsm + (tk + 1) * DQ4_ROW + 18 * warp;
                const uint32_t x = (uint32_t) *(const uint16_t *) (v0 + 2 + 2 * g) |
                                  ((uint32_t) *(const uint16_t *) (v1 + 2 + 2 * g) << 16);
                const half2 d2 = __halves2half2(*(const half *) v0, *(const half *) v1);
                const uint32_t lo4 = x & 0x0F0F0F0Fu;
                const uint32_t hi4 = (x >> 4) & 0x0F0F0F0Fu;
                bfr[0][h] = dq4_deq(lo4, 0x4240, d2);
                bfr[1][h] = dq4_deq(lo4, 0x4341, d2);
                bfr[2][h] = dq4_deq(hi4, 0x4240, d2);
                bfr[3][h] = dq4_deq(hi4, 0x4341, d2);
            }
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const half * pA = Ps + (mt * 16 + g) * DQ4_PS + k0 + 2 * t;
                const half * pB = pA + 8 * DQ4_PS;
                const uint32_t a[4] = {
                    *(const uint32_t *) (pA), *(const uint32_t *) (pB),
                    *(const uint32_t *) (pA + 8), *(const uint32_t *) (pB + 8),
                };
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    dq4_mma(O[mt][nt], a, bfr[nt][0], bfr[nt][1]);
                }
            }
        }
        __syncthreads();
    }

    // Partial results for valid rows: unnormalized O plus (max, sum) per row.
    const int64_t base = (((int64_t) split * gridDim.y + head) * gridDim.z + blockIdx.z) * R;
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int dim = warp * 32 + (nt / 2) * 16 + 4 * t + (nt % 2); // columns 2t, 2t + 1 -> dims +0, +2
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = mt * 16 + g + 8 * h;
                if (r < n_rows) {
                    float * o = O_part + (base + r) * DQ4_D + dim;
                    o[0] = O[mt][nt][2 * h + 0];
                    o[2] = O[mt][nt][2 * h + 1];
                }
            }
        }
    }
    if (sm_sub == 0 && sm_row < n_rows) {
        ML_part[base + sm_row] = make_float2(m_run, l_run);
    }
#else
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

// One block per (row, KV head), one thread per output dim.
static __global__ void dq4_combine(
        const float * __restrict__ O_part, const float2 * __restrict__ ML_part, const int n_splits, const int R,
        const int n_tiles, char * __restrict__ dst, const int64_t dst_nb1, const int64_t dst_nb2) {
    const int r = blockIdx.x;
    const int head = blockIdx.y;
    const int n_heads = gridDim.y;
    // row tile of this row (n_tiles == 1 for a decode: tile 0, local row r)
    const int tile = n_tiles == 1 ? 0 : (r / DQ4_GQA) / DQ4_TILE_TOK;
    const int rl   = r - tile * DQ4_TILE_TOK * DQ4_GQA;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        M = fmaxf(M, ML_part[(((int64_t) s * n_heads + head) * n_tiles + tile) * R + rl].x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const int64_t i = (((int64_t) s * n_heads + head) * n_tiles + tile) * R + rl;
        const float2 ml = ML_part[i];
        const float w = ml.x == -INFINITY ? 0.0f : expf(ml.x - M);
        num += w * O_part[i * DQ4_D + threadIdx.x];
        den += w * ml.y;
    }
    float * out = (float *) (dst + (r / DQ4_GQA) * dst_nb2 + (head * DQ4_GQA + r % DQ4_GQA) * dst_nb1);
    out[threadIdx.x] = den == 0.0f ? 0.0f : num / den;
}

static bool dq4_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_FA_DECODE_Q4");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
}

// nullptr when dst has a shape the kernel takes, else the first failed condition
static const char * dq4_reject_reason(int device, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!ampere_mma_available(cc) || !cp_async_available(cc)) return "arch";

    float params[3];
    memcpy(params, dst->op_params, sizeof(params)); // scale, max_bias, logit_softcap
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 || dst->type != GGML_TYPE_F32) return "types";
    if (Q->ne[0] != DQ4_D || K->ne[0] != DQ4_D || V->ne[0] != DQ4_D) return "head size";
    static const int64_t max_q = [] { // LLAMA_FA_DECODE_Q4_MAXQ: largest n_q taken (A/B of the prefill path)
        const char * e = getenv("LLAMA_FA_DECODE_Q4_MAXQ");
        return e && atoi(e) > 0 ? std::min<int64_t>(atoi(e), DQ4_MAXQ_ALL) : (int64_t) DQ4_MAXQ_ALL;
    }();
    if (Q->ne[1] < 1 || Q->ne[1] > max_q || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return "query tokens / ne3";
    if (K->ne[2] < 1 || Q->ne[2] != DQ4_GQA * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) return "gqa";
    if (K->ne[1] % DQ4_CHUNK != 0) return "n_kv % 64";
    if (dst->src[4] || params[1] != 0.0f || params[2] != 0.0f) return "sinks/alibi/softcap";
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
            mask->ne[2] != 1 || mask->ne[3] != 1) return "mask";
    if (K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16 || mask->nb[1] % 16) return "strides";
    if ((uintptr_t) K->data % 16 || (uintptr_t) V->data % 16 || (uintptr_t) mask->data % 16 || (uintptr_t) Q->data % 16) return "alignment";
    if (Q->nb[0] != sizeof(float) || Q->nb[1] % 16 || Q->nb[2] % 16 || dst->nb[0] != sizeof(float)) return "q/dst strides";
    return nullptr;
}

bool ggml_cuda_fattn_decode_q4_supported(int device, const ggml_tensor * dst) {
    if (!dq4_enabled()) {
        return false;
    }
    const char * why = dq4_reject_reason(device, dst);
    static const bool debug = getenv("LLAMA_FA_DECODE_Q4_DEBUG") != nullptr;
    if (debug && why && dst->src[0]->ne[0] == DQ4_D && dst->src[0]->ne[1] <= 8) {
        const ggml_tensor * Q = dst->src[0];
        const ggml_tensor * K = dst->src[1];
        const ggml_tensor * m = dst->src[3];
        GGML_LOG_WARN("%s: %s rejected (%s): Q %lld x %lld x %lld, K %s %lld x %lld x %lld, mask %s %lld x %lld\n", __func__,
            dst->name, why, (long long) Q->ne[0], (long long) Q->ne[1], (long long) Q->ne[2], ggml_type_name(K->type),
            (long long) K->ne[0], (long long) K->ne[1], (long long) K->ne[2], m ? ggml_type_name(m->type) : "none",
            m ? (long long) m->ne[0] : 0LL, m ? (long long) m->ne[1] : 0LL);
    }
    return why == nullptr;
}

template <int M_TILES, int NSTAGES>
static void dq4_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();
    const int device = ctx.device;

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    constexpr int R = M_TILES * 16;
    constexpr int smem = dq4_smem_bytes(M_TILES, NSTAGES);
    static int blocks_per_sm[GGML_CUDA_MAX_DEVICES] = {0};
    if (blocks_per_sm[device] == 0) {
        CUDA_CHECK(cudaFuncSetAttribute(dq4_partial<M_TILES, NSTAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        int n = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, dq4_partial<M_TILES, NSTAGES>, DQ4_NWARPS * 32, smem));
        blocks_per_sm[device] = std::max(1, n);
    }

    const int nb       = (int) Q->ne[1];
    const int n_heads  = (int) K->ne[2];
    const int n_chunks = (int) (K->ne[1] / DQ4_CHUNK);

    static const int splits_env = [] {
        const char * e = getenv("LLAMA_FA_DECODE_Q4_SPLITS"); // splits per KV head (tuning)
        return e ? atoi(e) : 0;
    }();
    // The split partition must not depend on the current n_kv (or on n_q / M_TILES): a verify batch can see a larger
    // padded n_kv than a single-token decode at the same position, and a different partition changes the fp order
    // of the combine. So the chunks per split come from the size of the whole cache tensor K views (constant for a
    // context); splits past the valid range are fully masked and add exact zeros in the combine.
    const ggml_tensor * K_root = K;
    while (K_root->view_src) {
        K_root = K_root->view_src;
    }
    const int64_t n_ref = K_root != K && K_root->ne[1] >= K->ne[1] ? K_root->ne[1] : K->ne[1];
    const int n_chunks_ref = (int) ((n_ref + DQ4_CHUNK - 1) / DQ4_CHUNK);
    const int want = ggml_cuda_info().devices[device].nsm; // 1 block/SM for every variant (do not use blocks_per_sm)
    // Opt-in: fix the split partition in TOKENS instead of deriving it from the cache size, so that the fp32
    // combine order (and therefore the output) does not depend on the -c value. Default 0 keeps the old
    // behaviour byte-identically. LLAMA_FA_Q4_SPLIT_TOKENS=16384 means "one split per 16384 tokens of KV".
    static const int split_tokens_env = [] {
        const char * e = getenv("LLAMA_FA_Q4_SPLIT_TOKENS");
        return e ? atoi(e) : 0;
    }();
    int chunks_per_split = 0;
    if (split_tokens_env > 0) {
        chunks_per_split = std::max(1, (split_tokens_env + DQ4_CHUNK - 1) / DQ4_CHUNK);
    } else {
        int n_splits_ref = splits_env > 0 ? splits_env : (want + n_heads - 1) / n_heads;
        n_splits_ref = std::max(1, std::min(n_chunks_ref, n_splits_ref));
        chunks_per_split = (n_chunks_ref + n_splits_ref - 1) / n_splits_ref;
    }
    const int n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

    // query tokens beyond one launch tile (prefill) are split into row tiles of DQ4_TILE_TOK tokens on grid z
    const int n_tiles = nb <= DQ4_TILE_TOK ? 1 : (nb + DQ4_TILE_TOK - 1) / DQ4_TILE_TOK;
    GGML_ASSERT(n_tiles == 1 || M_TILES == 4);

    ggml_cuda_pool_alloc<float>  O_part(ctx.pool(), (size_t) n_splits * n_heads * n_tiles * R * DQ4_D);
    ggml_cuda_pool_alloc<float2> ML_part(ctx.pool(), (size_t) n_splits * n_heads * n_tiles * R);

    dq4_partial<M_TILES, NSTAGES><<<dim3(n_splits, n_heads, n_tiles), DQ4_NWARPS * 32, smem, stream>>>(
        (const char *) Q->data, Q->nb[1], Q->nb[2],
        (const char *) K->data, K->nb[1], K->nb[2],
        (const char *) V->data, V->nb[1], V->nb[2],
        (const char *) mask->data, mask->nb[1],
        scale, nb, n_chunks, chunks_per_split, O_part.get(), ML_part.get());
    CUDA_CHECK(cudaGetLastError());
    dq4_combine<<<dim3(nb * DQ4_GQA, n_heads), DQ4_D, 0, stream>>>(O_part.get(), ML_part.get(), n_splits, R, n_tiles,
        (char *) dst->data, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_ext_decode_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const int nstages = [] {
        const char * e = getenv("LLAMA_FA_DECODE_Q4_STAGES"); // 2 or 3 (tuning)
        return e && atoi(e) == 3 ? 3 : 2;
    }();
    const int nb = (int) dst->src[0]->ne[1];
    if (nb * DQ4_GQA <= 16) {
        if (nstages == 3) { dq4_launch<1, 3>(ctx, dst); } else { dq4_launch<1, 2>(ctx, dst); }
    } else if (nb * DQ4_GQA <= 32) {
        if (nstages == 3) { dq4_launch<2, 3>(ctx, dst); } else { dq4_launch<2, 2>(ctx, dst); }
    } else {
        dq4_launch<4, 2>(ctx, dst);
    }
}
