// mmsb_bench.cu - Step 1 feasibility gate for a small-batch (1..16 column) quantized matmul kernel.
//
// Standalone microbenchmark, no model code. For each weight type (IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS) and each real
// shape it measures, on the real tensor layout:
//   (a) LOAD    : weight bytes streamed with the planned access pattern  -> attainable GB/s
//       STREAM  : contiguous 16 B/lane stream over the whole tensor     -> bandwidth ceiling for this size
//   (b) UNPACK  : load + full unpack of every weight to int8 in registers (no math besides a cheap sum)
//   (c) FULL    : unpack + int8 tensor-core mma + fp32 sub-block scale, at ncols 1,2,4,8,12,16
// plus a correctness check of the FULL kernel against ggml's own dequantizers + a scalar dot, and a
// per-column bit-exactness check (column invariance) of the f32 outputs for ncols 1..16.
//
// Build: tests\build_mmsb_bench.bat        Run: tests\bin\mmsb_bench.exe [--shapes a,b] [--types 0,1,2,3]
//
// Design of the access path that is measured (this is the candidate production design):
//   * one warp owns 16 output rows and all K; a warps walks K in 256-element superblocks.
//   * the A operand of mma.m16n8k32 is built directly in registers from the packed weights: with
//     t = lane%4, gr = lane/4, the thread needs k = 4t..4t+3 (group A) and 4t+16..4t+19 (group B) of
//     rows gr and gr+8, which is exactly 4 packed bytes + 1 qh byte + 2 sign bytes per row per 32-k block,
//     i.e. 4-byte loads on the real layout (rows are 8 B aligned, never misaligned).
//   * sign application with a 256-entry (low,high nibble) mask table: 1 load + 1 vsub4 per 4 values.
//   * mma.m16n8k32 for IQ3_S/IQ3_XXS/IQ4_XS (one scale per 32-k block); IQ2_S has two scales per 32-k
//     block so it uses two mma.m16n8k16 (k 0..15 and 16..31), one per scale.
//   * int32 dot per 32-k block -> fp32 accumulate with d_w * d_a, exactly the MMQ math.
//
// Nothing here touches the production tree; only reads ggml-base.dll for the reference dequantizers.

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------------------------
// ggml CPU reference dequantizers (exported by ggml-base.dll)
// ---------------------------------------------------------------------------------------------
extern "C" {
void dequantize_row_iq3_s  (const void * x, float * y, int64_t k);
void dequantize_row_iq3_xxs(const void * x, float * y, int64_t k);
void dequantize_row_iq2_s  (const void * x, float * y, int64_t k);
void dequantize_row_iq4_xs (const void * x, float * y, int64_t k);
}

enum { TY_IQ3S = 0, TY_IQ3XXS = 1, TY_IQ2S = 2, TY_IQ4XS = 3 };
static const char * TYNAME[4] = { "IQ3_S", "IQ3_XXS", "IQ2_S", "IQ4_XS" };

static int blk_bytes_h(int ty) {
    switch (ty) {
        case TY_IQ3S:   return 110;   // d(2)+qs(64)+qh(8)+signs(32)+scales(4)
        case TY_IQ3XXS: return  98;   // d(2)+qs(96)
        case TY_IQ2S:   return  82;   // d(2)+qs(64)+qh(8)+scales(8)
        default:        return 136;   // d(2)+scales_h(2)+scales_l(4)+qs(128)
    }
}
template<int TY> struct BB { static constexpr int v = (TY==TY_IQ3S)?110:(TY==TY_IQ3XXS)?98:(TY==TY_IQ2S)?82:136; };

// ---------------------------------------------------------------------------------------------
// sign mask tables (device global, filled from the host)
//   g_mask8[c]   : x = per-byte 0xFF where bit 0..3 of c set (low nibble),  y = bits 4..7 (high nibble)
//   g_xxs_mask[i]: same for ksigns_iq2xs[i]  (IQ3_XXS sign code)
// ---------------------------------------------------------------------------------------------
__device__ uint2 g_mask8[256];
__device__ uint2 g_xxs_mask[128];

static uint32_t nib_mask(int bits) {                 // 4 values, bit j -> byte j
    uint32_t m = 0;
    for (int j = 0; j < 4; ++j) if (bits & (1 << j)) m |= 0xFFu << (8*j);
    return m;
}
static const uint8_t ksigns_iq2xs_h[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// ---------------------------------------------------------------------------------------------
// per-type device helpers: 32-k block access
//   every accessor takes `sb` = pointer to the row's 256-element superblock (packed bytes)
// ---------------------------------------------------------------------------------------------

// unaligned-safe 32-bit read (all block headers are 2-byte aligned only)
__device__ __forceinline__ uint32_t ld_u32_2a(const uint8_t * p) {
    return (uint32_t)(*(const uint16_t *)p) | ((uint32_t)(*(const uint16_t *)(p + 2)) << 16);
}

// IQ3_S: qs[64] at +2, qh[8] at +66, signs[32] at +74, scales[4] at +106
template<int TY> struct U;

template<> struct U<TY_IQ3S> {
    static __device__ __forceinline__ float dval(const uint8_t * sb) { return __half2float(*(const __half *)sb); }
    static __device__ __forceinline__ float scale(const uint8_t * sb, int b32) {
        const int nib = (sb[106 + (b32 >> 1)] >> (4 * (b32 & 1))) & 0xF;
        return 1.0f + 2.0f * nib;
    }
    // group i = t + 4*g  (g=0 -> k 4t..4t+3, g=1 -> k 4t+16..4t+19)
    static __device__ __forceinline__ int grp(const uint8_t * sb, int b32, int i, const uint2 * mt) {
        const uint8_t * qs = sb + 2 + 8*b32;
        const int qhb = sb[66 + b32];
        const int idx = qs[i] | (((qhb >> i) & 1) << 8);
        const uint2 m  = __ldg(mt + sb[74 + 4*b32 + (i >> 1)]);
        const int   mi = (int)((i & 1) ? m.y : m.x);
        return (int)__vsub4((int)__ldg(iq3s_grid + idx) ^ mi, mi);
    }
};

// IQ3_XXS: qs[96] at +2 (first 64 = grid bytes, next 32 = scales_and_signs)
template<> struct U<TY_IQ3XXS> {
    static __device__ __forceinline__ float dval(const uint8_t * sb) { return __half2float(*(const __half *)sb); }
    static __device__ __forceinline__ uint32_t aux(const uint8_t * sb, int b32) { return ld_u32_2a(sb + 66 + 4*b32); }
    static __device__ __forceinline__ float scale(const uint8_t * sb, int b32) {
        return 0.5f * (0.5f + (float)((aux(sb, b32) >> 28) & 0xF));
    }
    static __device__ __forceinline__ int grp(const uint8_t * sb, int b32, int i, const uint2 * mt) {
        const uint8_t * qs = sb + 2 + 8*b32;
        const uint2 m  = __ldg(mt + ((aux(sb, b32) >> (7 * (i >> 1))) & 127));
        const int   mi = (int)((i & 1) ? m.y : m.x);
        return (int)__vsub4((int)__ldg(iq3xxs_grid + qs[i]) ^ mi, mi);
    }
};

// IQ2_S: qs[64] at +2 (bytes 0..31 grid, 32..63 signs), qh[8] at +66, scales[8] at +74
// two scales per 32-k block: nibble 0 -> k 0..15, nibble 1 -> k 16..31
template<> struct U<TY_IQ2S> {
    static __device__ __forceinline__ float dval(const uint8_t * sb) { return __half2float(*(const __half *)sb); }
    static __device__ __forceinline__ float scale(const uint8_t * sb, int b32) {
        const int nib = (sb[74 + b32] >> (4 * (b32 & 1))) & 0xF;
        return 0.25f * (0.5f + nib);
    }
    static __device__ __forceinline__ float scale_half(const uint8_t * sb, int b32, int half) {
        const int nib = (sb[74 + b32] >> (4 * half)) & 0xF;
        return 0.25f * (0.5f + (float)nib);
    }
    static __device__ __forceinline__ int grp(const uint8_t * sb, int b32, int i, const uint2 * mt) {
        const int l = i >> 1;
        const int qhb = sb[66 + b32];
        const int idx = sb[2 + 4*b32 + l] | (((qhb >> (2*l)) & 3) << 8);
        const uint32_t gv = (uint32_t)(__ldg(iq2s_grid + idx) >> (32 * (i & 1)));
        const uint2 m  = __ldg(mt + sb[2 + 32 + 4*b32 + l]);
        const int   mi = (int)((i & 1) ? m.y : m.x);
        return (int)__vsub4((int)gv ^ mi, mi);
    }
};

// IQ4_XS: d(2) scales_h(2) scales_l[4] at +4 qs[128] at +8
__device__ __forceinline__ int2 iq4nl_lookup(uint32_t q4) {
    // kvalues_iq4nl = -127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t o[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t sh = 16u * i;
        const uint32_t lo = __byte_perm(t0, t1, q4 >> sh);
        const uint32_t hi = __byte_perm(t2, t3, q4 >> sh);
        o[i] = __byte_perm(lo, hi, sel >> sh);
    }
    // o[0] = {tab[n0],tab[n1],tab[n2],tab[n3]} of qs[4t],qs[4t+1]; o[1] = same for qs[4t+2],qs[4t+3]
    // nibbles alternate k, k+16 -> de-interleave
    return make_int2((int)__byte_perm(o[0], o[1], 0x6420), (int)__byte_perm(o[0], o[1], 0x7531));
}

template<> struct U<TY_IQ4XS> {
    static __device__ __forceinline__ float dval(const uint8_t * sb) { return __half2float(*(const __half *)sb); }
    static __device__ __forceinline__ float scale(const uint8_t * sb, int b32) {
        const uint32_t sh = (uint32_t)(*(const uint16_t *)(sb + 2));
        const int ls = ((sb[4 + (b32 >> 1)] >> (4 * (b32 & 1))) & 0xF) | (int)(((sh >> (2*b32)) & 3) << 4);
        return (float)(ls - 32);
    }
    static __device__ __forceinline__ void pair(const uint8_t * sb, int b32, int t, int & lo, int & hi) {
        const uint32_t q4 = *(const uint32_t *)(sb + 8 + 16*b32 + 4*t);
        const int2 v = iq4nl_lookup(q4);
        lo = v.x; hi = v.y;
    }
};

// fill the four A registers (a0 = row gr group A, a1 = row gr+8 group A, a2 = row gr group B, a3 = row gr+8 group B)
template<int TY>
__device__ __forceinline__ void unpack_A(const uint8_t * sbA, const uint8_t * sbB, int b32, int t,
                                         const uint2 * mt, int & a0, int & a1, int & a2, int & a3) {
    if constexpr (TY == TY_IQ4XS) {
        int loA, hiA, loB, hiB;
        U<TY_IQ4XS>::pair(sbA, b32, t, loA, hiA);
        U<TY_IQ4XS>::pair(sbB, b32, t, loB, hiB);
        a0 = loA; a1 = loB; a2 = hiA; a3 = hiB;
    } else {
        a0 = U<TY>::grp(sbA, b32, t + 0, mt);
        a1 = U<TY>::grp(sbB, b32, t + 0, mt);
        a2 = U<TY>::grp(sbA, b32, t + 4, mt);
        a3 = U<TY>::grp(sbB, b32, t + 4, mt);
    }
}

// ---------------------------------------------------------------------------------------------
// mma
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16832(int & d0, int & d1, int & d2, int & d3,
                                         int a0, int a1, int a2, int a3, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816(int & d0, int & d1, int & d2, int & d3,
                                         int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(b0));
}

// ---------------------------------------------------------------------------------------------
// (c) FULL kernel: unpack + mma + fp32 block scaling
// activation layout: Aq[kblock][16][32] int8, Ad[kblock][16] float  (16 = padded column count)
// ---------------------------------------------------------------------------------------------
// STAGE: 0 = read packed metadata straight from global (v1), 1 = stage the 16 rows x block bytes
// into shared memory with coalesced u16 loads, then unpack from shared (v3)
template<int TY, int STAGE>
__global__ void k_mma(const uint8_t * __restrict__ W, const int8_t * __restrict__ Aq,
                      const float * __restrict__ Ad, const uint2 * __restrict__ mt,
                      int R, int K, int rowbytes, int ncols, int ostride, float * __restrict__ out) {
    constexpr int BLK = BB<TY>::v;
    constexpr int SHIFT = 0;
    constexpr int RAW = (TY == TY_IQ4XS) ? 144 : 112;
    constexpr int NWARPS = 4;
    extern __shared__ uint8_t smem3[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int t = lane & 3, gr = lane >> 2;
    const int nb = K >> 8, rowtiles = R >> 4, ntiles = (ncols + 7) >> 3;
    uint8_t * sh_raw = smem3 + warp * 16 * RAW;
    for (int rt = blockIdx.x * NWARPS + warp; rt < rowtiles; rt += NWARPS * (int)gridDim.x) {
        const uint8_t * wA = W + (size_t)(rt*16 + gr    ) * rowbytes;
        const uint8_t * wB = W + (size_t)(rt*16 + gr + 8) * rowbytes;
        const uint8_t * base = W + (size_t)(rt*16) * rowbytes;
        float acc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * sbA; const uint8_t * sbB;
            if constexpr (STAGE) {
#pragma unroll
                for (int it = 0; it < 16*RAW/2/32; ++it) {
                    const int c = it*32 + lane;
                    const int row = c / (RAW/2), off = (c % (RAW/2)) * 2;
                    *(uint16_t *)(sh_raw + row*RAW + off + SHIFT) =
                        *(const uint16_t *)(base + (size_t)row * rowbytes + (size_t)b*BLK + off);
                }
                __syncwarp();
                sbA = sh_raw + gr*RAW;
                sbB = sh_raw + (gr+8)*RAW;
            } else {
                sbA = wA + (size_t)b * BLK;
                sbB = wB + (size_t)b * BLK;
            }
            const float dA = U<TY>::dval(sbA), dB = U<TY>::dval(sbB);
#pragma unroll
            for (int b32 = 0; b32 < 8; ++b32) {
                const int kb = b*8 + b32;
                const int8_t * ab = Aq + (size_t)kb * (16*32);
                const int B_CONST = (STAGE == 2) ? 0x01010101 : 0;
                const float swA = dA * U<TY>::scale(sbA, b32);
                const float swB = dB * U<TY>::scale(sbB, b32);
                int a0, a1, a2, a3;
                unpack_A<TY>(sbA, sbB, b32, t, mt, a0, a1, a2, a3);
                if constexpr (TY == TY_IQ2S) {
                    // two scales per 32-k block (k 0..15 and 16..31) -> two m16n8k16 mmas
                    const float w0A = dA * U<TY_IQ2S>::scale_half(sbA, b32, 0), w0B = dB * U<TY_IQ2S>::scale_half(sbB, b32, 0);
                    const float w1A = dA * U<TY_IQ2S>::scale_half(sbA, b32, 1), w1B = dB * U<TY_IQ2S>::scale_half(sbB, b32, 1);
                    for (int nt = 0; nt < 2; ++nt) {
                        if (nt >= ntiles) break;
                        const int col = nt*8 + gr;
                        const int b0 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t);
                        const int b1 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t + 16);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        mma16816(d0, d1, d2, d3, a0, a1, b0);
                        int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                        mma16816(e0, e1, e2, e3, a2, a3, b1);
                        const float da0 = Ad[kb*16 + nt*8 + 2*t], da1 = Ad[kb*16 + nt*8 + 2*t + 1];
                        acc[nt][0] = fmaf((float)d0, w0A*da0, acc[nt][0]);
                        acc[nt][1] = fmaf((float)d1, w0A*da1, acc[nt][1]);
                        acc[nt][2] = fmaf((float)d2, w0B*da0, acc[nt][2]);
                        acc[nt][3] = fmaf((float)d3, w0B*da1, acc[nt][3]);
                        acc[nt][0] = fmaf((float)e0, w1A*da0, acc[nt][0]);
                        acc[nt][1] = fmaf((float)e1, w1A*da1, acc[nt][1]);
                        acc[nt][2] = fmaf((float)e2, w1B*da0, acc[nt][2]);
                        acc[nt][3] = fmaf((float)e3, w1B*da1, acc[nt][3]);
                    }
                } else {
                    for (int nt = 0; nt < 2; ++nt) {
                        if (nt >= ntiles) break;
                        const int col = nt*8 + gr;
                        const int b0 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t);
                        const int b1 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t + 16);
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        mma16832(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                        const float da0 = Ad[kb*16 + nt*8 + 2*t], da1 = Ad[kb*16 + nt*8 + 2*t + 1];
                        acc[nt][0] = fmaf((float)d0, swA*da0, acc[nt][0]);
                        acc[nt][1] = fmaf((float)d1, swA*da1, acc[nt][1]);
                        acc[nt][2] = fmaf((float)d2, swB*da0, acc[nt][2]);
                        acc[nt][3] = fmaf((float)d3, swB*da1, acc[nt][3]);
                    }
                }
            }
        }
        for (int nt = 0; nt < 2; ++nt) {
            if (nt >= ntiles) break;
            const int c0 = nt*8 + 2*t;
            float * o0 = out + (size_t)(rt*16 + gr    ) * ostride + c0;
            float * o1 = out + (size_t)(rt*16 + gr + 8) * ostride + c0;
            if (c0 + 0 < ncols) { o0[0] = acc[nt][0]; o1[0] = acc[nt][2]; }
            if (c0 + 1 < ncols) { o0[1] = acc[nt][1]; o1[1] = acc[nt][3]; }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// (c2) FULL kernel v2: shared-memory staged raw bytes + tables, unpack into a shared int8 tile,
//      mma.m16n8k32 per 32-k block (IQ2_S: two m16n8k16, one per 16-k scale half).
//      Same math and same column-invariant structure as k_mma.
// ---------------------------------------------------------------------------------------------
template<int TY> struct V2 {
    static constexpr int  BLK   = BB<TY>::v;
    static constexpr int  SHIFT = (TY == TY_IQ4XS) ? 0 : 2;      // makes qs 4-byte aligned in shared
    static constexpr int  RAW   = (TY == TY_IQ4XS) ? 144 : 112;  // staged bytes per row per superblock
    static constexpr int  GRID  = (TY == TY_IQ3S) ? 2048 : (TY == TY_IQ3XXS) ? 1024 : (TY == TY_IQ2S) ? 8192 : 16;
    static constexpr int  MASK  = (TY == TY_IQ4XS) ? 16 : (TY == TY_IQ3XXS) ? 1024 : 2048;
    static constexpr int  TILE  = 32*36;                         // 2 blocks x 16 rows x 36 B (padded rows)
    static constexpr bool  K16  = (TY == TY_IQ2S);
};
static constexpr int V2_NWARPS = 4;

// unpack one 32-k block of one row (32 weights) into 8 x uint32 (k-major), straight into the shared tile
template<int TY>
__device__ __forceinline__ void v2_unpack(const uint8_t * sb, int blk, uint32_t * dst,
                                          const uint8_t * sh_grid, const uint2 * sh_mask) {
    if constexpr (TY == TY_IQ3S) {
        const uint32_t * g32 = (const uint32_t *) sh_grid;             // iq3s_grid, 512 x u32
        const uint32_t q0 = *(const uint32_t *)(sb +  4 + 8*blk);
        const uint32_t q1 = *(const uint32_t *)(sb +  8 + 8*blk);
        const uint32_t qhb = sb[68 + blk];
        const uint32_t sg  = *(const uint32_t *)(sb + 76 + 4*blk);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t qsb = ((i < 4) ? (q0 >> (8*i)) : (q1 >> (8*(i-4)))) & 0xFFu;
            const int idx = (int)qsb | (int)(((qhb >> i) & 1) << 8);
            const uint2 m = sh_mask[(sg >> (8*(i>>1))) & 0xFFu];
            const int mi = (int)((i & 1) ? m.y : m.x);
            dst[i] = (uint32_t)__vsub4((int)g32[idx] ^ mi, mi);
        }
    } else if constexpr (TY == TY_IQ3XXS) {
        const uint32_t * g32 = (const uint32_t *) sh_grid;             // iq3xxs_grid, 256 x u32
        const uint32_t q0 = *(const uint32_t *)(sb +  4 + 8*blk);
        const uint32_t q1 = *(const uint32_t *)(sb +  8 + 8*blk);
        const uint32_t aux = *(const uint32_t *)(sb + 68 + 4*blk);     // scales_and_signs
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t qsb = ((i < 4) ? (q0 >> (8*i)) : (q1 >> (8*(i-4)))) & 0xFFu;
            const uint2 m = sh_mask[(aux >> (7*(i>>1))) & 127u];       // ksigns table
            const int mi = (int)((i & 1) ? m.y : m.x);
            dst[i] = (uint32_t)__vsub4((int)g32[qsb] ^ mi, mi);
        }
    } else if constexpr (TY == TY_IQ2S) {
        const uint64_t * g64 = (const uint64_t *) sh_grid;             // iq2s_grid, 1024 x u64
        const uint32_t qs  = *(const uint32_t *)(sb +  4 + 4*blk);     // 4 grid bytes
        const uint32_t qhb = sb[68 + blk];
        const uint32_t sg  = *(const uint32_t *)(sb + 36 + 4*blk);     // 4 sign bytes
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int i = 2*l + e;
                const uint32_t qsb = (qs >> (8*l)) & 0xFFu;
                const int idx = (int)qsb | (int)(((qhb >> (2*l)) & 3u) << 8);
                const uint32_t gv = (uint32_t)(g64[idx] >> (32*e));
                const uint2 m = sh_mask[(sg >> (8*l)) & 0xFFu];
                const int mi = (int)(e ? m.y : m.x);
                dst[i] = (uint32_t)__vsub4((int)gv ^ mi, mi);
            }
        }
    } else {
        // IQ4_XS: qs[128] at +8, k i (low nibble) and k i+16 (high nibble) for qs byte i
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = *(const uint32_t *)(sb + 8 + 16*blk + 4*j);
            const int2 v = iq4nl_lookup(q4);
            dst[j] = (uint32_t)v.x;          // k 4j .. 4j+3
            dst[j+4] = (uint32_t)v.y;        // k 4j+16 .. 4j+19
        }
    }
}

// weight scale factor (without d) for one row / one 32-k block; half is only used for IQ2_S
template<int TY>
__device__ __forceinline__ float v2_wscale(const uint8_t * sb, int blk, int half = 0) {
    if constexpr (TY == TY_IQ3S) {
        return 1.0f + 2.0f * (float)((sb[108 + (blk >> 1)] >> (4*(blk & 1))) & 0xF);
    } else if constexpr (TY == TY_IQ3XXS) {
        return 0.5f * (0.5f + (float)((*(const uint32_t *)(sb + 68 + 4*blk) >> 28) & 0xF));
    } else if constexpr (TY == TY_IQ2S) {
        return 0.25f * (0.5f + (float)((sb[76 + blk] >> (4*half)) & 0xF));
    } else {
        const uint32_t sh = (uint32_t)(*(const uint16_t *)(sb + 2));
        const int ls = ((sb[4 + (blk >> 1)] >> (4*(blk & 1))) & 0xF) | (int)(((sh >> (2*blk)) & 3u) << 4);
        return (float)(ls - 32);
    }
}

template<int TY>
__global__ void k_mma6(const uint8_t * __restrict__ W, const int8_t * __restrict__ Aq,
                       const float * __restrict__ Ad, const uint2 * __restrict__ d_mask,
                       int R, int K, int rowbytes, int ncols, int ostride, float * __restrict__ out) {
    using V = V2<TY>;
    extern __shared__ uint8_t smem[];
    const int grid_off = 0;
    const int mask_off = V::GRID;
    const int acts_off = mask_off + V::MASK;
    const int asc_off  = acts_off + 4096;                    // 8 blocks x 16 cols x 32 int8
    const int raw_off  = asc_off + 512;                      // 8 blocks x 16 cols x fp32
    const int tile_off = raw_off + V2_NWARPS * 16 * V::RAW;
    const int total    = tile_off + V2_NWARPS * V::TILE;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane & 3, gr = lane >> 2;
    const int nb = K >> 8, rowtiles = R >> 4, ntiles = (ncols + 7) >> 3;

    // ---- one-time table copy (grid + sign masks)
    {
        if constexpr (TY == TY_IQ2S) {
            for (int i = threadIdx.x; i < V::GRID / 8; i += blockDim.x)
                ((uint64_t *)(smem + grid_off))[i] = iq2s_grid[i];
        } else if constexpr (TY == TY_IQ3S) {
            for (int i = threadIdx.x; i < V::GRID / 4; i += blockDim.x)
                ((uint32_t *)(smem + grid_off))[i] = iq3s_grid[i];
        } else if constexpr (TY == TY_IQ3XXS) {
            for (int i = threadIdx.x; i < V::GRID / 4; i += blockDim.x)
                ((uint32_t *)(smem + grid_off))[i] = iq3xxs_grid[i];
        }
        if constexpr (TY != TY_IQ4XS) {
            const int nm = V::MASK / 8;
            for (int i = threadIdx.x; i < nm; i += blockDim.x)
                ((uint2 *)(smem + mask_off))[i] = __ldg(d_mask + i);
        }
    }
    __syncthreads();

    uint8_t * sh_grid = smem + grid_off;
    const uint2 * sh_mask = (const uint2 *)(smem + mask_off);
    int8_t * sh_acts = (int8_t *)(smem + acts_off);
    float  * sh_asc  = (float  *)(smem + asc_off);
    uint8_t * sh_raw = smem + raw_off + warp * 16 * V::RAW;
    uint8_t * sh_tile = smem + tile_off + warp * V::TILE;

    for (int rt = blockIdx.x * V2_NWARPS + warp; rt < rowtiles; rt += V2_NWARPS * (int)gridDim.x) {
        const uint8_t * wA = W + (size_t)(rt*16 + gr    ) * rowbytes;
        const uint8_t * wB = W + (size_t)(rt*16 + gr + 8) * rowbytes;
        const uint8_t * base = W + (size_t)(rt*16) * rowbytes;    // 16 rows staged together
        const int NCH = 16 * V::RAW / 2;                          // u16 chunks per superblock staging
        float acc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;

        // activation staging is shared by all warps -> needs a block sync per superblock
        for (int b = 0; b < nb; ++b) {
            // stage this warp's 16 rows x RAW bytes
#pragma unroll
            for (int it = 0; it < NCH/32; ++it) {
                const int c = it*32 + lane;
                const int row = c / (V::RAW/2), off = (c % (V::RAW/2)) * 2;
                const uint16_t v = *(const uint16_t *)(base + (size_t)row * rowbytes + (size_t)b*V::BLK + off);
                *(uint16_t *)(sh_raw + row*V::RAW + off + V::SHIFT) = v;
            }
            __syncwarp();

            const float dA = __half2float(*(const __half *)(sh_raw + gr*V::RAW + V::SHIFT));
            const float dB = __half2float(*(const __half *)(sh_raw + (gr+8)*V::RAW + V::SHIFT));
            const uint8_t * sbA = sh_raw + gr*V::RAW;
            const uint8_t * sbB = sh_raw + (gr+8)*V::RAW;

#pragma unroll 1
            for (int chunk = 0; chunk < 4; ++chunk) {          // 4 chunks x 2 blocks = 8 x 32 k
                const int row = lane & 15, blk = 2*chunk + (lane >> 4);
                v2_unpack<TY>(sh_raw + row*V::RAW, blk,
                              (uint32_t *)(sh_tile + ((lane >> 4)*16 + row)*36), sh_grid, sh_mask);
                __syncwarp();
#pragma unroll
                for (int bb = 0; bb < 2; ++bb) {
                    const int b32 = 2*chunk + bb;
                    const uint32_t * tile = (const uint32_t *)(sh_tile + bb*16*36);
                    const int a0 = (int)tile[gr*9 + t];
                    const int a1 = (int)tile[(gr+8)*9 + t];
                    const int a2 = (int)tile[gr*9 + t + 4];
                    const int a3 = (int)tile[(gr+8)*9 + t + 4];
                    const int kb = b*8 + b32;
                    const int8_t * ab = Aq + (size_t)kb * (16*32);
                    const int B_CONST = 0;
                    const float * asc = Ad + (size_t)kb * 16;
                    if constexpr (V::K16) {
                        const float w0A = dA * v2_wscale<TY>(sbA, b32, 0), w0B = dB * v2_wscale<TY>(sbB, b32, 0);
                        const float w1A = dA * v2_wscale<TY>(sbA, b32, 1), w1B = dB * v2_wscale<TY>(sbB, b32, 1);
                        for (int nt = 0; nt < 2; ++nt) {
                            if (nt >= ntiles) break;
                            const int col = nt*8 + gr;
                            const int b0 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t);
                            const int b1 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t + 16);
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            mma16816(d0, d1, d2, d3, a0, a1, b0);
                            int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                            mma16816(e0, e1, e2, e3, a2, a3, b1);
                            const float da0 = asc[nt*8 + 2*t], da1 = asc[nt*8 + 2*t + 1];
                            acc[nt][0] = fmaf((float)d0, w0A*da0, acc[nt][0]);
                            acc[nt][1] = fmaf((float)d1, w0A*da1, acc[nt][1]);
                            acc[nt][2] = fmaf((float)d2, w0B*da0, acc[nt][2]);
                            acc[nt][3] = fmaf((float)d3, w0B*da1, acc[nt][3]);
                            acc[nt][0] = fmaf((float)e0, w1A*da0, acc[nt][0]);
                            acc[nt][1] = fmaf((float)e1, w1A*da1, acc[nt][1]);
                            acc[nt][2] = fmaf((float)e2, w1B*da0, acc[nt][2]);
                            acc[nt][3] = fmaf((float)e3, w1B*da1, acc[nt][3]);
                        }
                    } else {
                        const float swA = dA * v2_wscale<TY>(sbA, b32);
                        const float swB = dB * v2_wscale<TY>(sbB, b32);
                        for (int nt = 0; nt < 2; ++nt) {
                            if (nt >= ntiles) break;
                            const int col = nt*8 + gr;
                            const int b0 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t);
                            const int b1 = B_CONST ? B_CONST : *(const int *)(ab + col*32 + 4*t + 16);
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            mma16832(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                            const float da0 = asc[nt*8 + 2*t], da1 = asc[nt*8 + 2*t + 1];
                            acc[nt][0] = fmaf((float)d0, swA*da0, acc[nt][0]);
                            acc[nt][1] = fmaf((float)d1, swA*da1, acc[nt][1]);
                            acc[nt][2] = fmaf((float)d2, swB*da0, acc[nt][2]);
                            acc[nt][3] = fmaf((float)d3, swB*da1, acc[nt][3]);
                        }
                    }
                    (void)kb;
                }
                __syncwarp();
            }
        }
        for (int nt = 0; nt < 2; ++nt) {
            if (nt >= ntiles) break;
            const int c0 = nt*8 + 2*t;
            float * o0 = out + (size_t)(rt*16 + gr    ) * ostride + c0;
            float * o1 = out + (size_t)(rt*16 + gr + 8) * ostride + c0;
            if (c0 + 0 < ncols) { o0[0] = acc[nt][0]; o1[0] = acc[nt][2]; }
            if (c0 + 1 < ncols) { o0[1] = acc[nt][1]; o1[1] = acc[nt][3]; }
        }
    }
    (void)total;
}



// ---------------------------------------------------------------------------------------------
// (b) UNPACK-only kernel: same loads, same unpack, no mma (cheap int reduction keeps it alive)
// AB: 0 full, 1 no grid lookup, 2 no mask lookup, 3 metadata only, 4 no metadata (hoisted, meaningless)
// ---------------------------------------------------------------------------------------------
template<int TY, int AB>
__global__ void k_unpack(const uint8_t * __restrict__ W, const uint2 * __restrict__ mt,
                         int R, int K, int rowbytes, int * __restrict__ sink) {
    constexpr int BLK = BB<TY>::v;
    constexpr int NWARPS = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int t = lane & 3, gr = lane >> 2;
    const int nb = K >> 8, rowtiles = R >> 4;
    for (int rt = blockIdx.x * NWARPS + warp; rt < rowtiles; rt += NWARPS * (int)gridDim.x) {
        const uint8_t * wA = W + (size_t)(rt*16 + gr    ) * rowbytes;
        const uint8_t * wB = W + (size_t)(rt*16 + gr + 8) * rowbytes;
        int s = 0;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * sbA = wA + (size_t)b * BLK;
            const uint8_t * sbB = wB + (size_t)b * BLK;
#pragma unroll
            for (int b32 = 0; b32 < 8; ++b32) {
                int a0, a1, a2, a3;
                if constexpr (AB == 0) {
                    unpack_A<TY>(sbA, sbB, b32, t, mt, a0, a1, a2, a3);
                } else {
                    const uint8_t * qs = sbA + 2 + 8*b32;
                    const int i0 = t, i1 = t + 4;
                    const int qhb = sbA[66 + b32], qhbB = sbB[66 + b32];
                    const int sg = sbA[74 + 4*b32 + (i0 >> 1)];
                    const uint8_t * qsB = sbB + 2 + 8*b32;
                    int g0, g1, g2, g3;
                    if constexpr (AB == 1 || AB == 3) {
                        g0 = qs[i0] | ((qhb >> i0) & 1);
                        g1 = qs[i1] | ((qhb >> i1) & 1);
                        g2 = qsB[i0] | ((qhbB >> i0) & 1);
                        g3 = qsB[i1] | ((qhbB >> i1) & 1);
                    } else {
                        g0 = (int)__ldg(iq3s_grid + (qs[i0]  | (((qhb  >> i0) & 1) << 8)));
                        g1 = (int)__ldg(iq3s_grid + (qs[i1]  | (((qhb  >> i1) & 1) << 8)));
                        g2 = (int)__ldg(iq3s_grid + (qsB[i0] | (((qhbB >> i0) & 1) << 8)));
                        g3 = (int)__ldg(iq3s_grid + (qsB[i1] | (((qhbB >> i1) & 1) << 8)));
                    }
                    if constexpr (AB == 2 || AB == 3) {
                        const int mi = sg * 0x01010101;
                        a0 = (int)__vsub4(g0 ^ mi, mi); a1 = (int)__vsub4(g2 ^ mi, mi);
                        a2 = (int)__vsub4(g1 ^ mi, mi); a3 = (int)__vsub4(g3 ^ mi, mi);
                    } else {
                        const uint2 m = __ldg(mt + sg);
                        const int mi = (i0 & 1) ? (int)m.y : (int)m.x;
                        a0 = (int)__vsub4(g0 ^ mi, mi); a1 = (int)__vsub4(g2 ^ mi, mi);
                        a2 = (int)__vsub4(g1 ^ mi, mi); a3 = (int)__vsub4(g3 ^ mi, mi);
                    }
                    if constexpr (AB == 4) {
                        a0 = (int)__ldg(iq3s_grid + ((lane * 7 + b32) & 511));
                        a1 = (int)__ldg(mt + ((lane * 13 + b32) & 255)).x;
                        a2 = a0 ^ 0x01010101; a3 = a1 ^ 0x01010101;
                    }
                }
                s += a0 + a1 + a2 + a3;
            }
        }
        if (s == 0x7f3d5a1b) sink[0] = s;
    }
}

// ---------------------------------------------------------------------------------------------
// (c3) FULL kernel v5: staged 16 rows in shared with an odd-word row stride (conflict-free banks),
//      +2 byte shift so qs/signs are 4-byte aligned, u32 metadata loads shared by the quad.
//      Same math and k order as k_mma (bit-identical output).
// ---------------------------------------------------------------------------------------------
template<int TY> struct V5S {
    static constexpr int BLK = BB<TY>::v;
    static constexpr int RAW = 116;      // 110 bytes + 2 shift + 4 pad; 29 words per row -> odd stride
    static constexpr int SH  = 2;        // shift so that qs is 4-byte aligned in shared
};

// weight scale factor for the +2-shifted shared layout
template<int TY>
__device__ __forceinline__ float v5_wscale(const uint8_t * sb, int blk, int half = 0) {
    if constexpr (TY == TY_IQ3S) {
        return 1.0f + 2.0f * (float)((sb[108 + (blk >> 1)] >> (4*(blk & 1))) & 0xF);
    } else if constexpr (TY == TY_IQ3XXS) {
        return 0.5f * (0.5f + (float)((*(const uint32_t *)(sb + 68 + 4*blk) >> 28) & 0xF));
    } else if constexpr (TY == TY_IQ2S) {
        return 0.25f * (0.5f + (float)((sb[76 + blk] >> (4*half)) & 0xF));
    } else {
        return 0.0f;
    }
}

// one row, one 32-k block -> the two A-fragment groups of this row (a0 = k 4t..4t+3, a2 = 4t+16..+19)
template<int TY>
__device__ __forceinline__ void v5_unpack(const uint8_t * sb, int blk, int t,
                                          const uint32_t * g32, const uint2 * maskt,
                                          int & a0, int & a2) {
    if constexpr (TY == TY_IQ3S) {
        const uint32_t qw0 = *(const uint32_t *)(sb +  4 + 8*blk);   // qs[0..3]
        const uint32_t qw1 = *(const uint32_t *)(sb +  8 + 8*blk);   // qs[4..7]
        const int qhb = sb[68 + blk];
        const uint32_t sg = *(const uint32_t *)(sb + 76 + 4*blk);    // 4 sign bytes
        const int qA = (int)((qw0 >> (8*t)) & 0xFF);
        const int qB = (int)((qw1 >> (8*t)) & 0xFF);
        const int hbA = (qhb >> t) & 1, hbB = (qhb >> (t+4)) & 1;
        const uint2 mA = maskt[(sg >> (8*(t>>1))) & 0xFF];
        const uint2 mB = maskt[(sg >> (8*((t>>1)+2))) & 0xFF];
        const int miA = (int)((t & 1) ? mA.y : mA.x);
        const int miB = (int)((t & 1) ? mB.y : mB.x);
        a0 = (int)__vsub4((int)g32[qA | (hbA << 8)] ^ miA, miA);
        a2 = (int)__vsub4((int)g32[qB | (hbB << 8)] ^ miB, miB);
    } else if constexpr (TY == TY_IQ3XXS) {
        const uint32_t qw0 = *(const uint32_t *)(sb +  4 + 8*blk);
        const uint32_t qw1 = *(const uint32_t *)(sb +  8 + 8*blk);
        const uint32_t aux = *(const uint32_t *)(sb + 68 + 4*blk);
        const int qA = (int)((qw0 >> (8*t)) & 0xFF);
        const int qB = (int)((qw1 >> (8*t)) & 0xFF);
        const uint2 mA = maskt[(aux >> (7*(t>>1))) & 127u];
        const uint2 mB = maskt[(aux >> (7*((t>>1)+2))) & 127u];
        const int miA = (int)((t & 1) ? mA.y : mA.x);
        const int miB = (int)((t & 1) ? mB.y : mB.x);
        a0 = (int)__vsub4((int)g32[qA] ^ miA, miA);
        a2 = (int)__vsub4((int)g32[qB] ^ miB, miB);
    } else if constexpr (TY == TY_IQ2S) {
        const uint32_t qs  = *(const uint32_t *)(sb + 4 + 4*blk);    // 4 grid bytes
        const uint32_t sg  = *(const uint32_t *)(sb + 36 + 4*blk);   // 4 sign bytes
        const int qhb = sb[68 + blk];
        const int lA = t >> 1, lB = lA + 2, e = t & 1;
        const int idxA = (int)((qs >> (8*lA)) & 0xFF) | (int)(((qhb >> (2*lA)) & 3) << 8);
        const int idxB = (int)((qs >> (8*lB)) & 0xFF) | (int)(((qhb >> (2*lB)) & 3) << 8);
        const uint2 mA = maskt[(sg >> (8*lA)) & 0xFF];
        const uint2 mB = maskt[(sg >> (8*lB)) & 0xFF];
        const int miA = (int)(e ? mA.y : mA.x);
        const int miB = (int)(e ? mB.y : mB.x);
        a0 = (int)__vsub4((int)(uint32_t)(((const uint64_t *)g32)[idxA] >> (32*e)) ^ miA, miA);
        a2 = (int)__vsub4((int)(uint32_t)(((const uint64_t *)g32)[idxB] >> (32*e)) ^ miB, miB);
    } else {
        a0 = 0; a2 = 0;
    }
}

template<int TY>
__global__ void k_mma5(const uint8_t * __restrict__ W, const int8_t * __restrict__ Aq,
                       const float * __restrict__ Ad, const uint2 * __restrict__ d_mask,
                       int R, int K, int rowbytes, int ncols, int ostride, float * __restrict__ out) {
    extern __shared__ uint8_t smem5[];
    const int grid_bytes = (TY == TY_IQ3S) ? 2048 : (TY == TY_IQ3XXS) ? 1024 : (TY == TY_IQ2S) ? 8192 : 0;
    const int mask_bytes = (TY == TY_IQ3XXS) ? 1024 : 2048;
    uint8_t * sh_grid = smem5;
    const uint2 * sh_mask = (const uint2 *)(smem5 + grid_bytes);
    const int raw_off = grid_bytes + mask_bytes;
    const int NWARPS = 4;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int t = lane & 3, gr = lane >> 2;
    const int nb = K >> 8, rowtiles = R >> 4, ntiles = (ncols + 7) >> 3;
    uint8_t * sh_raw = smem5 + raw_off + warp * 16 * V5S<TY>::RAW;

    {
        if constexpr (TY == TY_IQ3S) {
            for (int i = threadIdx.x; i < grid_bytes/4; i += blockDim.x) ((uint32_t *)sh_grid)[i] = iq3s_grid[i];
        } else if constexpr (TY == TY_IQ3XXS) {
            for (int i = threadIdx.x; i < grid_bytes/4; i += blockDim.x) ((uint32_t *)sh_grid)[i] = iq3xxs_grid[i];
        } else if constexpr (TY == TY_IQ2S) {
            for (int i = threadIdx.x; i < grid_bytes/8; i += blockDim.x) ((uint64_t *)sh_grid)[i] = iq2s_grid[i];
        }
        for (int i = threadIdx.x; i < mask_bytes/8; i += blockDim.x)
            ((uint2 *)(smem5 + grid_bytes))[i] = __ldg(d_mask + i);
    }
    __syncthreads();

    for (int rt = blockIdx.x * NWARPS + warp; rt < rowtiles; rt += NWARPS * (int)gridDim.x) {
        const uint8_t * base = W + (size_t)(rt*16) * rowbytes;
        float acc[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        for (int b = 0; b < nb; ++b) {
#pragma unroll
            for (int it = 0; it < 16*V5S<TY>::RAW/2/32; ++it) {
                const int c = it*32 + lane;
                const int row = c / (V5S<TY>::RAW/2), off = (c % (V5S<TY>::RAW/2)) * 2;
                if (off + V5S<TY>::SH + 1 < V5S<TY>::RAW)
                    *(uint16_t *)(sh_raw + row*V5S<TY>::RAW + off + V5S<TY>::SH) =
                        *(const uint16_t *)(base + (size_t)row * rowbytes + (size_t)b*V5S<TY>::BLK + off);
            }
            __syncwarp();
            const uint8_t * sbA = sh_raw + gr*V5S<TY>::RAW;
            const uint8_t * sbB = sh_raw + (gr+8)*V5S<TY>::RAW;
            const float dA = __half2float(*(const __half *)(sbA + V5S<TY>::SH)), dB = __half2float(*(const __half *)(sbB + V5S<TY>::SH));
            const uint32_t * g32 = (const uint32_t *)sh_grid;
#pragma unroll
            for (int b32 = 0; b32 < 8; ++b32) {
                int a0, a2, c0, c2;
                v5_unpack<TY>(sbA, b32, t, g32, sh_mask, a0, a2);
                v5_unpack<TY>(sbB, b32, t, g32, sh_mask, c0, c2);
                const int a1 = c0, a3 = c2;
                const float swA = dA * v5_wscale<TY>(sbA, b32);
                const float swB = dB * v5_wscale<TY>(sbB, b32);
                const int kb = b*8 + b32;
                const int8_t * ab = Aq + (size_t)kb * (16*32);
                for (int nt = 0; nt < 2; ++nt) {
                    if (nt >= ntiles) break;
                    const int col = nt*8 + gr;
                    const int b0 = *(const int *)(ab + col*32 + 4*t);
                    const int b1 = *(const int *)(ab + col*32 + 4*t + 16);
                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma16832(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                    const float da0 = Ad[kb*16 + nt*8 + 2*t], da1 = Ad[kb*16 + nt*8 + 2*t + 1];
                    acc[nt][0] = fmaf((float)d0, swA*da0, acc[nt][0]);
                    acc[nt][1] = fmaf((float)d1, swA*da1, acc[nt][1]);
                    acc[nt][2] = fmaf((float)d2, swB*da0, acc[nt][2]);
                    acc[nt][3] = fmaf((float)d3, swB*da1, acc[nt][3]);
                }
            }
            __syncwarp();
        }
        for (int nt = 0; nt < 2; ++nt) {
            if (nt >= ntiles) break;
            const int c0 = nt*8 + 2*t;
            float * o0 = out + (size_t)(rt*16 + gr    ) * ostride + c0;
            float * o1 = out + (size_t)(rt*16 + gr + 8) * ostride + c0;
            if (c0 + 0 < ncols) { o0[0] = acc[nt][0]; o1[0] = acc[nt][2]; }
            if (c0 + 1 < ncols) { o0[1] = acc[nt][1]; o1[1] = acc[nt][3]; }
        }
    }
}

template<int TY>
static void launch_v5(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * mt,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    const int gb = (TY == TY_IQ3S) ? 2048 : (TY == TY_IQ3XXS) ? 1024 : (TY == TY_IQ2S) ? 8192 : 0;
    const int mb = (TY == TY_IQ3XXS) ? 1024 : 2048;
    const int shm = gb + mb + 4 * 16 * V5S<TY>::RAW;
    cudaFuncSetAttribute((const void *)k_mma5<TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
    k_mma5<TY><<<gdim, 128, shm, st>>>(W, Aq, Ad, mt, R, K, rb, ncols, 16, out);
}

// ---------------------------------------------------------------------------------------------
// (c4) FULL kernel v7: like v1 (direct register unpack, global reads) but one warp owns 32 rows
//      (two 16-row mma tiles). Doubles the independent dependency chains per warp and shares the
//      activation B fragments between the two tiles. Same math and k order as k_mma.
// ---------------------------------------------------------------------------------------------
template<int TY>
__global__ void k_mma7(const uint8_t * __restrict__ W, const int8_t * __restrict__ Aq,
                       const float * __restrict__ Ad, const uint2 * __restrict__ mt,
                       int R, int K, int rowbytes, int ncols, int ostride, float * __restrict__ out) {
    constexpr int BLK = BB<TY>::v;
    constexpr int NWARPS = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int t = lane & 3, gr = lane >> 2;
    const int nb = K >> 8, rowtiles = R >> 5, ntiles = (ncols + 7) >> 3;   // 32 rows per warp
    for (int rt = blockIdx.x * NWARPS + warp; rt < rowtiles; rt += NWARPS * (int)gridDim.x) {
        const uint8_t * wA0 = W + (size_t)(rt*32 + gr    ) * rowbytes;
        const uint8_t * wB0 = W + (size_t)(rt*32 + gr + 8) * rowbytes;
        const uint8_t * wA1 = W + (size_t)(rt*32 + gr + 16) * rowbytes;
        const uint8_t * wB1 = W + (size_t)(rt*32 + gr + 24) * rowbytes;
        float acc[2][2][4];
#pragma unroll
        for (int tl = 0; tl < 2; ++tl)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[tl][nt][i] = 0.0f;
        for (int b = 0; b < nb; ++b) {
            const uint8_t * sA0 = wA0 + (size_t)b * BLK;
            const uint8_t * sB0 = wB0 + (size_t)b * BLK;
            const uint8_t * sA1 = wA1 + (size_t)b * BLK;
            const uint8_t * sB1 = wB1 + (size_t)b * BLK;
            const float dA0 = U<TY>::dval(sA0), dB0 = U<TY>::dval(sB0);
            const float dA1 = U<TY>::dval(sA1), dB1 = U<TY>::dval(sB1);
#pragma unroll
            for (int b32 = 0; b32 < 8; ++b32) {
                const int kb = b*8 + b32;
                const int8_t * ab = Aq + (size_t)kb * (16*32);
                int a0, a1, a2, a3, c0, c1, c2, c3;
                unpack_A<TY>(sA0, sB0, b32, t, mt, a0, a1, a2, a3);
                unpack_A<TY>(sA1, sB1, b32, t, mt, c0, c1, c2, c3);
                const float w0A = dA0 * U<TY>::scale(sA0, b32), w0B = dB0 * U<TY>::scale(sB0, b32);
                const float w1A = dA1 * U<TY>::scale(sA1, b32), w1B = dB1 * U<TY>::scale(sB1, b32);
                for (int nt = 0; nt < 2; ++nt) {
                    if (nt >= ntiles) break;
                    const int col = nt*8 + gr;
                    const int b0 = *(const int *)(ab + col*32 + 4*t);
                    const int b1 = *(const int *)(ab + col*32 + 4*t + 16);
                    const float da0 = Ad[kb*16 + nt*8 + 2*t], da1 = Ad[kb*16 + nt*8 + 2*t + 1];
                    int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                    mma16832(e0, e1, e2, e3, a0, a1, a2, a3, b0, b1);
                    acc[0][nt][0] = fmaf((float)e0, w0A*da0, acc[0][nt][0]);
                    acc[0][nt][1] = fmaf((float)e1, w0A*da1, acc[0][nt][1]);
                    acc[0][nt][2] = fmaf((float)e2, w0B*da0, acc[0][nt][2]);
                    acc[0][nt][3] = fmaf((float)e3, w0B*da1, acc[0][nt][3]);
                    int f0 = 0, f1 = 0, f2 = 0, f3 = 0;
                    mma16832(f0, f1, f2, f3, c0, c1, c2, c3, b0, b1);
                    acc[1][nt][0] = fmaf((float)f0, w1A*da0, acc[1][nt][0]);
                    acc[1][nt][1] = fmaf((float)f1, w1A*da1, acc[1][nt][1]);
                    acc[1][nt][2] = fmaf((float)f2, w1B*da0, acc[1][nt][2]);
                    acc[1][nt][3] = fmaf((float)f3, w1B*da1, acc[1][nt][3]);
                }
            }
        }
        for (int tl = 0; tl < 2; ++tl) {
            for (int nt = 0; nt < 2; ++nt) {
                if (nt >= ntiles) break;
                const int c0 = nt*8 + 2*t;
                const int r0 = rt*32 + tl*16 + gr;
                float * o0 = out + (size_t)(r0    ) * ostride + c0;
                float * o1 = out + (size_t)(r0 + 8) * ostride + c0;
                if (c0 + 0 < ncols) { o0[0] = acc[tl][nt][0]; o1[0] = acc[tl][nt][2]; }
                if (c0 + 1 < ncols) { o0[1] = acc[tl][nt][1]; o1[1] = acc[tl][nt][3]; }
            }
        }
    }
}

template<int TY>
static void launch_v7(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * mt,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    k_mma7<TY><<<gdim, 128, 0, st>>>(W, Aq, Ad, mt, R, K, rb, ncols, 16, out);
}

template<int TY>
static void launch_v6(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * mt,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    using V = V2<TY>;
    const int shm = V::GRID + V::MASK + 4096 + 512 + 4 * (16 * V::RAW + V::TILE);
    cudaFuncSetAttribute((const void *)k_mma6<TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
    k_mma6<TY><<<gdim, 128, shm, st>>>(W, Aq, Ad, mt, R, K, rb, ncols, 16, out);
}

// ---------------------------------------------------------------------------------------------
template<int TY>
static void launch_v3(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * mt,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    constexpr int RAW = (TY == TY_IQ4XS) ? 144 : 112;
    k_mma<TY,1><<<gdim, 128, 4 * 16 * RAW, st>>>(W, Aq, Ad, mt, R, K, rb, ncols, 16, out);
}

template<int TY>
static void launch_v4(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * mt,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    constexpr int RAW = (TY == TY_IQ4XS) ? 144 : 112;
    k_mma<TY,2><<<gdim, 128, 4 * 16 * RAW, st>>>(W, Aq, Ad, mt, R, K, rb, ncols, 16, out);
}

template<int TY>
static double time_unpack_ab(int ab, const uint8_t * d_W, const uint2 * mt, int out, int in, int rb,
                             int * d_sink, int gdim, int reps) {
    switch (ab) {
        case 0: return timeit([&]{ k_unpack<TY,0><<<gdim,128>>>(d_W, mt, out, in, rb, d_sink); }, reps);
        case 1: return timeit([&]{ k_unpack<TY,1><<<gdim,128>>>(d_W, mt, out, in, rb, d_sink); }, reps);
        case 2: return timeit([&]{ k_unpack<TY,2><<<gdim,128>>>(d_W, mt, out, in, rb, d_sink); }, reps);
        case 3: return timeit([&]{ k_unpack<TY,3><<<gdim,128>>>(d_W, mt, out, in, rb, d_sink); }, reps);
        default:return timeit([&]{ k_unpack<TY,4><<<gdim,128>>>(d_W, mt, out, in, rb, d_sink); }, reps);
    }
}

// ---------------------------------------------------------------------------------------------
// (a) LOAD-only kernels
//   k_stream   : contiguous, 16 B per lane over the whole tensor (bandwidth ceiling)
//   k_tileload : the planned per-warp row-tile pattern, coalesced u32, no unpack
// ---------------------------------------------------------------------------------------------
__global__ void k_stream(const uint4 * __restrict__ p, size_t n, uint32_t * __restrict__ sink) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 a = make_uint4(0,0,0,0);
    for (; i < n; i += stride) {
        uint4 v = __ldg(p + i);
        a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w;
    }
    if ((a.x ^ a.y ^ a.z ^ a.w) == 0x7f3d5a1b) sink[0] = a.x;
}

__global__ void k_tileload(const uint8_t * __restrict__ W, int R, int K, int rowbytes,
                           uint32_t * __restrict__ sink) {
    constexpr int NWARPS = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nb = K >> 8, rowtiles = R >> 4;
    uint32_t s = 0;
    for (int rt = blockIdx.x * NWARPS + warp; rt < rowtiles; rt += NWARPS * (int)gridDim.x) {
        for (int b = 0; b < nb; ++b) {
            // 16 rows x 112 B (110 B rounded up) = 448 u32 chunks, 32 lanes -> 14 iterations
            const uint8_t * base = W + (size_t)(rt*16) * rowbytes + (size_t)b * 112;
#pragma unroll
            for (int it = 0; it < 14; ++it) {
                const int c   = it*32 + lane;
                const int row = c / 28;
                const int off = (c % 28) * 4;
                s += *(const uint32_t *)(base + (size_t)row * rowbytes + off);
            }
        }
    }
    if (s == 0x7f3d5a1bU) sink[0] = s;
}

// ---------------------------------------------------------------------------------------------
// data generation
// ---------------------------------------------------------------------------------------------
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint64_t rnd64() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
static uint8_t  rnd8()  { return (uint8_t)(rnd64() >> 27); }
static float    rndf()  { return (float)((rnd64() >> 11) * (1.0 / 9007199254740992.0)); }   // [0,1)

// produce `nrows` template rows of packed weights for (type, K)
static std::vector<uint8_t> make_rows(int ty, int K, int nrows) {
    const int rb = (K/256) * blk_bytes_h(ty);
    std::vector<uint8_t> out((size_t)rb * nrows);
    for (int r = 0; r < nrows; ++r) {
        uint8_t * p = out.data() + (size_t)r * rb;
        for (int b = 0; b < K/256; ++b) {
            uint8_t * q = p + b * blk_bytes_h(ty);
            const float d = 0.0015f + 0.004f * rndf();
            const __half dh = __float2half(d);
            memcpy(q, &dh, 2);
            for (int i = 2; i < blk_bytes_h(ty); ++i) q[i] = rnd8();
        }
    }
    return out;
}

__global__ void k_fill(const uint8_t * __restrict__ tmpl, int ntmpl, uint8_t * __restrict__ W,
                       size_t nrows, int rb) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t tot = nrows * rb;
    for (; i < tot; i += stride) {
        const size_t r = i / rb, o = i % rb;
        W[i] = tmpl[(r % ntmpl) * (size_t)rb + o];
    }
}

// q8_1 quantization: Aq[kb][16][32] int8, Ad[kb][16] float (per column, per 32-element block)
static void quant_acts(const std::vector<float> & x, int K, int ncols,
                       std::vector<int8_t> & Aq, std::vector<float> & Ad) {
    const int nb32 = K / 32;
    Aq.assign((size_t)nb32 * 16 * 32, 0);
    Ad.assign((size_t)nb32 * 16, 0.0f);
    for (int c = 0; c < ncols; ++c) {
        for (int b = 0; b < nb32; ++b) {
            const float * xs = x.data() + (size_t)c * K + b * 32;
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xs[j]));
            const float d = amax / 127.0f;
            const float id = d > 0 ? 1.0f / d : 0.0f;
            Ad[(size_t)b * 16 + c] = d;
            for (int j = 0; j < 32; ++j) {
                int v = (int)roundf(xs[j] * id);
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                Aq[((size_t)b * 16 + c) * 32 + j] = (int8_t)v;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
template<typename F>
static double timeit(F launch, int reps) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    launch();                                     // warm-up
    cudaError_t le = cudaDeviceSynchronize();
    if (le != cudaSuccess) { printf("LAUNCH/SYNC ERROR: %s", cudaGetErrorString(le)); printf("\n"); exit(1); }
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(e0);
        launch();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return t[t.size() / 2];
}

// ---------------------------------------------------------------------------------------------
static void check_cuda(const char * what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error in %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------------------------------------
// correctness of the FULL kernel vs ggml dequant + scalar dot
// ---------------------------------------------------------------------------------------------
static double check_full(int ty, cudaStream_t st, const uint2 * d_mt, int * fail) {
    const int R = 32, K = 512, ncols = 8;
    const int rb = (K/256) * blk_bytes_h(ty);
    std::vector<uint8_t> w = make_rows(ty, K, R);
    std::vector<float> x((size_t)K * 16);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (rndf() - 0.5f) * 2.0f;
    std::vector<int8_t> Aq; std::vector<float> Ad;
    quant_acts(x, K, 16, Aq, Ad);

    uint8_t * d_W = nullptr; int8_t * d_Aq = nullptr; float * d_Ad = nullptr; float * d_out = nullptr;
    cudaMalloc(&d_W, w.size()); cudaMalloc(&d_Aq, Aq.size()); cudaMalloc(&d_Ad, Ad.size()*sizeof(float));
    cudaMalloc(&d_out, (size_t)R * 16 * sizeof(float));
    cudaMemcpy(d_W, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Aq, Aq.data(), Aq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ad, Ad.data(), Ad.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));

    void (*deq)(const void*, float*, int64_t) = ty==TY_IQ3S ? dequantize_row_iq3_s
                                             : ty==TY_IQ3XXS ? dequantize_row_iq3_xxs
                                             : ty==TY_IQ2S ? dequantize_row_iq2_s : dequantize_row_iq4_xs;
    std::vector<double> ref((size_t)R * 8, 0.0);
    std::vector<float> wrow(K);
    for (int c = 0; c < 8; ++c) {
        for (int r = 0; r < R; ++r) {
            deq(w.data() + (size_t)r * rb, wrow.data(), K);
            double s = 0;
            for (int b = 0; b < K/32; ++b) {
                const int8_t * q = Aq.data() + ((size_t)b*16 + c)*32;
                const double da = Ad[(size_t)b*16 + c];
                for (int j = 0; j < 32; ++j) s += (double)wrow[b*32 + j] * ((double)q[j] * da);
            }
            ref[(size_t)r*8 + c] = s;
        }
    }

    if (ty == TY_IQ2S) { k_mma<TY_IQ2S,0><<<2, 128, 0, st>>>(d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, 16, d_out); }
    else if (ty == TY_IQ3S)  { k_mma<TY_IQ3S,0><<<2, 128, 0, st>>>(d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, 16, d_out); }
    else if (ty == TY_IQ3XXS){ k_mma<TY_IQ3XXS,0><<<2, 128, 0, st>>>(d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, 16, d_out); }
    else                     { k_mma<TY_IQ4XS,0><<<2, 128, 0, st>>>(d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, 16, d_out); }
    check_cuda("k_mma");
    std::vector<float> got((size_t)R*16);
    cudaMemcpy(got.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);

    // v3 (staged) must agree bit-exactly with v1: same math, same k order
    std::vector<float> got3((size_t)R*16);
    cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));
    if      (ty == TY_IQ2S)   launch_v3<TY_IQ2S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
    else if (ty == TY_IQ3S)   launch_v3<TY_IQ3S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
    else if (ty == TY_IQ3XXS) launch_v3<TY_IQ3XXS> (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
    else                      launch_v3<TY_IQ4XS>  (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
    check_cuda("k_mma3");
    cudaMemcpy(got3.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);
    int v1v3 = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < 8; ++c)
        if (memcmp(&got[(size_t)r*16+c], &got3[(size_t)r*16+c], 4) != 0) ++v1v3;
    printf("#   v1 vs v3 bit diffs: %d\n", v1v3);

    if (getenv("MMSB_DEBUG")) {
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c)
            printf("#    r%d c%d got %12.5f  ref %12.5f\n", r, c, got[(size_t)r*16+c], ref[(size_t)r*8+c]);
    }
    {
        std::vector<float> got7((size_t)R*16);
        cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));
        if      (ty == TY_IQ2S)   launch_v7<TY_IQ2S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 1, st);
        else if (ty == TY_IQ3S)   launch_v7<TY_IQ3S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 1, st);
        else if (ty == TY_IQ3XXS) launch_v7<TY_IQ3XXS> (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 1, st);
        else                      launch_v7<TY_IQ4XS>  (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 1, st);
        check_cuda("k_mma7");
        cudaMemcpy(got7.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);
        int v1v7 = 0;
        for (int r = 0; r < R; ++r) for (int c = 0; c < 8; ++c)
            if (memcmp(&got[(size_t)r*16+c], &got7[(size_t)r*16+c], 4) != 0) ++v1v7;
        printf("#   v1 vs v7 bit diffs: %d", v1v7); printf("\n");
    }
    {
        std::vector<float> got6((size_t)R*16);
        cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));
        if      (ty == TY_IQ2S)   launch_v6<TY_IQ2S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        else if (ty == TY_IQ3S)   launch_v6<TY_IQ3S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        else if (ty == TY_IQ3XXS) launch_v6<TY_IQ3XXS> (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        else                      launch_v6<TY_IQ4XS>  (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        check_cuda("k_mma6");
        cudaMemcpy(got6.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);
        int v1v6 = 0;
        for (int r = 0; r < R; ++r) for (int c = 0; c < 8; ++c)
            if (memcmp(&got[(size_t)r*16+c], &got6[(size_t)r*16+c], 4) != 0) ++v1v6;
        printf("#   v1 vs v6 bit diffs: %d\n", v1v6);
    }
    if (ty == TY_IQ3S || ty == TY_IQ3XXS) {
        std::vector<float> got5((size_t)R*16);
        cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));
        if      (ty == TY_IQ2S)   launch_v5<TY_IQ2S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        else if (ty == TY_IQ3S)   launch_v5<TY_IQ3S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        else                      launch_v5<TY_IQ3XXS> (d_W, d_Aq, d_Ad, d_mt, R, K, rb, ncols, d_out, 2, st);
        check_cuda("k_mma5");
        cudaMemcpy(got5.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);
        int v1v5 = 0;
        for (int r = 0; r < R; ++r) for (int c = 0; c < 8; ++c)
            if (memcmp(&got[(size_t)r*16+c], &got5[(size_t)r*16+c], 4) != 0) ++v1v5;
        printf("#   v1 vs v5 bit diffs: %d\n", v1v5);
    }
    double maxrel = 0, maxabs = 0, scale = 0;
    for (int r = 0; r < R; ++r) for (int c = 0; c < 8; ++c) {
        const double g = got[(size_t)r*16 + c], rr = ref[(size_t)r*8 + c];
        maxabs = fmax(maxabs, fabs(g - rr));
        scale  = fmax(scale, fabs(rr));
    }
    maxrel = scale > 0 ? maxabs / scale : maxabs;
    if (maxrel > 1e-5) *fail = 1;
    cudaFree(d_W); cudaFree(d_Aq); cudaFree(d_Ad); cudaFree(d_out);
    return maxrel;
}

// column invariance: outputs of column j must be bit-identical for ncols 1..16
static int check_cols(int ty, cudaStream_t st, const uint2 * d_mt) {
    const int R = 64, K = 512;
    const int rb = (K/256) * blk_bytes_h(ty);
    std::vector<uint8_t> w = make_rows(ty, K, R);
    std::vector<float> x((size_t)K * 16);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (rndf() - 0.5f) * 2.0f;
    std::vector<int8_t> Aq; std::vector<float> Ad;
    quant_acts(x, K, 16, Aq, Ad);
    uint8_t * d_W = nullptr; int8_t * d_Aq = nullptr; float * d_Ad = nullptr; float * d_out = nullptr;
    cudaMalloc(&d_W, w.size()); cudaMalloc(&d_Aq, Aq.size()); cudaMalloc(&d_Ad, Ad.size()*sizeof(float));
    cudaMalloc(&d_out, (size_t)R * 16 * sizeof(float));
    cudaMemcpy(d_W, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Aq, Aq.data(), Aq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ad, Ad.data(), Ad.size()*sizeof(float), cudaMemcpyHostToDevice);

    std::vector<std::vector<float>> outs;
    for (int nc : {1,2,4,8,12,16}) {
        cudaMemset(d_out, 0, (size_t)R*16*sizeof(float));
        if      (ty == TY_IQ2S)   launch_v3<TY_IQ2S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, nc, d_out, 4, st);
        else if (ty == TY_IQ3S)   launch_v3<TY_IQ3S>   (d_W, d_Aq, d_Ad, d_mt, R, K, rb, nc, d_out, 4, st);
        else if (ty == TY_IQ3XXS) launch_v3<TY_IQ3XXS> (d_W, d_Aq, d_Ad, d_mt, R, K, rb, nc, d_out, 4, st);
        else                      launch_v3<TY_IQ4XS>  (d_W, d_Aq, d_Ad, d_mt, R, K, rb, nc, d_out, 4, st);
        check_cuda("k_mma cols");
        std::vector<float> o((size_t)R*16);
        cudaMemcpy(o.data(), d_out, (size_t)R*16*sizeof(float), cudaMemcpyDeviceToHost);
        outs.push_back(o);
    }
    const int nc_all[6] = {1,2,4,8,12,16};
    // for each column compare every run that actually wrote it (nc > c) against the smallest such run
    int diffs = 0;
    for (int c = 0; c < 16; ++c) {
        int base = -1;
        for (size_t v = 0; v < outs.size(); ++v) if (nc_all[v] > c) { base = (int)v; break; }
        if (base < 0) continue;
        for (size_t v = 0; v < outs.size(); ++v) {
            if (nc_all[v] <= c) continue;
            for (int r = 0; r < R; ++r) {
                if (memcmp(&outs[v][(size_t)r*16 + c], &outs[base][(size_t)r*16 + c], 4) != 0) ++diffs;
            }
        }
    }
    cudaFree(d_W); cudaFree(d_Aq); cudaFree(d_Ad); cudaFree(d_out);
    return diffs;
}

// ---------------------------------------------------------------------------------------------
int main(int argc, char ** argv) {
    int types_arg = -1;
    std::vector<std::pair<int,int>> shapes = {
        {5120, 17408}, {17408, 5120}, {5120, 10240}, {6144, 5120}, {5120, 248320}
    };
    bool do_check = true;
    int reps = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--reps") && i+1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shapes")) shapes.clear();
        else if (!strcmp(argv[i], "--one") && i+1 < argc) { int a, b; if (sscanf(argv[++i], "%d,%d", &a, &b) == 2) shapes.push_back({a,b}); }
        else if (!strcmp(argv[i], "--no-check")) do_check = false;
        else if (!strcmp(argv[i], "--types") && i+1 < argc) types_arg = atoi(argv[++i]);
    }
    if (shapes.empty()) shapes.push_back({5120, 17408});
    (void)types_arg;

    setvbuf(stdout, nullptr, _IONBF, 0);
    cudaSetDevice(0);
    cudaStream_t st = 0;

    // sign mask tables
    uint2 h8[256]; uint2 hx[128];
    for (int i = 0; i < 256; ++i) { h8[i].x = nib_mask(i & 0xF); h8[i].y = nib_mask((i >> 4) & 0xF); }
    for (int i = 0; i < 128; ++i) { const int b = ksigns_iq2xs_h[i]; hx[i].x = nib_mask(b & 0xF); hx[i].y = nib_mask((b >> 4) & 0xF); }
    cudaMemcpyToSymbol(g_mask8, h8, sizeof(h8));
    cudaMemcpyToSymbol(g_xxs_mask, hx, sizeof(hx));
    uint2 * d_mask8 = nullptr; uint2 * d_xxs = nullptr;
    cudaMalloc(&d_mask8, sizeof(h8)); cudaMalloc(&d_xxs, sizeof(hx));
    cudaMemcpy(d_mask8, h8, sizeof(h8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_xxs,  hx, sizeof(hx), cudaMemcpyHostToDevice);

    int device = 0; cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    printf("# %s, %d SMs, sm_%d%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // ---- correctness first
    if (do_check) {
        printf("# correctness (R=32, K=512, ncols=8): max |err| / max |ref|\n");
        for (int ty = 0; ty < 4; ++ty) {
            int fail = 0;
            const uint2 * mt = (ty == TY_IQ3XXS) ? d_xxs : d_mask8;
            const double e = check_full(ty, st, mt, &fail);
            const int diffs = check_cols(ty, st, mt);
            printf("#   %-8s rel_err %.3e %s   column-invariance: %d bit diffs over ncols 1,2,4,8,12,16\n",
                   TYNAME[ty], e, fail ? "FAIL" : "ok", diffs);
            if (fail) { printf("# ABORT: kernel does not match the reference\n"); return 1; }
        }
    }

    const int NC[] = {1, 2, 4, 8, 12, 16};
    printf("type,shape,ncols,stream_gbs,load_us,load_x,unpack_us,unpack_x,full_us,full_x,att_pct,v2_us,v2_x,v2_att\n");

    for (int ty = 0; ty < 4; ++ty) {
        const int bbty = blk_bytes_h(ty);
        const uint2 * mt = (ty == TY_IQ3XXS) ? d_xxs : d_mask8;
        for (auto & sh : shapes) {
            const int in = sh.first, out = sh.second;
            const int rb = (in/256) * bbty;
            const size_t nbytes = (size_t)rb * out;
            // weights
            std::vector<uint8_t> tmpl = make_rows(ty, in, 256);
            uint8_t * d_W = nullptr;
            cudaMalloc(&d_W, nbytes + 16);
            uint8_t * d_tmpl = nullptr; cudaMalloc(&d_tmpl, tmpl.size());
            cudaMemcpy(d_tmpl, tmpl.data(), tmpl.size(), cudaMemcpyHostToDevice);
            k_fill<<<256, 256>>>(d_tmpl, 256, d_W, (size_t)out, rb);
            check_cuda("k_fill");
            cudaFree(d_tmpl);
            // activations for the 16-column case
            std::vector<float> x((size_t)in * 16);
            for (size_t i = 0; i < x.size(); ++i) x[i] = (rndf() - 0.5f) * 2.0f;
            std::vector<int8_t> Aq; std::vector<float> Ad;
            quant_acts(x, in, 16, Aq, Ad);
            int8_t * d_Aq = nullptr; float * d_Ad = nullptr; float * d_out = nullptr; int * d_sink = nullptr;
            cudaMalloc(&d_Aq, Aq.size()); cudaMalloc(&d_Ad, Ad.size()*sizeof(float));
            cudaMalloc(&d_out, (size_t)out * 16 * sizeof(float)); cudaMalloc(&d_sink, 4);
            cudaMemcpy(d_Aq, Aq.data(), Aq.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Ad, Ad.data(), Ad.size()*sizeof(float), cudaMemcpyHostToDevice);

            const int sms = prop.multiProcessorCount;
            const int gdim = sms * 8;

            // STREAM
            const double t_stream = timeit([&]{ k_stream<<<gdim*4, 128>>>( (const uint4*)d_W, (nbytes+15)/16, (uint32_t*)d_sink); }, reps);
            const double gbs_stream = nbytes / (t_stream * 1e-3) / 1e9;
            const double floor_ms = nbytes / (gbs_stream * 1e9) * 1e3;   // this size's own ceiling

            // LOAD (tile pattern)
            const double t_load = timeit([&]{ k_tileload<<<gdim, 128>>>(d_W, out, in, rb, (uint32_t*)d_sink); }, reps);
            // UNPACK
            double t_unp;
            t_unp = 1e9;
            double tab[5] = {0,0,0,0,0};
            for (int ab = 0; ab < 5; ++ab) {
                const int rr = ab ? 5 : reps;
                if      (ty == TY_IQ3S)   tab[ab] = time_unpack_ab<TY_IQ3S>  (ab, d_W, mt, out, in, rb, d_sink, gdim, rr);
                else if (ty == TY_IQ3XXS) tab[ab] = time_unpack_ab<TY_IQ3XXS>(ab, d_W, mt, out, in, rb, d_sink, gdim, rr);
                else if (ty == TY_IQ2S)   tab[ab] = time_unpack_ab<TY_IQ2S>  (ab, d_W, mt, out, in, rb, d_sink, gdim, rr);
                else                      tab[ab] = time_unpack_ab<TY_IQ4XS> (ab, d_W, mt, out, in, rb, d_sink, gdim, rr);
            }
            t_unp = tab[0];
            if (getenv("MMSB_ABLATE")) printf("#  ablate %s %dx%d: full %.1f us | nogrid %.1f | nomask %.1f | metadataonly %.1f | gridmaskonly %.1f (floor %.1f)\n",
                TYNAME[ty], in, out, tab[0]*1e3, tab[1]*1e3, tab[2]*1e3, tab[3]*1e3, tab[4]*1e3, floor_ms*1e3);
            (void)tab;

            const char * shs = "";
            char shbuf[64];
            snprintf(shbuf, sizeof(shbuf), "%dx%d", in, out); shs = shbuf;
            const double gbs_load = nbytes / (t_load * 1e-3) / 1e9;
            const double gbs_unp  = nbytes / (t_unp  * 1e-3) / 1e9;
            printf("%s,%s,0,%.1f,%.1f,%.3f,%.1f,%.3f,0,0,%.1f\n", TYNAME[ty], shs, gbs_stream,
                   t_load*1e3, t_load/floor_ms, t_unp*1e3, t_unp/floor_ms, 100.0*gbs_load/gbs_stream);
            (void)gbs_unp;
            for (int i = 0; i < 6; ++i) {
                const int nc = NC[i];
                double tf;
                if      (ty == TY_IQ3S)   tf = timeit([&]{ k_mma<TY_IQ3S,0><<<gdim,128>>>(d_W, d_Aq, d_Ad, mt, out, in, rb, nc, 16, d_out); }, reps);
                else if (ty == TY_IQ3XXS) tf = timeit([&]{ k_mma<TY_IQ3XXS,0><<<gdim,128>>>(d_W, d_Aq, d_Ad, mt, out, in, rb, nc, 16, d_out); }, reps);
                else if (ty == TY_IQ2S)   tf = timeit([&]{ k_mma<TY_IQ2S,0><<<gdim,128>>>(d_W, d_Aq, d_Ad, mt, out, in, rb, nc, 16, d_out); }, reps);
                else                      tf = timeit([&]{ k_mma<TY_IQ4XS,0><<<gdim,128>>>(d_W, d_Aq, d_Ad, mt, out, in, rb, nc, 16, d_out); }, reps);
                double tf4;
                if      (ty == TY_IQ3S)   tf4 = timeit([&]{ launch_v4<TY_IQ3S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ3XXS) tf4 = timeit([&]{ launch_v4<TY_IQ3XXS> (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ2S)   tf4 = timeit([&]{ launch_v4<TY_IQ2S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else                      tf4 = timeit([&]{ launch_v4<TY_IQ4XS>  (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                double tf5 = 0.0;
                if (ty == TY_IQ3S)        tf5 = timeit([&]{ launch_v5<TY_IQ3S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ3XXS) tf5 = timeit([&]{ launch_v5<TY_IQ3XXS> (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                double tf7;
                if      (ty == TY_IQ3S)   tf7 = timeit([&]{ launch_v7<TY_IQ3S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ3XXS) tf7 = timeit([&]{ launch_v7<TY_IQ3XXS> (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ2S)   tf7 = timeit([&]{ launch_v7<TY_IQ2S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else                      tf7 = timeit([&]{ launch_v7<TY_IQ4XS>  (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                if (getenv("MMSB_V7")) printf("#  v7 %s %s nc=%d us=%.1f x=%.3f\n", TYNAME[ty], shs, nc, tf7*1e3, tf7/floor_ms);
                double tf6;
                if      (ty == TY_IQ3S)   tf6 = timeit([&]{ launch_v6<TY_IQ3S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ3XXS) tf6 = timeit([&]{ launch_v6<TY_IQ3XXS> (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ2S)   tf6 = timeit([&]{ launch_v6<TY_IQ2S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else                      tf6 = timeit([&]{ launch_v6<TY_IQ4XS>  (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                double tf3, tf2;
                if      (ty == TY_IQ3S)   tf3 = timeit([&]{ launch_v3<TY_IQ3S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ3XXS) tf3 = timeit([&]{ launch_v3<TY_IQ3XXS> (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else if (ty == TY_IQ2S)   tf3 = timeit([&]{ launch_v3<TY_IQ2S>   (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                else                      tf3 = timeit([&]{ launch_v3<TY_IQ4XS>  (d_W, d_Aq, d_Ad, mt, out, in, rb, nc, d_out, gdim, 0); }, reps);
                tf2 = 0.0;
                printf("%s,%s,%d,%.1f,0,0,0,0,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f,%.1f,%.3f,%.1f\n", TYNAME[ty], shs, nc, gbs_stream,
                       tf*1e3, tf/floor_ms, 100.0*(nbytes/(tf*1e-3)/1e9)/gbs_stream,
                       tf2*1e3, tf2/floor_ms, 100.0*(nbytes/(tf2*1e-3)/1e9)/gbs_stream,
                       tf3*1e3, tf3/floor_ms, 100.0*(nbytes/(tf3*1e-3)/1e9)/gbs_stream,
                       tf4*1e3, tf4/floor_ms, 100.0*(nbytes/(tf4*1e-3)/1e9)/gbs_stream,
                       tf5*1e3, tf5/floor_ms, 100.0*(nbytes/(tf5*1e-3)/1e9)/gbs_stream,
                       tf6*1e3, tf6/floor_ms, 100.0*(nbytes/(tf6*1e-3)/1e9)/gbs_stream,
                       tf7*1e3, tf7/floor_ms, 100.0*(nbytes/(tf7*1e-3)/1e9)/gbs_stream);
                fflush(stdout);
            }
            cudaFree(d_W); cudaFree(d_Aq); cudaFree(d_Ad); cudaFree(d_out); cudaFree(d_sink);
        }
    }
    cudaFree(d_mask8); cudaFree(d_xxs);
    return 0;
}
