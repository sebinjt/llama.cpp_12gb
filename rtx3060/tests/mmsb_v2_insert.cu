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
__global__ void k_mma2(const uint8_t * __restrict__ W, const int8_t * __restrict__ Aq,
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
        const int n32 = V::GRID / 4;
        for (int i = threadIdx.x; i < n32; i += blockDim.x) {
            uint32_t v;
            if constexpr (TY == TY_IQ3S)        v = iq3s_grid[i];
            else if constexpr (TY == TY_IQ3XXS) v = iq3xxs_grid[i];
            else if constexpr (TY == TY_IQ2S)   v = (uint32_t)(iq2s_grid[2*i] & 0xFFFFFFFFu);
            else                                v = 0;
            ((uint32_t *)(smem + grid_off))[i] = v;
        }
        if constexpr (TY == TY_IQ2S) {
            for (int i = threadIdx.x; i < V::GRID / 8; i += blockDim.x)
                ((uint64_t *)(smem + grid_off))[i] = iq2s_grid[i];
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
            __syncthreads();
            for (int i = threadIdx.x; i < 8*16*32/4; i += blockDim.x)
                ((uint32_t *)(smem + acts_off))[i] = ((const uint32_t *)Aq)[(size_t)b*8*16*32/4 + i];
            for (int i = threadIdx.x; i < 8*16; i += blockDim.x)
                sh_asc[i] = Ad[(size_t)b*8*16 + i];
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
                    const int8_t * ab = sh_acts + (size_t)b32 * (16*32);
                    const float * asc = sh_asc + b32*16;
                    if constexpr (V::K16) {
                        const float w0A = dA * v2_wscale<TY>(sbA, b32, 0), w0B = dB * v2_wscale<TY>(sbB, b32, 0);
                        const float w1A = dA * v2_wscale<TY>(sbA, b32, 1), w1B = dB * v2_wscale<TY>(sbB, b32, 1);
                        for (int nt = 0; nt < 2; ++nt) {
                            if (nt >= ntiles) break;
                            const int col = nt*8 + gr;
                            const int b0 = *(const int *)(ab + col*32 + 4*t);
                            const int b1 = *(const int *)(ab + col*32 + 4*t + 16);
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
                            const int b0 = *(const int *)(ab + col*32 + 4*t);
                            const int b1 = *(const int *)(ab + col*32 + 4*t + 16);
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

template<int TY>
static int v2_smem_bytes() {
    using V = V2<TY>;
    return V::GRID + V::MASK + 4096 + 512 + V2_NWARPS*16*V::RAW + V2_NWARPS*V::TILE;
}

template<int TY>
static void launch_v2(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * d_mask,
                      int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    const int nbytes = v2_smem_bytes<TY>();
    cudaFuncSetAttribute((const void *)k_mma2<TY>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes);
    k_mma2<TY><<<gdim, 128, nbytes, st>>>(W, Aq, Ad, d_mask, R, K, rb, ncols, 16, out);
}

template<int TY>
static void launch_v2_check(const uint8_t * W, const int8_t * Aq, const float * Ad, const uint2 * d_mask,
                            int R, int K, int rb, int ncols, float * out, int gdim, cudaStream_t st) {
    launch_v2<TY>(W, Aq, Ad, d_mask, R, K, rb, ncols, out, gdim, st);
}
