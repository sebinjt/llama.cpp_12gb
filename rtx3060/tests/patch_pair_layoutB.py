import re
p = 'tests/mmsb_bench_v9.cu'
s = open(p, 'rb').read().decode('utf-8')

def L(t):
    return t.replace('\n', '\r\n')

# ------------------------------------------------------------------ header comment
i = s.find('// Word map. The 880 words are partitioned per QUAD')
j = s.find('// in-place host repack: natural blocks -> paired planar chunks')
assert i > 0 and j > i
new_map = L("""// Word map. Instead of the flat "word w" map, the 16 row chunks of a tile-block are assigned one
// (half h, 32-k step) each: chunk index = h*8 + step. A chunk then holds ONE step for all 8 quads, and
// the 8 quads' 24-B slots are contiguous (0..191), so a warp's whole weight read for a step is a single
// 192-B burst instead of 8 scattered 24-B pieces. Every offset below is a compile-time constant, so the
// 128-entry offset table of the flat word map collapses to an immediate plus the unrolled step index:
// no LDS, no per-access division.
//   IQ3_S, 220 B per chunk (8 slots x 24 B = 192, then qh/scale/d):
//     [24*quad + 0..15]  qs      word 4t = {qA[t], qA[t+4], qB[t], qB[t+4]}  (lane-adjacent, 4 B)
//     [24*quad + 16..19] signsA  u32 {s0, s2, s1, s3}; lane reads u16 at 2*(t>>1)
//     [24*quad + 20..23] signsB  u32
//     [192 + 2*quad]     qh      u16 = qhA | qhB << 8
//     [208 + quad]       scale   u8  = scA | scB << 4
//     [216..219]         d       u16 dA at 216, u16 dB at 218 (of quad = step, half h)
//   IQ3_XXS, 196 B per chunk: [24*quad + 0..15] qs, [+16..19] auxA u32, [+20..23] auxB u32,
//     [192..195] d (u16 dA, u16 dB).  8*24 + 4 = 196 exactly: all 49 words, 0 spare.
// d is one fp16 per row per superblock: 16 rows x 2 halves x 2 B = 64 B per tile-block = 4 B per chunk,
// so the d word of (half h, quad q) is stored in chunk h*8+q. That is all 880 (784) words, 0 spare.
// Every field lies inside one u32 (2-B fields at even offsets) and no field straddles a chunk; the
// repack is a permutation of the tensor's own bytes within each tile-block, verified by the
// byte-exact round trip below.
// =============================================================================================
static inline int pair_blk(int ty) { return (ty == TY_IQ3S) ? 110 : 98; }
static inline int pair_cw (int ty) { return (ty == TY_IQ3S) ? 55  : 49; }   // u32 per row chunk

""")
# drop the old helper definitions that followed the old comment (they are re-added above)
k = s.find('static inline int pair_blk(int ty)')
k2 = s.find('// in-place host repack: natural blocks -> paired planar chunks')
s = s[:i] + new_map + s[k2:]

# ------------------------------------------------------------------ repack
i = s.find('// in-place host repack: natural blocks -> paired planar chunks')
j = s.find('// exact inverse of pair_repack')
assert i > 0 and j > i
s = s[:i] + L("""// in-place host repack: natural blocks -> paired planar chunks (same bytes, permuted)
static void pair_repack(uint8_t * W, int ty, int K, int nrows) {
    const int blk = pair_blk(ty), cw = pair_cw(ty), chunk = cw * 4;
    const int nsb = K/256, np = nsb/2;
    const size_t rb = (size_t)nsb * blk;
    if ((nsb & 1) || (nrows & 15)) { fprintf(stderr, "pair_repack: guard failed nsb=%d nrows=%d\\n", nsb, nrows); return; }
    std::vector<uint8_t> nat((size_t)16 * chunk);
    std::vector<uint8_t> pl ((size_t)16 * chunk);
    for (int rt = 0; rt < nrows/16; ++rt) {
        for (int j = 0; j < np; ++j) {
            for (int r = 0; r < 16; ++r)                                   // stage before overwriting
                memcpy(&nat[(size_t)r*chunk], W + (size_t)(rt*16 + r)*rb + (size_t)j*chunk, chunk);
            for (int h = 0; h < 2; ++h) {
                for (int step = 0; step < 8; ++step) {
                    uint8_t * c = &pl[(size_t)(h*8 + step)*chunk];          // chunk = this (half, step)
                    for (int q = 0; q < 8; ++q) {
                        const uint8_t * bA = &nat[(size_t) q      *chunk] + (size_t)h*blk;
                        const uint8_t * bB = &nat[(size_t)(q + 8) *chunk] + (size_t)h*blk;
                        uint8_t * s = c + 24*q;
                        for (int t = 0; t < 4; ++t) {
                            uint8_t * p = s + 4*t;
                            p[0] = bA[2 + 8*step + t];
                            p[1] = bA[2 + 8*step + t + 4];
                            p[2] = bB[2 + 8*step + t];
                            p[3] = bB[2 + 8*step + t + 4];
                        }
                        if (ty == TY_IQ3S) {
                            uint8_t * sa = s + 16;
                            uint8_t * sb = s + 20;
                            for (int kk = 0; kk < 4; ++kk) {
                                const int src = (kk == 1) ? 2 : (kk == 2) ? 1 : kk;   // {s0,s2,s1,s3}
                                sa[kk] = bA[74 + 4*step + src];
                                sb[kk] = bB[74 + 4*step + src];
                            }
                            c[192 + 2*q]     = bA[66 + step];                     // qh row A
                            c[192 + 2*q + 1] = bB[66 + step];                     // qh row B
                            const int sh = (step & 1) * 4;
                            c[208 + q] = (uint8_t)(((bA[106 + (step >> 1)] >> sh) & 0xF) |
                                                   (((bB[106 + (step >> 1)] >> sh) & 0xF) << 4));
                        } else {
                            memcpy(s + 16, bA + 66 + 4*step, 4);                  // aux A (7-bit masks + scale)
                            memcpy(s + 20, bB + 66 + 4*step, 4);                  // aux B
                        }
                    }
                }
                for (int q = 0; q < 8; ++q) {                                     // d of (half h, quad q) -> chunk h*8+q
                    const uint8_t * bA = &nat[(size_t) q      *chunk] + (size_t)h*blk;
                    const uint8_t * bB = &nat[(size_t)(q + 8) *chunk] + (size_t)h*blk;
                    uint8_t * dw = &pl[(size_t)(h*8 + q)*chunk] + ((ty == TY_IQ3S) ? 216 : 192);
                    memcpy(dw,     bA, 2);   // d of row A for this half
                    memcpy(dw + 2, bB, 2);   // d of row B for this half
                }
            }
            for (int r = 0; r < 16; ++r)
                memcpy(W + (size_t)(rt*16 + r)*rb + (size_t)j*chunk, &pl[(size_t)r*chunk], chunk);
        }
    }
}

""") + s[j:]

# ------------------------------------------------------------------ unpack
i = s.find('// exact inverse of pair_repack')
j = s.find('template<int TY, int SIGN>\r\n__device__ __forceinline__ int pair_grp')
assert i > 0 and j > i
s = s[:i] + L("""// exact inverse of pair_repack: paired planar chunks -> natural blocks
static void pair_unpack(const uint8_t * W, int ty, int K, int nrows, std::vector<uint8_t> & out) {
    const int blk = pair_blk(ty), cw = pair_cw(ty), chunk = cw * 4;
    const int nsb = K/256, np = nsb/2;
    const size_t rb = (size_t)nsb * blk;
    out.assign(rb * (size_t)nrows, 0);
    std::vector<uint8_t> nat((size_t)16 * chunk);
    std::vector<uint8_t> pl ((size_t)16 * chunk);
    for (int rt = 0; rt < nrows/16; ++rt) {
        for (int j = 0; j < np; ++j) {
            for (int r = 0; r < 16; ++r)
                memcpy(&pl[(size_t)r*chunk], W + (size_t)(rt*16 + r)*rb + (size_t)j*chunk, chunk);
            memset(nat.data(), 0, nat.size());
            for (int h = 0; h < 2; ++h) {
                for (int step = 0; step < 8; ++step) {
                    const uint8_t * c = &pl[(size_t)(h*8 + step)*chunk];
                    for (int q = 0; q < 8; ++q) {
                        uint8_t * bA = &nat[(size_t) q      *chunk] + (size_t)h*blk;
                        uint8_t * bB = &nat[(size_t)(q + 8) *chunk] + (size_t)h*blk;
                        const uint8_t * s = c + 24*q;
                        for (int t = 0; t < 4; ++t) {
                            const uint8_t * p = s + 4*t;
                            bA[2 + 8*step + t]     = p[0];
                            bA[2 + 8*step + t + 4] = p[1];
                            bB[2 + 8*step + t]     = p[2];
                            bB[2 + 8*step + t + 4] = p[3];
                        }
                        if (ty == TY_IQ3S) {
                            const uint8_t * sa = s + 16;
                            const uint8_t * sb = s + 20;
                            for (int kk = 0; kk < 4; ++kk) {
                                const int dst = (kk == 1) ? 2 : (kk == 2) ? 1 : kk;
                                bA[74 + 4*step + dst] = sa[kk];
                                bB[74 + 4*step + dst] = sb[kk];
                            }
                            bA[66 + step] = c[192 + 2*q];
                            bB[66 + step] = c[192 + 2*q + 1];
                            const uint8_t sc = c[208 + q];
                            const int sh = (step & 1) * 4;
                            bA[106 + (step >> 1)] = (uint8_t)((bA[106 + (step >> 1)] & ~(0xF << sh)) | ((sc & 0xF) << sh));
                            bB[106 + (step >> 1)] = (uint8_t)((bB[106 + (step >> 1)] & ~(0xF << sh)) | (((sc >> 4) & 0xF) << sh));
                        } else {
                            memcpy(bA + 66 + 4*step, s + 16, 4);
                            memcpy(bB + 66 + 4*step, s + 20, 4);
                        }
                    }
                }
                for (int q = 0; q < 8; ++q) {
                    uint8_t * bA = &nat[(size_t) q      *chunk] + (size_t)h*blk;
                    uint8_t * bB = &nat[(size_t)(q + 8) *chunk] + (size_t)h*blk;
                    const uint8_t * dw = &pl[(size_t)(h*8 + q)*chunk] + ((ty == TY_IQ3S) ? 216 : 192);
                    memcpy(bA, dw,     2);
                    memcpy(bB, dw + 2, 2);
                }
            }
            for (int r = 0; r < 16; ++r)
                memcpy(&out[(size_t)(rt*16 + r)*rb + (size_t)j*chunk], &nat[(size_t)r*chunk], chunk);
        }
    }
}

""") + s[j:]

# ------------------------------------------------------------------ kernel inner loops
old = L("""            for (int h = 0; h < 2; ++h) {
                const uint8_t * c = tile + (size_t)(2*q + h) * rowbytes + (size_t)j * (2*CW*4);
                uint32_t qhw[4], scw[2], dw;
                if constexpr (TY == TY_IQ3S) {
                    for (int i = 0; i < 4; ++i) qhw[i] = *(const uint32_t *)(c + 192 + 4*i);   // qh for 8 steps
                    for (int i = 0; i < 2; ++i) scw[i] = *(const uint32_t *)(c + 208 + 4*i);   // scale for 8 steps
                    dw = *(const uint32_t *)(c + 216);
                } else {
                    dw = *(const uint32_t *)(c + 192);
                }
                const float dA = __half2float(__ushort_as_half((unsigned short)(dw & 0xFFFFu)));
                const float dB = __half2float(__ushort_as_half((unsigned short)(dw >> 16)));
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    const uint8_t * s = c + 24*step;                    // this step's 24-B slot
                    const uint32_t q4 = *(const uint32_t *)(s + 4*t);
                    const int kb = kb0 + h*8 + step;
                    const int8_t * ab = Aq + (size_t)kb * (16*32);
                    int a0, a1, a2, a3; float swA, swB;
                    if constexpr (TY == TY_IQ3S) {
                        const uint32_t sA = *(const uint16_t *)(s + 16 + 2*(t>>1));
                        const uint32_t sB = *(const uint16_t *)(s + 20 + 2*(t>>1));
                        const uint32_t qh16 = (qhw[step>>1] >> (16*(step&1))) & 0xFFFFu;
                        const int qhA = (int)( qh16      & 0xFFu);
                        const int qhB = (int)((qh16 >> 8) & 0xFFu);
                        const int sc  = (int)((scw[step>>2] >> (8*(step&3))) & 0xFFu);""")
new = L("""            for (int h = 0; h < 2; ++h) {
                // d of (half h, quad q) sits in chunk h*8+q, bytes 216 (A) / 218 (B) or 192 for IQ3_XXS
                const uint8_t * dch = tile + (size_t)(h*8 + q) * rowbytes + (size_t)j * (2*CW*4);
                const uint32_t dw = *(const uint32_t *)(dch + ((TY == TY_IQ3S) ? 216 : 192));
                const float dA = __half2float(__ushort_as_half((unsigned short)(dw & 0xFFFFu)));
                const float dB = __half2float(__ushort_as_half((unsigned short)(dw >> 16)));
#pragma unroll
                for (int step = 0; step < 8; ++step) {
                    const uint8_t * c = tile + (size_t)(h*8 + step) * rowbytes + (size_t)j * (2*CW*4);
                    const uint8_t * s = c + 24*q;                       // this quad's 24-B slot
                    const uint32_t q4 = *(const uint32_t *)(s + 4*t);
                    const int kb = kb0 + h*8 + step;
                    const int8_t * ab = Aq + (size_t)kb * (16*32);
                    int a0, a1, a2, a3; float swA, swB;
                    if constexpr (TY == TY_IQ3S) {
                        const uint32_t sA = *(const uint16_t *)(s + 16 + 2*(t>>1));
                        const uint32_t sB = *(const uint16_t *)(s + 20 + 2*(t>>1));
                        const int qh16 = (int)(*(const uint16_t *)(c + 192 + 2*q));
                        const int qhA = ( qh16      & 0xFF);
                        const int qhB = ((qh16 >> 8) & 0xFF);
                        const int sc  = (int)c[208 + q];""")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

open(p, 'wb').write(s.encode('utf-8'))
print('patched: single-stream chunk map')
