// mma_layout_test.cu - determine the exact mma.m16n8k32 / m16n8k16 (int8) fragment layouts empirically.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ int ld4(const int8_t * p) { return *(const int *)p; }

// variant bit 0: A row grouping (0: a0/a1 = k low rows gr/gr+8 ; 1: a0/a2 = rows gr, k low/high)
// variant bit 1: B k grouping (0: b0 = k 4t, b1 = k 4t+16 ; 1: b1 = k 4t+8)
// variant bit 2: D mapping (0: d0,d1 = cols 2t,2t+1 row gr ; 1: d0,d2 = row gr/gr+8 col 2t)
__global__ void k32(const int8_t * __restrict__ A, const int8_t * __restrict__ B, int * __restrict__ D, int variant) {
    const int lane = threadIdx.x & 31, t = lane & 3, gr = lane >> 2;
    int a0, a1, a2, a3, b0, b1;
    const int kv = 4*t;
    if ((variant & 1) == 0) {
        a0 = ld4(A + (size_t)gr*32 + kv);
        a1 = ld4(A + (size_t)(gr+8)*32 + kv);
        a2 = ld4(A + (size_t)gr*32 + kv + 16);
        a3 = ld4(A + (size_t)(gr+8)*32 + kv + 16);
    } else {
        a0 = ld4(A + (size_t)gr*32 + kv);
        a1 = ld4(A + (size_t)gr*32 + kv + 16);
        a2 = ld4(A + (size_t)(gr+8)*32 + kv);
        a3 = ld4(A + (size_t)(gr+8)*32 + kv + 16);
    }
    b0 = ld4(B + (size_t)kv*8 + gr);
    b1 = ld4(B + (size_t)((variant & 2) ? kv + 8 : kv + 16)*8 + gr);
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    if ((variant & 4) == 0) {
        D[(size_t)gr*8 + 2*t + 0] = d0;  D[(size_t)gr*8 + 2*t + 1] = d1;
        D[(size_t)(gr+8)*8 + 2*t + 0] = d2;  D[(size_t)(gr+8)*8 + 2*t + 1] = d3;
    } else {
        D[(size_t)gr*8 + 2*t + 0] = d0;  D[(size_t)(gr+8)*8 + 2*t + 0] = d1;
        D[(size_t)gr*8 + 2*t + 1] = d2;  D[(size_t)(gr+8)*8 + 2*t + 1] = d3;
    }
}

__global__ void k16(const int8_t * __restrict__ A, const int8_t * __restrict__ B, int * __restrict__ D, int variant) {
    const int lane = threadIdx.x & 31, t = lane & 3, gr = lane >> 2;
    const int kv = 4*t;
    int a0, a1, b0;
    if (variant == 0) {
        a0 = ld4(A + (size_t)gr*16 + kv);
        a1 = ld4(A + (size_t)(gr+8)*16 + kv);
    } else {
        a0 = ld4(A + (size_t)gr*16 + kv);
        a1 = ld4(A + (size_t)gr*16 + kv + 4);
    }
    b0 = ld4(B + (size_t)kv*8 + gr);
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(b0));
    D[(size_t)gr*8 + 2*t + 0] = d0;  D[(size_t)gr*8 + 2*t + 1] = d1;
    D[(size_t)(gr+8)*8 + 2*t + 0] = d2;  D[(size_t)(gr+8)*8 + 2*t + 1] = d3;
}

int main() {
    const int M = 16, K = 32, N = 8;
    int8_t A[M*K], B[K*N];
    for (int i = 0; i < M*K; ++i) A[i] = (int8_t)((i % 21) - 10);
    for (int i = 0; i < K*N; ++i) B[i] = (int8_t)((i % 13) - 6);
    int ref[M*N];
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        int s = 0; for (int k = 0; k < K; ++k) s += (int)A[i*K+k] * (int)B[k*N+j];
        ref[i*N+j] = s;
    }
    int8_t *dA; int8_t *dB; int *dD;
    cudaMalloc(&dA, sizeof(A)); cudaMalloc(&dB, sizeof(B)); cudaMalloc(&dD, sizeof(ref));
    cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);
    for (int v = 0; v < 8; ++v) {
        cudaMemset(dD, 0, sizeof(ref));
        k32<<<1,32>>>(dA, dB, dD, v);
        int got[M*N];
        cudaMemcpy(got, dD, sizeof(ref), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < M*N; ++i) if (got[i] != ref[i]) ++bad;
        printf("k32 variant %d (A%d B%d D%d): %d/%d wrong  first: got %d ref %d\n", v,
               v&1, (v&2)>>1, (v&4)>>2, bad, M*N, got[0], ref[0]);
    }
    // k=16
    for (int i = 0; i < M*K; ++i) A[i] = (int8_t)((i % 21) - 10);
    for (int v = 0; v < 2; ++v) {
        for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
            int s = 0; for (int k = 0; k < 16; ++k) s += (int)A[i*K+k] * (int)B[k*N+j];
            ref[i*N+j] = s;
        }
        cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
        cudaMemset(dD, 0, sizeof(ref));
        k16<<<1,32>>>(dA, dB, dD, v);
        int got[M*N];
        cudaMemcpy(got, dD, sizeof(ref), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < M*N; ++i) if (got[i] != ref[i]) ++bad;
        printf("k16 variant %d: %d/%d wrong  first: got %d ref %d\n", v, bad, M*N, got[0], ref[0]);
    }
    return 0;
}
