// wstage_register_test.cu - can a GGUF region be pinned in place on Windows?
// M2 precursor: compare HostToDevice copy bandwidth into a 70 MiB device slot from
//   (a) a pageable heap buffer          (what a malloc'd mirror costs without pinning)
//   (b) a read-only file mapping of the GGUF (what the mmap'd weights actually are)
//   (c) the same file mapping after cudaHostRegister(..., cudaHostRegisterReadOnly)
//   (d) an explicit pinned mirror (cudaHostAlloc)
// Usage: wstage_register_test.exe <gguf path> [MiB=70]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <windows.h>

static void chk(cudaError_t e, const char * what) {
    if (e != cudaSuccess) { printf("  %-46s FAILED: %s\n", what, cudaGetErrorString(e)); }
}
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double copy_rate(void * dst, const void * src, size_t bytes, int reps, int chunks) {
    cudaStream_t s; cudaStreamCreate(&s);
    // warm-up
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_s();
        for (int c = 0; c < chunks; ++c) {
            const size_t off = bytes * c / chunks;
            const size_t len = bytes / chunks;
            cudaMemcpyAsync((char *) dst + off, (const char *) src + off, len, cudaMemcpyHostToDevice, s);
        }
        cudaStreamSynchronize(s);
        t.push_back(now_s() - t0);
    }
    cudaStreamDestroy(s);
    std::sort(t.begin(), t.end());
    return (double) bytes / (t[t.size() / 2] * 1e9);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "D:/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf";
    const size_t mib = argc > 2 ? (size_t) atoi(argv[2]) : 70;
    const size_t bytes = mib << 20;
    const int reps = 9, chunks = 6;
    printf("region %zu MiB from %s, %d chunks/copy, %d reps\n", mib, path, chunks, reps);

    void * slot = nullptr;
    if (cudaMalloc(&slot, bytes) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }

    // (a) pageable heap
    void * heap = malloc(bytes);
    memset(heap, 0, bytes);
    printf("  (a) pageable heap        : %6.2f GB/s\n", copy_rate(slot, heap, bytes, reps, chunks));

    // (b) read-only file mapping of the GGUF
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hf == INVALID_HANDLE_VALUE) { printf("  cannot open %s (err %lu)\n", path, GetLastError()); return 1; }
    HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hm) { printf("  CreateFileMapping failed (err %lu)\n", GetLastError()); return 1; }
    const char * map = (const char *) MapViewOfFile(hm, FILE_MAP_READ, 0, 0, bytes);
    if (!map) { printf("  MapViewOfFile failed (err %lu)\n", GetLastError()); return 1; }
    printf("  (b) file mapping (raw)   : %6.2f GB/s\n", copy_rate(slot, map, bytes, reps, chunks));

    // (c) cudaHostRegister on the file mapping
    cudaError_t re = cudaHostRegister((void *) map, bytes, cudaHostRegisterReadOnly);
    if (re == cudaSuccess) {
        printf("  (c) file mapping+register: %6.2f GB/s  [cudaHostRegister OK]\n", copy_rate(slot, map, bytes, reps, chunks));
        cudaHostUnregister((void *) map);
    } else {
        printf("  (c) file mapping+register: cudaHostRegister FAILED: %s\n", cudaGetErrorString(re));
        // try without the read-only flag, in case the flag is what Windows rejects
        cudaError_t re2 = cudaHostRegister((void *) map, bytes, 0);
        if (re2 == cudaSuccess) {
            printf("      (c2) register without ReadOnly: %6.2f GB/s\n", copy_rate(slot, map, bytes, reps, chunks));
            cudaHostUnregister((void *) map);
        } else {
            printf("      (c2) register without ReadOnly also FAILED: %s\n", cudaGetErrorString(re2));
        }
    }

    // (d) explicit pinned mirror
    void * pin = nullptr;
    if (cudaHostAlloc(&pin, bytes, cudaHostAllocDefault) == cudaSuccess) {
        memcpy(pin, map, bytes);
        printf("  (d) pinned mirror        : %6.2f GB/s  [load-time copy of %zu MiB]\n",
               copy_rate(slot, pin, bytes, reps, chunks), mib);
        cudaFreeHost(pin);
    }
    UnmapViewOfFile(map); CloseHandle(hm); CloseHandle(hf); free(heap); cudaFree(slot);
    return 0;
}
