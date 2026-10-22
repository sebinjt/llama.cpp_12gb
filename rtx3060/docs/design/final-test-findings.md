# Historical test notes — not a validated final performance report

**Superseded by [staged-kv-implementation.md](staged-kv-implementation.md).**
The old `llama-bench` tests did not establish the advertised context ceilings:
standalone `tg16` starts with a small cache, and `pp+tg` reports combined time, not
isolated long-context decode. The transport harness did not load Qwen weights.
Do not use the numbers below as real 128K Qwen inference measurements.

Model IQ3_XXS-mtp (9.72 GiB weights, 27.32 B params effective) · q4_0 K/V · RTX 3060 12 GiB.

## Resident (all-KV-in-VRAM) — this is what works WITHOUT offload
| context | KV q4_0 | result |
|---|---:|---|
| 96K (c=98304) | 1.87 GiB | **runs, tg16 = 19.60 t/s** |
| 128K (c=131072) | 2.50 GiB | **OOM** (weights 9.72 + KV 2.5 + GDN state 0.15 + workspace > 12 GiB) |

=> Resident ceiling for the -mtp model ≈ 96–98K. 128K is impossible resident on 12 GiB.

## So 128K on IQ3_XXS-mtp REQUIRES offload
- llama.cpp native `--no-kv-offload` (KV in host, un-staged, on-demand PCIe reads): ~3.6 t/s at ~48K, and scales worse — not viable at 128K.
- Staged per-layer host-KV (transport-validated, kv_stage2): ~17 t/s @128K.

## Tier-2 status (honest)
The staged integration into llama.cpp's graph is NOT yet shipped. Real reasons:
1. llama.cpp's ggml scheduler runs the decode graph on a single CUDA stream; there is no in-graph primitive to say "copy layer i+1's KV host->VRAM on a side stream overlapped with layer i's compute, then swap the attention's KV pointer." 
2. Per-layer buffer swap mid-graph + event choreography + MTP rollback (drafted KV must be dropped) is a ~15-20 function CUDA-backend change; shipping it unvalidated risks breaking the working llama-server build and the 96K baseline.
3. The transport design it must implement is already proven (kv_stage2: 17 t/s @128K), so the remaining work is plumbing into ggml, documented in dev/staged-kv-offload-design.md.

## Numbers currently real (committed)
- 96K resident (llama.cpp end-to-end): 19.60 t/s
- 128K resident: OOM (must offload)
- 128K staged (transport harness): ~17 t/s
