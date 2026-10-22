# NInfer borrow-analysis for Qwen3.8-27B on RTX 3060

Repo: https://github.com/Neroued/ninfer (cloned to ./ninfer, master 2026-10-16)
Target: Qwen3.5/Qwen3.8-27B dense + 35B-A3B MoE on single RTX 5090 (sm_120a). Same model arch as ours.

## 1. Their claimed numbers (5090, Qwen3.8-27B) vs ours (3060)

| metric | ninfer (5090, groupwise-int 16.6 GiB) | ninfer (5090, nvfp4 19.0 GiB) | ours (3060, IQ2_S 8.61 GiB) |
|---|---:|---:|---:|
| prefill 7.7k ctx | 3,332 t/s | 12,819 t/s | ~556 t/s |
| decode (no spec) 7.7k | 84.1 t/s | 74.1 t/s | ~21-21.5 t/s |
| decode MTP3 | 123-215 t/s | 132-207 t/s | (MTP earlier: ~30-40 t/s class) |
| decode DFlash2 K=7 | 115-350 t/s | 129-377 t/s | 11.4 (dflash standalone) |
| decode eff. BW | ~1,500 GB/s = 80% of 1674.5 sustained-read | — | ~200-210 GB/s = 60% of 340 |
| KV | fp8 e4m3 row-256 | fp8 | q8_0 (1.0625 B/elem, ALU dequant) |

So: on 5090 they run decode at ~80% of measured HBM read ceiling; we run ~60% of our GDDR6
ceiling. The gap is partly hardware (fp8/fp4 tensor cores, 5x BW) and partly technique.

## 2. What they do (evidence from docs/src)

- Weight formats: q4_g64_fp16 / q5/q6 / q8_g32_fp16 (fp16 scale per 64: 4.25-8.5 bpw),
  nvfp4 (E2M1 + E4M3 scale/16), fp8_e4m3 rows, nf4_g16 codebook. Row-split storage
  (codes then scales, contiguous per output row) — tensor-formats.md, storage-layouts.md.
- Linear dispatch: dedicated T=1 GEMV kernels (q4/fp8/nvfp4 `*_a16_gemv`), MMA for T>=2,
  sliced-K MMA routes for prefill T=512/1024; per-T instances tuned to priority points
  1/4/8/16../128 (linear-tuning.md). GEMV uses multi-stage cp.async tile prefetch into
  shared (kStages up to 8), several warps per row, fp16x2 scale handling.
  Q4 6144x5120 T=1 microbench: 1069 GB/s = 63.9% READ_% — for a single op even on 5090.
- Attention/KV: fp8 e4m3 row-256 KV, paged, kata layouts (kv_cache/, linear_attention/).
- CUDA Graphs for decode with cudaGraphExecUpdate (core/decode_graph.cpp).
- Next-layer weight prefetch (`projection_prefetch` / `next_projection_hints`): consumed ONLY
  by the MoE paths (expert-weight window cache, 8 MiB pages double-buffered) for the
  Qwen3.6-35B-A3B / Qwen4Exp MoE models. The dense 27B path (`dense_mlp`) never receives
  hints — no dense-layer weight staging.
- Host budget: 8 GiB pinned host + 8 model StateImages is a CAPACITY BACKSTOP, not a
  steady-state path: device-first execution; host intercepts only pause/resume snapshots,
  retained-prefix cache and eviction fallbacks (resource-scheduling-and-context-cache.md:
  "Device 槽紧张且有 Host 副本时…降级"). Steady-state decode of the dense 27B on the 5090
  is 100% device-resident (weights 16.6-19 GiB + fp8 KV ~4.6 GiB at 240k fits 32 GB).
  (Correction: this is NOT a precedent for per-token parallel KV/weight offload.)
- GDN: fused recurrence + ReplaySSM for speculation (48 layers x 48 value heads x 64 KiB
  = 144 MiB recurrent state image — matches llama.cpp RS buffer 149.6 MiB we measured).
- MTP3 / DFlash2 via optimized proposal head + lm-head-draft; prefill chunk 1024; FIFO
  lanes 1..8 for concurrency.
- Roofline methodology: HBM probe (tools/hbm_bandwidth_probe.cu), READ_% vs nominal;
  the same metric I used in bw_test.cu (340 GB/s measured on 3060).

## 3. What transfers to the 3060 (measured evidence)

### Experiment I ran today: IQ3_XXS-no-mtp -> Q3_K_S (--pure), 11.58 GB, 3.44 BPW
llama-bench tg128: **13.95 t/s (161 GB/s effective)** vs IQ3_XXS-no-mtp 9.39 GiB:
**20.83 t/s (210 GB/s)**. CONCLUSION (table flipped): llama.cpp's mmvq on Ampere already
handles the ISTA GSQ mixed-IQ format better than the generic k-quant path per byte. The
tiny-scale groupwise formats help ninfer mainly because they unlock tensor-core MMA on
sm_120a; on sm_86 they would not beat the current IQ path. DO NOT requantize to Q3_K/Q4_K.

### Borrowable, high value (today, no code):
1. KV dtype: they use fp8 e4m3 row-256 -> on 3060, minimum-equivalent = f16 (native, llama
   supports) or q4_0/q5_0 (halves KV bytes and dequant cost vs q8_0). Test q5_0/q4_0 KV at
   32-72K to cut the ~6.8+2 ms attention+dequant we measured per token.
2. MTP3/DFlash: llama.cpp has draft-mtp + draft-dflash already (we ran them earlier on this
   box; ninfer shows acceptance 55-77% for MTP3, ~8-40 t/s -> 1.8-2.3x decode on 5090). With
   dflash2 Q2 draft we can test K=7 on 3060 to amortize the 24 ms weight stream.
3. Prefill ubatch: they chunk prefill at 1024; llama.cpp default 512 -> try -b/-ub 1024/2048
   for an occupancy-bound PP gain (PP is NOT BW-bound on 3060: 9.2 GB / 0.92 s = 10 GB/s).
4. CUDA graphs: llama-server uses them; verify they are not disabled for qwen35 (our decode
   showed 40-60% host idle -> graph capture may not be active; check server log
   "capturing graph" lines).

### Borrowable, code work (port concepts into llama.cpp/kernels):
5. Dedicated T=1 GEMV with cp.async staged prefetch + rowsplit layout for the formats we
   keep (iq3_xxs/q2_K-ish). llama.cpp has mmvq but no multi-stage cp.async prefetch for
   decode; this is exactly the "hide dequant ALU under DRAM" trick that could push the
   weight stream from 210 toward 280-320 GB/s on 3060. Biggest kernel win available.
6. fp8 KV path: llama.cpp has no f8 cache type; adding f8_e4m3 decode (convert-to-f16 in
   flash kernel) is a well-scoped change (halves KV bytes, kills q8_0 dequant).
7. (Corrected) ninfer's MoE expert-weight window cache (8 MiB double-buffered pages,
   host↔device transfer + prefetch/lease pipeline) is the model for a staged-weight engine;
   the same plumbing transfers to llama.cpp if we choose to stage GDN-layer weights or KV on
   host for the 3060, but ninfer itself does NOT do per-token staged offload for the dense
   27B on the 5090 (all-device, fits 32 GB). Our sysmem-KV plan remains an original
   experiment the codebase proves feasible only as capacity/retention machinery.
8. Host-resident KV/state: in ninfer this is pause/retained-prefix/eviction backstop only
   (device-first); the "cache in sysmem, stage per layer" steady-state design for our
   12 GiB 3060 stays our own thing to validate — borrow the pinned-backing + transfer
   + prefetch/lease *API design*, not a demonstrated free-lunch.

### Not transferable (hardware-bound):
- fp8/fp4 tensor-core MMA & NVFP4 (sm_120a-only, no path on sm_86).
- Their 12x prefill numbers (5090 MMA/TFLOPS, fp8 A8 kernels) — on 3060 keep optimizing
  MMQ occupancy; expect only incremental PP gains.
- 1-8 lane serving scheduler — irrelevant to single-stream decode latency.

## 4. Action list (priority order for our box)
1. (a) KV: bench q5_0/q4_0 K/V at 4K and 72K ctx — expected +1-3 t/s at 32K, frees VRAM.
   (b) PP: try ub 1024-2048.
2. Verify CUDA-graph capture in server decode (host idle ~40% is the visible symptom).
3. MTP3 / DFlash2 long-draft K=7 acceptance runs on IQ2_S (its MTP tensors exist) — target
   >1.5x decode.
4. Kernel work: T=1 cp.async rowsplit GEMV for the mixed-IQ formats (the 210->290 GB/s play).
5. Offload: host pinned KV + per-layer prefetch staging (their production design == our plan).