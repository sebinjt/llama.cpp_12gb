# GPU-vs-PCIe measurements — 2026-10-17 (no MTP)

HW: RTX 3060 12 GiB (360 GB/s GDDR6, PCIe Gen3 x16), Ryzen 5 4600G, 32 GB DDR4
Model: Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf (8.61 GiB, 26.90 B params, IQ3_XXS 3.0625 bpw, no MTP spec used)
KV: q8_0 K + q8_0 V, flash-attn on, full GPU offload (-ngl 99)

## 1) Generation tps, VRAM loaded to the brink (no MTP)

| config | VRAM used | prompt t/s | generation t/s | notes |
|---|---|---|---|---|
| 4K ctx, tg128 (llama-bench) | ~72% | 556 | **21.5** | small-context decode, weights-bound |
| 72K ctx alloc, 32K tokens in cache | **11,699/12,288 MiB = 95.2%** | 426 | **16.4** | max context that loads on this setup (96K/88K/80K all OOM) |

- Mental peak used: 360/12 = 30 t/s. Achieved 21.5 t/s at small ctx = **72% of that peak**, and 16.4 t/s with VRAM at 95% = **55% of peak**.
- vs pure bandwidth bound for this file size: 360/8.61 GiB = 41.8 t/s ideal -> 21.5 t/s real = 51% (dequant+decode kernel overhead, not just BW).
- 32K of real attention costs ~5 t/s (21.5 -> 16.4). Attention grows ~linearly: at 96K expect ~12 t/s.
- Practical VRAM cap: VRAM was 95.2% full at 72K; 80K fails on compute buffers (365 MiB) due to WDDM fragmentation, 96K fails on KV itself (3.26 GiB alloc).

## 2) PCIe / DRAM<->VRAM point-to-point copy bandwidth (bw_test.exe, runs during Gen3)

| copy | 64 MiB | 256 MiB | 1 GiB |
|---|---|---|---|
| H2D pageable | 5.40 | 5.48 | 5.29 |
| H2D pinned | 13.37 | 13.40 | 13.39 |
| D2H pageable | 5.15 | 5.16 | 5.13 |
| D2H pinned | 13.14 | 13.14 | 13.08 |
| H2D/D2H async pinned | 13.36/13.13 | 13.40/13.17 | 13.39/13.11 |
| Round-trip H2D+D2H (pinned) | 6.63 (13.27/way) | 6.64 (13.27/way) | 6.61 (13.22/way) |
| H2H memcpy (DRAM ref) | 8.37 | 8.35 | 8.43 |

- PCIe link ramps to **Gen3 x16** under load. Practical one-way copy = **~13.2-13.4 GB/s** = ~84% of the 15.75 GB/s theoretical. Symmetric both directions; no round-trip penalty.
- Pageable (bounce-buffer path) is ~3x slower: **~5.1-5.5 GB/s** — use pinned (cudaHostAlloc / CUDA_Host) for anything offload-related.
- H2H memcpy 8.4 GB/s is single-thread CPU copy on the 4600G (DRAM itself is ~43-51 GB/s dual-channel DDR4-3200).

## Implication for the KV-offload plan
- Measured 13.3 GB/s replaces the 15.75 theoretical used in previous estimates (~18% worse).
- 256K ctx q8_0: 544 MiB per attention layer -> ~41 ms stage time (was 36 ms); GDN gap between attention layers must cover that.
- Ratio 360/13.3 = 27x => raw bandwidth ratio ~3.7% (was 19x / 5.25% ideal): staging window is tighter than hoped.
- Pageable copies are 5.3 GB/s: keeping KV host buffers pinned is important.

Artifacts: bw_test.cu / bw_test.exe / bw_test_results.txt, brink_72k.log (VRAM-trace in vram_trace.csv), brim logs for 80k/88k/96k failures.