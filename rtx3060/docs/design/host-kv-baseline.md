# Historical native host-KV baseline

**Correction:** `--no-kv-offload` may schedule attention on the CPU. This experiment
was not a GPU flash-attention comparison, and the synthetic staged harness did not
establish the promised 3.6 -> 17 t/s improvement. See
[the actual implementation](staged-kv-implementation.md).

Run: llama-cli (llama.cpp), IQ3_XXS-mtp, c=49152, -ngl 99, q4_0 K/V, --no-kv-offload
(→ whole KV in pinned host buffer; attention reads it over PCIe on demand, NO staging overlap).

    [ Prompt: 343.9 t/s | Generation: 3.6 t/s ]

Compare:
- Resident q4_0 @32K ......... 19.9 t/s  (llama-bench tg16)
- Staged-host (transport, kv_stage2)  17-19 t/s
- llama.cpp --no-kv-offload @48K .. 3.6 t/s  (un-staged host KV)

Conclusion: llama.cpp's built-in host-KV path is ~5-6x slower than our staged design at big
context because attention stalls on mid-kernel PCIe reads (no GDN-overlap). This quantifies the
win the staged-prefetch integration must recover (3.6 -> ~17+ t/s) and is the motivation for
dev/staged-kv-offload-design.md Tier-2.
