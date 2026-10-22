# Historical staged host-KV offload spec

**Superseded:** actual implementation and numerical validation now live in
[`staged-kv-implementation.md`](staged-kv-implementation.md). The text below records
an earlier plan, not current status. Synthetic harness throughput was not an
end-to-end performance result and did not validate its shared-slot synchronization.

The transport-level proof (`kv_stage2.cu`) is done and measured; wiring it into llama.cpp's
decode/attention so `llama-server` actually reads KV from staged VRAM slots is NOT yet done.
This doc is the implementation plan + the honest current state.

## Current state of the llama.cpp patch (`llama_cpp_patch.diff`)
Three orthogonal, inert-by-default edits (already committed to llama.cpp tree):
1. `ggml-cuda.cu`: runtime `GGML_CUDA_PREFER_MMQ` switch — A/B test mmq-vs-mmvq at T=1.
   Result: mmvq wins on 3060 (21.6 vs 14.1 t/s), default unchanged.
2. `src/llama-context.cpp` + `src/llama-kv-cache.cpp/.h`: `LLAMA_DUMP_KV=<dir>` env hook that
   writes the real K/V tensors (via `llama_kv_cache::dump_files`) for compression analysis.
   (Note: with the 2026 memory refactor, K/V materializes only after real decode; the hook
   fires in `llama_kv_cache_context::apply()`.)
3. (nothing else) — **no staged host-KV path in llama.cpp yet.**

## What "complete the implementation" means (the remaining work)
Two realistic tiers:

### Tier 1 — use llama.cpp's native host KV (`--no-kv-offload`)
llama.cpp already supports keeping KV in a pinned host buffer; attention reads it across PCIe
*on demand* (un-staged). This is the honest "all KV in host, hosted on llama-server" baseline.
Expected: regresses vs resident, because attention stalls on PCIe mid-kernel (no overlap with
GDN). Measuring this (see `host_kv_baseline` run in notes) quantifies exactly how much the
staged design recovers — that is the motivation.

### Tier 2 — staged per-layer prefetch (the real design; NOT trivial)
Requires rewiring the attention graph so each attention layer reads its K/V from a device
staging slot that was prefilled (double/triple buffered) from host on a side stream overlapped
with the preceding GDN+MLP compute. Touch points in llama.cpp:
- Where the attention `KQ`/flash nodes reference `layers[i].k/.v`: add a device-side "current
  layer KV view" that points at the staged slot instead of the (host) cache buffer.
- A host→device staging engine (pinned pool + `cudaMemcpyAsync` on a non-default stream +
  events + `cudaStreamWaitEvent` at each layer's attention node, mirroring `kv_stage2.cu`).
- Policy: GPU holds newest ~128K tokens resident for all layers; oldest evicted to host;
  `--kv-offload-staged` runtime flag; derive the resident-window from free VRAM.
- Rollback safety for MTP (drafted tokens extend KV; reject must drop staged slots too).

This is a multi-file, days-of-work change in the (2026) `llama_memory`/graph refactor and is the
single remaining step to make the measured 17 t/s at 128K real in `llama-server`.

## Why the 150K/240K numbers are valid but are NOT llama.cpp end-to-end
They come from `kv_stage2.cu`, which replays real KV buffer sizes/counts over the real PCIe bus
with the staged-overlap engine (compute stream + copy stream + events), using a calibrated
per-layer compute slice equal to this model's measured weight-stream+attention budget. It
measures the *transport correctness* of the design. llama.cpp itself has not run this path yet.

## Measured reference (to beat)
- Resident q4_0 @32K: 19.9 t/s (llama-bench tg16).
- Staged-host (transport) @32K: ~19.0 t/s, +5.4% overhead.
- Evict-first @128K (transport): ~17.1 t/s, +0.6% (all-real-memory, KV in host).
