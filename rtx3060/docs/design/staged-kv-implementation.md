# Staged host KV: actual implementation (2026-10-17)

## Status / evidence standard

This supersedes the earlier **pending** spec. The CUDA backend now implements real
host-backed KV consumed by real GPU flash attention in llama-server. Measured results
will be recorded in `results/` and summarized here after the runs complete.

Earlier `kv_stage*.cu` numbers are synthetic transport microbenchmarks, **not** Qwen
inference measurements. Their shared-slot copies also lacked a consumer-completion
fence, so those numbers cannot validate correctness or promise an inference speedup.
The new implementation has that fence and is numerically checked against resident KV.

## Design

- `LLAMA_KV_STAGED_LAYERS=N`: choose N trunk attention layers, spaced evenly across the
  network. Remaining layers retain ordinary device KV. Zero/unset changes nothing.
- Alternatively `LLAMA_KV_GPU_BUDGET_MIB=M`: choose the smallest staged-layer count whose
  **resident KV + both GPU staging slots** fit M MiB at the configured context capacity.
  This is a **KV-only budget**, not total process VRAM. Leave space for weights, GDN,
  workspaces, CUDA runtime and MTP. At least two layers remain resident in budget mode.
- Placement is chosen at **context initialization**, not online token eviction. DMA
  grows dynamically with the used cache prefix (including llama.cpp's padding), not
  with the maximum context allocation. This is whole-layer placement, not token-tier
  attention, automatic free-memory probing or a rolling GPU token window.
- NextN/MTP's separate, single-layer KV remains resident. It does need KV. Drafting
  executes that head, **not a second full backbone**. Backbone verification is a
  multi-token pass; host traffic amortization is workload/acceptance dependent.

### CUDA data path

1. A dedicated buffer type owns mapped, **pinned** canonical host storage. The scheduler
   keeps its SET_ROWS and FLASH_ATTN on CUDA, rather than falling back to CPU attention.
2. Scan the current CUDA graph for offloaded flash-attention consumers. Allocate two
   stable K+V device slots outside CUDA capture, each with one layer's full capacity.
3. Fork a **nonblocking copy stream** from the compute stream. Prefetch the first two
   offloaded layers' used prefixes, asynchronously.
4. Before SET_ROWS, wait for that layer's slot-ready event. Use the **existing CUDA
   quantization kernel** to write both canonical host rows and the staged device rows.
   Only newly computed rows are written; the full history is not written back.
5. Flash attention receives stack-copy tensor descriptors pointing at the staged
   device slot, with exactly the original type, strides, mask and positions.
6. After ALL attention kernels complete, record a consumed event. The copy stream
   must wait for that event before reusing the slot for layer i+2. This is essential:
   a ready event alone does not protect a slot from being overwritten during reads.
7. Join the copy stream before ending capture or returning from graph submission.
   All the copies, waits, writes and attention are captured and replayed together.

The existing cache owns positions, masks and sequence operations. Rejected spec rows
can remain physically present but are masked/replaced under the existing metadata;
there is no separate eviction/rollback position ledger. Canonical host data makes
state serialization, clearing, restoration and prefix reuse work through buffer IO.

### Files

- `ggml/src/ggml-cuda/staged-kv.cuh`: pinned storage, two slots, transfer/consumer events,
  graph plan and SET_ROWS/flash interception.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: buffer support, registry extension, capture integration,
  fusion exclusion only where needed for staged writes.
- `ggml/src/ggml-cuda/common.cuh`: backend-owned staging lifetime.
- `src/llama-kv-cache.cpp`: opt-in layer placement / KV-only budget.

No new approximation or custom attention math. Ordinary CUDA matmul/GDN/attention
kernels remain in use. This is experimental local functionality, not upstream support.

## Restrictions

- One CUDA device, one server slot (`-np 1`), flash attention (`-fa on`).
- Non-transposed K/V, non-MLA, non-SWA, no cross-layer cache sharing/reuse.
- Supported CUDA flash-attention head dimensions / dtype pairs are required. The old
  stories15M model has a 48-wide head unsupported by this build's CUDA FA, so it is
  not a suitable GPU flash-attention test. Use SmolLM2-135M (64-wide) instead.
- Selected pinned backing remains full-context capacity; physical pinning can fail if
  RAM/OS limits are too small. No pageable fallback, which would destroy overlap.
- Tensor parallelism, multi-GPU, concurrent slots, automatic online migration and
  token-wise spill are not implemented. Unsupported configurations fail explicitly.

## Numerical checks

`tests/staged_kv_check.cpp`, linked against the built DLLs, runs identical real model
forwards in a resident context and a staged context sharing the loaded model.

Initial checks completed:

- SmolLM2-135M-Instruct Q4_0 weights: FP16 KV, graphs disabled: **all logits exact**.
- Same, graphs enabled: **all logits exact**.
- Same, Q5_0 K / Q4_0 V: **all logits exact**.
- Each includes multi-ubatch prefill, 32 decode steps, canonical state save/clear/restore,
  four-token verification, and suffix removal/overwrite.
- Target Qwen IQ3_XXS-mtp weights, Q5_0 K/Q4_0 V, 4K test: prefill, 32 decode steps and
  state restore + verify4 **all logits exact** (248,320 logits per comparison).
  Ordinary suffix removal is refused by BOTH resident and staged GDN models, as
  expected; actual MTP uses its own recurrent-state checkpoint/rollback path.

SmolLM2 fixture downloaded from
`bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_0.gguf`.
The test book is a contiguous excerpt of Project Gutenberg's *Pride and Prejudice*,
not random or repeated token IDs. `tests/run_staged_server.py` saves invocation and
real HTTP completion timings, with prompt token counts.

## Usage (PowerShell)

```powershell
$env:LLAMA_KV_STAGED_LAYERS = '8'
.\llama_cpp_src\build-cuda\bin\llama-server.exe `
  -m .\Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf `
  -c 131072 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 `
  -np 1 -b 512 -ub 128 --fit off --no-context-shift `
  --host 127.0.0.1 --port 8080
```

Enable MTP with `--spec-type draft-mtp --spec-draft-n-max 3`. Its extra weights,
head KV, recurrent checkpoints and workspaces need additional VRAM, so more trunk
layers may need staging. Do not infer the configuration fits from weights alone.

Budget mode, instead of an explicit layer count:

```powershell
Remove-Item Env:LLAMA_KV_STAGED_LAYERS -ErrorAction SilentlyContinue
$env:LLAMA_KV_GPU_BUDGET_MIB = '1440'
# Same server command: choose placement for resident trunk KV + two staging slots.
```

Do not enable `LLAMA_DUMP_KV` during timing. Trace diagnostics can be enabled with
`LLAMA_KV_STAGE_TRACE=1`; they print scheduling operations during eager execution or
capture, not every CUDA graph replay.

## Corrections to previous notes

- Q4_0 is **4.5 bits/element** (18 bytes/32), not ~8 bits. Q5_0 is 5.5 bits.
- 128K q4/q4 trunk KV = **2.25 GiB**; 128K q5/q4 = **2.50 GiB**.
- True FP8 is distinct from llama.cpp Q8_0 (8.5 bits/element).
- `graphs reused` in the server timing counts reuse of the **ggml graph**, not proof
  of CUDA graph replay. CUDA API capture/launch evidence or CUDA graph debug is needed.
- Nsight tracing of CUDA Graphs must expand graph nodes before kernel-time summations.
  Previous estimates of 30–60% host-idle time were not established reliably.
- `--no-kv-offload` can cause **CPU attention**, not on-demand GPU mapped-memory
  attention. Earlier descriptions of why that baseline was slow were unverified.
- Low measured logical GB/s is not a decomposition of time into ALU vs DRAM. Kernel
  counters/roofline experiments are needed to establish the actual bottleneck.
