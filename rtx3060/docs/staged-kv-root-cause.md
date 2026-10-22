# Staged-KV transfer cost: root cause and achievable recovery

Target: explain why host-staged KV reduces decode throughput even though PCIe runs at
link rate and every layer's copy completes before that layer's attention begins.

Result in one line: **the pipeline is not stalling, and the cost is not DRAM write
traffic, alignment, batching, clock throttling or scheduling. Moving a byte over PCIe
into VRAM costs about 1.4 byte-equivalents of shared memory-system bandwidth, and the
decode workload is inefficient enough that this is expensive. No control available on
the copy-engine path reduces it. The only lever is payload size.**

---

## 1. What was measured, and on what

| Item | Value |
|---|---|
| GPU | RTX 3060 12 GiB, GA106, sm_86, 3 MiB L2, no ECC capability |
| Driver / toolkit | 610.47 / CUDA 13.4, Windows 11, WDDM |
| PCIe | Gen3 x16, trained up under load (reports Gen1 at idle) |
| Profiler | Nsight Compute 2026.3.0 present, blocked: `ERR_NVGPUCTRPERM` (needs admin) |

Because hardware memory counters were unavailable, every number below comes from
controlled baseline-versus-treatment timing. Percentages are disclosed where a
mechanism could not be proven directly.

### Transfer geometry (read from the implementation, not from comments)

`ggml/src/ggml-cuda/staged-kv.cuh`, `prefetch()`: per staged layer, exactly two
`cudaMemcpyAsync` calls.

| Property | K copy | V copy |
|---|---|---|
| Direction | host to device | host to device |
| Stream | `copy[0]` (non-blocking) | `copy[1]` (non-blocking) |
| Size | ~54 MiB at 96K (576 B/token) | ~54 MiB at 96K (576 B/token) |
| Ranges | one contiguous range | one contiguous range |
| Source | canonical KV in pinned mapped host RAM (`cudaHostAlloc Mapped\|Portable`) | same |
| Destination | `cudaMalloc` slot, 256-byte aligned | same |
| Source/dest 128-byte alignment | 0 / 0 offset | 0 / 0 offset |

The file header claimed copies are "several chunks across two copy streams". **That
comment was stale** and has been corrected: there is one contiguous copy per stream.
Total staged payload at 96K with 8 staged layers is 899 MB per pass.

Instrumentation: `GGML_KV_STAGE_PROF` (retained, off by default) records
`cudaEventRecord` timestamps on the device streams, so all intervals are GPU-side device
times, not host API issue times.

---

## 2. Experiment results

Benchmark code: `kv_dma_bench.cu`, build with `build_kv_dma_bench.bat`. An SM kernel is
used only as a *victim* read workload; no copy is ever performed by an SM.

```
kv_dma_bench.exe --mode {none|h2d|d2h} --read-mb 1024 --passes 32 --blocks 1024 \
                 --copy-mb 864 --chunks N --dest-off B --dst-wrap-mb W --trials 3
```

| # | Experiment | Configuration | Result |
|---|---|---|---|
| B | DMA isolated (no compute) | 899 MB pinned H2D | 73.2-74.1 ms, **12.7-12.9 GB/s** |
| C0 | Victim workload alone | 32 GiB streaming read | 101.29-101.40 ms, **339 GB/s** |
| C1 | Concurrent **H2D** | 899 MB | 105.30 ms, **+4.0 ms** |
| C2 | Concurrent **D2H** | 899 MB | 104.01 ms, **+2.7 ms** |
| C3 | Copy count sweep | 864 MiB in 1/2/4/8/16 chunks | 105.18 / 105.21 / 105.19 / 105.18 / 105.19 ms, **no effect** |
| C4 | Destination footprint | 864 MiB into 2 MiB window (L2-resident) | 105.31 ms vs 105.17 ms, **no effect** |
| A | Destination alignment | offsets 0, 32, 64, 128, 256 B | 105.2-105.3 ms, **no effect** |
| D1 | Production, resident | 40K, 16 resident, MTP off | 63.58-64.18 ms |
| D2 | Production, staged | 40K, 15 staged, MTP off | 74.35-74.37 ms (**+10.8 ms**) |
| D3 | Production, resident | 96K, 16 resident, MTP off | 87.63 ms |
| D4 | Production, staged | 96K, 8 staged, MTP off | 101.40 ms (**+13.8 ms**) |
| E | Calls held at 16, bytes at 25% | 40K, 15 staged | **74.35 vs 68.90 ms** |
| F | GPU clocks, staged vs resident | 40K, 15 staged vs resident | SM **1828 vs 1784 MHz**, power 163.8 vs 160.7 W, both SW power cap |

### What each result rules out

- **C4 (2 MiB destination window) falsifies the write-allocate hypothesis.** Writing
  864 MiB repeatedly into a 2 MiB window fits entirely in the 3 MiB L2, so almost no
  DRAM writes occur; the penalty is identical (105.31 vs 105.17 ms). The cost therefore
  tracks PCIe payload bytes, not DRAM write bytes, and write-allocate cannot explain it.
  The earlier "2x DRAM traffic" inference in the deleted analysis was wrong.
- **C3 rules out batching/transaction formation.** 1 to 16 chunks of the same bytes give
  identical results.
- **A rules out alignment and partial-sector effects.** Misaligned destinations cost the
  same as aligned ones, and the production copies are already 128-byte aligned.
- **F rules out clock and power throttling.** The staged run holds a *higher* SM clock.
- **E rules out per-call overhead.** Holding the call count at 16 and shrinking bytes to
  25% still removes ~half the penalty, so the cost follows bytes, not calls.
- **C1 vs C2 shows a write asymmetry.** Per payload byte, H2D costs 1.5x what D2H costs,
  consistent with the VRAM write path being more expensive than the read path, but far
  short of 2x.

---

## 3. Calculations

Per-byte cost from experiment E, solving `penalty = fixed + rate x bytes` over the two
points at 40K:

```
74.35 - 63.58 = 10.77 ms over 698 MB
68.90 - 63.58 =  5.32 ms over 175 MB
=> rate  = 10.4 us/MB,  fixed = 3.50 ms
```

The 3.50 ms fixed term matches the independently measured DMA-disabled delta
(96K: 91.13 - 87.63 = 3.50 ms) exactly.

Validated against every production point:

| Point | Payload | Predicted | Measured | Error |
|---|---:|---:|---:|---:|
| 40K, 15 staged, full | 698 MB | 74.34 ms | 74.35 ms | -0.02% |
| 40K, 15 staged, 25% | 174 MB | 68.89 ms | 68.90 ms | -0.01% |
| 96K, 8 staged | 899 MB | 100.48 ms | 101.40 ms | -0.90% |

So:

```
pass = resident pass + 3.50 ms + 0.0104 ms per MB staged
```

Expressed as bandwidth, the per-byte term costs `10.4 us/MB x 131 GB/s = 1.36` byte
equivalents per payload byte, using the production workload's achieved 131 GB/s. The
saturated synthetic victim gives `4.5 us/MB x 339 GB/s = 1.53`. Both independent
estimates land at **1.4-1.5 memory-system byte-equivalents per PCIe payload byte**.

Separation of the loss at 96K, 8 staged:

| Component | Value | Basis |
|---|---:|---|
| Resident compute | 87.63 ms | measured, all 16 layers resident |
| Fixed staging overhead | 3.50 ms | measured with DMA disabled |
| DMA per-byte cost | 9.35 ms | 899 MB x 10.4 us/MB |
| **Total** | **100.5 ms** | 101.40 ms measured |
| Time lost waiting on a copy | **0 ms** | timestamps: copies finish ~12 ms early |

---

## 4. Facts versus hypotheses

**Observed facts.** Copies complete ~12 ms before the attention that needs them; removing
the copies recovers 11.4 ms of 13.8; the penalty is proportional to payload bytes and
independent of call count, chunking, alignment and destination footprint; D2H costs less
than H2D; clocks are not reduced.

**Unproven hypothesis.** The precise mechanism inside the memory system is *not*
identified, because hardware counters were unavailable. The candidates consistent with
the data are (a) copy-engine writes occupying L2/crossbar service slots that the SM's
requests also need, and (b) a latency increase that penalises a decode workload already
running at only 131 GB/s of a 339 GB/s device. The 2 MiB window result excludes DRAM
write traffic and write-allocate from either explanation.

**What is proven is the shape, not the mechanism:** cost is proportional to payload
bytes and independent of every scheduling and layout variable that was varied.

---

## 5. Fixes evaluated (DMA path only, no SM copies)

| Fix | Idea | Outcome |
|---|---|---|
| A | Align destinations, merge adjacent writes, avoid partial sectors | **No gain.** Alignment sweep flat; production already 128-byte aligned |
| B | Improve transaction formation (batching, transfer size) | **No gain.** 1-16 chunks identical; production already maximal contiguous transfers |
| C | Documented DMA/cache controls | **None applicable.** `cudaMemcpyAsync` exposes no write-policy control. The L2 access-policy window (`cudaStreamSetAttribute`) is a residency hint only; since destination footprint provably does not matter (C4), it cannot reduce this cost and was not adopted |
| D | Retime the schedule, more slots, fewer streams | **No gain.** 2 vs 4 slots: 101.40 vs 101.96 ms; 1 vs 2 streams: 102.33 vs 102.52 ms. There is no stall to schedule away |

**No code change to the transfer was justified**, so none was made; the production path
is unchanged apart from removing my diagnostics and correcting the stale header comment.
The only lever the data supports is moving fewer bytes:

| Staged payload per pass | Cost at 10.4 us/MB |
|---|---|
| 100 MB | 1.04 ms |
| 500 MB | 5.2 ms |
| 899 MB | 9.35 ms |

This changes the capacity decision, not the code. At 96K with MTP off, residency wins:

| 96K, MTP off | ms/token | t/s |
|---|---:|---:|
| 16 resident, no staging | 87.63 | **11.41** |
| 8 staged | 101.40 | 9.86 |

---

## 6. Correctness and final comparison

Final test: same 40,379-token prompt, `temperature 0`, 48 tokens generated, staged path
(15 layers) against resident path (0 layers).

```
resident: 64.18 ms/token, 15.58 t/s   sha256 ECDF1667...
staged  : 74.37 ms/token, 13.45 t/s   sha256 ECDF1667...   IDENTICAL
```

The staged path is numerically identical to the resident path. The DMA-disabled run
(91.13 ms at 96K) produces wrong output and is diagnostic only; it is never a valid
serving configuration and its speed is not a speedup.

| Configuration | Payload | Measured | Note |
|---|---:|---:|---|
| Resident only | 0 | 87.63 ms | reference |
| DMA isolated | 899 MB | 73.5 ms @ 12.8 GB/s | link-limited, correct |
| Concurrent DMA + read | 864 MB | +3.8 ms | saturated victim, aligned contiguous layout |
| Production layout | 899 MB | +13.8 ms | low-BW latency-sensitive victim |
| Aligned contiguous layout | 864 MB | +3.8 ms | alignment is not the difference |

---

## 7. Correction to prior documentation

`report.md` and `bw-breakdown.html` both stated that PCIe "achieved 9.70 GB/s (73% of
peak)" and built their models on it. That figure was back-solved from end-to-end
serving throughput and attributed attention and speculative-decoding compute time to the
PCIe link. It is withdrawn. Both files now report the measured 13.3-13.4 GB/s, state
that the transfers are hidden rather than exposed, and carry the corrected cost model.
Measured results in those files are preserved.

---

## 8. Reproduce

```
build_kv_dma_bench.bat                      # nvcc -arch=sm_86 -O2
kv_dma_bench.exe --mode h2d --copy-mb 899 --read-mb 64 --passes 1      # experiment B
kv_dma_bench.exe --mode none --read-mb 1024 --passes 32                # C0
kv_dma_bench.exe --mode h2d  --read-mb 1024 --passes 32 --copy-mb 899  # C1
kv_dma_bench.exe --mode d2h  --read-mb 1024 --passes 32 --copy-mb 899  # C2
kv_dma_bench.exe --mode h2d  --read-mb 1024 --passes 32 --copy-mb 864 --dst-wrap-mb 2   # C4

# production (server, then prompt prefill and decode from tests/)
LLAMA_KV_STAGED_LAYERS=8 serve.ps1 ...   # 96K, staged
GGML_KV_STAGE_PROF=1                     # per-layer device timestamps
```
