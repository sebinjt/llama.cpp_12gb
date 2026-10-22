# Qwen3.8-27B on a 12 GiB RTX 3060: Staged Host KV Cache

Living log of what was tried, what works, what fails and why. Every number is measured on
this machine unless marked **modelled**. Sections 0, 4, 5, 8 and 10 are the ones to read
first.

New offline prefill/cache producer: see section 17 and `dev/group-prefill.md`.
The old `kv-200k-q4_0-L14.bin` snapshot is suspect. Use the validated section 17 cache
for new experiments; earlier 200K acceptance claims are not a reliable baseline.

---

## 0. TL;DR

**Current validated PP/cache path:** offline two-group prefill, 265.17 tok/s before
save, 12m57s including save/shutdown, then a 2.4-second restore with one prompt token
left to process. See section 17. The older decode tables below are historical and
their 200K acceptance/correctness claims used a suspect snapshot.

The pipeline is now **chain-bound, not PCIe-bound**: at 8 staged layers the chain is
~156 ms against 142 ms of DMA, so staging further down buys nothing (section 14.4).
The chain is 47.4 ms of weights, 84.4 ms of attention and 24 ms of the CPU output layer.
**Attention is now the only lever that matters** (section 14.5).

| Use case | Context | Command |
|---|---|---|
| 200K, fastest decode | 204,800 | `.\serve.ps1 -Ctx 204800 -StagedLayers 8 -VecQuantKv -Mtp 0` plus `-ot output=CPU` |
| 200K, from a saved snapshot | 204,800 | same, then `POST /slots/0?action=restore` |
| 96K, general use | 98,304 | `.\serve.ps1` (defaults: 12 staged, MTP on) |

Headline numbers measured to date:

| Context | Config | Decode | Prefill |
|---|---|---:|---:|
| 200K | **8 staged, 8 resident, output on CPU** | **5.77 t/s** (170.6 ms/tok) | 219.3 t/s |
| 200K | 9 staged, 7 resident, vector kernel | 5.63 t/s (174.8 ms/tok) | 219.3 t/s |
| 200K | 14 staged, 2 resident, MMA + MTP | 3.63 t/s (271.0 ms/tok) | not measurable |
| 200K | 14 staged, 2 resident (original default) | 3.79 t/s (259.5 ms/tok) | 219.3 t/s |
| 96K | 10 staged, 6 resident | 11.52 t/s | - |
| 96K | 12 staged, 4 resident (shipped default) | 11.31 t/s (88.5 ms/tok) | - |
| 96K | 0 staged, all resident | 11.41 t/s (87.63 ms/tok) | - |
| 40K | 0 staged, all resident | 15.5 t/s (63.58 ms/tok) | - |

---

## 1. Objective, hardware, model

Run a 27B model on a single 12 GiB card at long context by keeping the KV cache in host
RAM and streaming only what each layer needs, fast enough to be usable. The bar to beat is
llama.cpp's standard layer offload to CPU (`-ngl 48`), which gives 0.97 t/s at 96K.

**Hardware**

| Component | Detail |
|---|---|
| GPU | RTX 3060 12 GiB, CC 8.6 (GA106), 3 MiB L2, no ECC capability |
| Bus | PCIe Gen3 x16, pinned H2D 13.4 GB/s (64 MiB), 13.3 GB/s (256 MiB - 1 GiB) |
| CPU | Ryzen 5 4600G, L3 8 MiB |
| RAM | 32 GB, single DIMM at 2400 MT/s |
| Software | driver 610.47, CUDA 13.4, Windows 11 WDDM, `NO_VMM = 1` |

**Model**: `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`

- 64 trunk layers, of which 16 carry KV (attention) at `blk.3, 7, 11 ... 63`
- 48 GDN (recurrent) layers
- MTP draft head = `blk.64`
- Weights 9,948.6 MiB with MTP, **9,227.87 MiB without**

**KV geometry**: 1024 values per token per layer, each for K and V. At q4_0 that is
0.5625 B/value = **1,152 B/token/layer**, 18,432 B/token across all 16 layers.

| Context | Per layer | All 16 layers |
|---|---:|---:|
| 40K | 45 MiB | 720 MiB |
| 96K | 108 MiB | 1,728 MiB |
| 128K | 144 MiB | 2,304 MiB |
| 200K | **225 MiB** | 3,600 MiB |

---

## 2. Design

Canonical KV lives in pinned, mapped host RAM. Two device staging slots; two non-blocking
copy streams with persistent `consumed` / `ready` events move only the prefix that is new
since the last pass. The decoder never uses CUDA graphs. Staged layers are selected
positionally (evenly spaced), not by heat:

```cpp
if ((i + 1) * count / cache_layers != i * count / cache_layers)   // llama-kv-cache.cpp:213
```

The transfer is fully overlapped, not exposed. Per-layer GPU timestamps show every copy
finishing about 12 ms before that layer's attention starts, so attention never waits on
its own copy. The cost is memory-system bandwidth consumed while DMA and compute run
together.

**Correction.** An earlier version claimed the pass was governed by
`max(weights / VRAM bandwidth, staged bytes / 9.70 GB/s)` and that 9.70 GB/s predicted
10.55 t/s exactly. That rate was back-solved from end-to-end throughput, not measured, and
it silently attributed attention and speculative-decoding compute to the PCIe link. The
measured link rate is 13.3-13.4 GB/s, saturated in isolation. Keep the four quantities
apart:

| Quantity | Value | How measured |
|---|---|---|
| Link bandwidth | 13.3-13.4 GB/s | `bw_test.exe`, pinned async H2D, 64 MiB - 1 GiB |
| Payload bandwidth in service | 12.5-13.4 GB/s | per-layer copy timestamps, 54 MiB contiguous copies |
| Extra memory cost per payload byte | ~1.4 byte-equivalents | see `staged-kv-root-cause.md` |
| End-to-end throughput | 3.63-15.7 t/s | depends on context, residency, staging |

---

## 3. Results in detail

### 3.1 200K decode sweep (current focus)

204,800 ctx, q4_0 K and V, **no MTP**, `LLAMA_FA_VEC_QUANT_KV=1`, KV restored from the
saved snapshot so no prefill was paid per point.

| Staged | Resident | Free VRAM @load | Decode | ms/token | DMA @13.3 GB/s | DMA util |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 2 | 1,639 MiB | 3.79 t/s | 259.5 | 248 ms | 96% |
| 13 | 3 | 1,413 MiB | 3.99 t/s | 246.8 | 231 ms | 94% |
| 12 | 4 | 1,189 MiB | 4.28 t/s | 230.0 | 213 ms | 93% |
| 11 | 5 | 963 MiB | 4.65 t/s | 211.7 | 195 ms | 92% |
| 10 | 6 | 739 MiB | 5.07 t/s | 194.3 | 177 ms | 91% |
| **9** | **7** | **513 MiB** | **5.58-5.63 t/s** | **174.8-176.4** | **160 ms** | **91%** |
| 8 | 8 | 33 MiB | - | - | 142 ms | fails to load, slots do not fit |
| 8 | 8, 1 slot | 517 MiB | 4.63 t/s | 212.4 | 142 ms | loads, but slower (see 3.1.1) |
| 9 | 7, 1 slot | 741 MiB | 4.14 t/s | 237.8 | 160 ms | control for the serialisation cost |

#### 3.1.1 Single staging slot: measured and rejected

`LLAMA_KV_STAGE_SLOTS=1` is now allowed (default remains 2). It frees 225 MiB, which is
exactly what L=8 needed, and L=8 does load with it. It is still slower:

| Config | Free VRAM | Decode | Pass |
|---|---:|---:|---:|
| **9 staged, 2 slots** | 513 MiB | **5.61 t/s** | **175.4 ms** |
| 9 staged, 1 slot | 741 MiB | 4.14 t/s | 237.8 ms |
| 8 staged, 1 slot | 517 MiB | 4.63 t/s | 212.4 ms |

With one slot, `prefetch(i+1)` waits on `consumed[0]`, which is recorded after attention for
layer i (`staged-kv.cuh:393-396`), so the copy can no longer hide behind compute. At 200K
the copy (160 ms) and compute (132 ms) are comparable, so serialising them costs about
62 ms per pass (175.4 -> 237.8 ms). Dropping one staged layer saves only 17.7 ms of DMA.
Net effect: -0.98 t/s.

The prediction in the previous revision of this document ("a few ms per pass") was wrong:
it assumed only a small gap would be exposed, not the whole copy.

Payload per token is `staged x 225 MiB`, so one staged layer is 17.7 ms of link time.
At 200K the pass is DMA-bound and removing one staged layer is worth 0.2 t/s at L=14
rising to 0.5 t/s at L=9, because the same 18 ms is a larger share of a shorter pass.

**L=8 is a hard wall**, not a tuning problem:

```
staged KV: 2 slots x 225.00 MiB = 450.00 MiB; 33.00 MiB dedicated VRAM free
staged-kv.cuh:259: staged KV staging slots do not fit in VRAM
```

**Correctness**: L=9 with the vector kernel and L=14 with the MMA kernel produce
byte-identical output on a 204,365-token context (temperature 0), sha256 `F38C5BCB...`.

### 3.2 200K prefill

| Metric | Value |
|---|---|
| Tokens | 204,365 |
| Time | 931.8 s |
| Throughput | **219.3 t/s** |
| Kernel | MMA/f16 (default path) |

Prefill at 200K moves the whole prefix once per 128-token ubatch, so it is dominated by
per-pass cost. A vector-kernel prefill would be about 3.8x slower (see 3.4), i.e. roughly
60 minutes, which is why the snapshot exists.

### 3.3 MTP at 200K: a net loss

| Config | Decode | Pass | Acceptance |
|---|---:|---:|---|
| 13 staged, MTP n_max=2 | 3.64 t/s | 270.2 ms | 0.4058, mean len 1.80 |
| 14 staged, MTP n_max=2 | 3.63 t/s | 271.0 ms | 0.4058, mean len 1.80 |
| 14 staged, no MTP | 3.79 t/s | 259.5 ms | - |

Acceptance is healthy (1.80 tokens per pass), but each pass costs 271 ms instead of
259.5 ms, because the two draft forwards each pay full 200K attention. The gain is eaten
exactly. MTP is therefore off for 200K.

MTP also does not fit below L=13 at 200K, even with the vector kernel:

| Config | Failure |
|---|---|
| L=9 | `cudaMalloc failed` on 87.39 MiB |
| L=10, L=11 | staging slots do not fit |
| L=12 | `cudaMalloc failed` on 225.00 MiB (second slot) |

Draft overhead is roughly 750-950 MiB at 200K once the f16 scratch is removed (it was
934 MiB before). `--spec-draft-ngl 0` has **no effect** for `draft-mtp` (705 MiB free at
200K either way), and there is no `--spec-draft-n-ctx` to shrink the draft context, which
inherits the full 204,800.

### 3.4 Prefill kernel A/B (32K, L=14)

| Path | Prefill | Free VRAM after prefill |
|---|---:|---:|
| MMA/f16 (default) | 373.6 t/s | 1,107 MiB |
| vector, no f16 scratch | 98.1 t/s | **2,233 MiB** |
| vector, ub=32 | 83.0 t/s | 2,271 MiB |

Decode is unaffected (14.59 vs 14.48 t/s) because decode already used the vector kernel.
The saving is real; the prefill cost is structural, see 5.4.

### 3.5 VRAM budget at 200K

Measured from the server's own buffer accounting (`--log-verbosity 4`), L=9, MTP off,
`-b 512 -ub 128`:

| Component | MiB | Share |
|---|---:|---:|
| Weights (`CUDA0 model buffer`) | 9,227.87 | 75.1% |
| Resident KV, 7 layers x 225 | 1,575.00 | 12.8% |
| Free (`nvidia-smi`) | 513.00 | 4.2% |
| Staging slots, 2 x 225 | 450.00 | 3.7% |
| CUDA context / driver residual | 286.24 | 2.3% |
| Recurrent state, 48 GDN layers (`RS buffer`) | 149.62 | 1.2% |
| Compute buffer | 85.77 | 0.7% |
| **Total** | **12,287.50** | 100% |

Host side, not VRAM: staged KV 2,025.00 MiB pinned, `CPU_Mapped` model part 388.38 MiB,
host compute buffer 60.27 MiB, host output buffer 0.95 MiB.

Turning MTP off already saves 720.73 MiB of weights (9,948.60 -> 9,227.87), which is 3.2
KV layers. That is the largest single VRAM saving made so far.

**Reducible, and what it buys**

| Component | Floor | Saving | Verdict |
|---|---:|---:|---|
| Compute buffer | 15.51 MiB at `-b 16 -ub 16` | 70.26 MiB | does not cross the layer boundary, kills prefill |
| RS buffer | 0 if moved to host | 149.62 MiB | costs 149.62 MiB/token = 11.25 ms per token on the link |
| Staging slots | 225 MiB (1 slot) | 225.00 MiB | measured at -0.98 t/s, see 3.1.1 |
| Weights | quant-dependent | 3,000 MiB possible | only real lever, see 9 |

**The binding constraint is granularity.** One more resident layer needs exactly 225 MiB,
and the total of every other reducible item is 220 MiB (compute 70 + RS 150), which is
still short. Even at the compute floor, L=8 leaves 61 MiB free against the 450 MiB of
slots it needs, so L=8 is not reachable by tuning.

**Ceiling at 200K**: with all 16 KV layers resident, decode is compute-bound at
`47.43 + 0.4119 x 204.8 = 131.8 ms/token`, i.e. **7.59 t/s**, whatever is done to staging.
L=9 delivers 5.61 t/s, which is 74% of that ceiling, and 91% of the 160 ms the link
requires for its 2,025 MiB/token payload.

### 3.6 Note on the reported free VRAM

`cudaMemGetInfo` free and `nvidia-smi` free disagree by 255-300 MiB under WDDM, and the
`cudaMemGetInfo` value does not move proportionally when buffers shrink: cutting the
compute buffer from 85.77 to 15.51 MiB raised it only from 33 to 61 MiB. Slot allocation
also succeeded at L=9 while it reported 215 MiB free for a 450 MiB request, so the driver
overcommits to a degree. Never size a configuration from the reported free value; use the
buffer accounting above.

### 3.7 96K (earlier work, shipped defaults)

| Metric | Before fixes | Now |
|---|---:|---:|
| 96K decode | 10.55 t/s (94.8 ms) | **11.31 t/s (88.5 ms)** |
| vs stock `-ngl 48` | 0.97 t/s | 11.31 t/s (11.7x) |
| Free VRAM after a 96K request | ~273 MiB | 547 MiB |
| Target compute buffer | 433.27 MiB | 61.39 MiB |
| MTP draft (KV + compute) | 536.27 MiB | 110.83 MiB |

Staged-layer sweep at 96K:

| Staged | Resident | Decode | Note |
|---:|---:|---:|---|
| 10 | 6 | 11.52 t/s | only 35 MiB free during a 96K prefill |
| **12** | **4** | **11.31 t/s** | **default: 251 MiB free during prefill** |
| 14 | 2 | 10.79 t/s | |
| 14 (f16 draft) | 2 | 11.36 t/s | draft KV costs 276 MiB more |

At 96K, and with q4_0 KV, every staged megabyte costs 10.4 us of pass time, so residency
beats staging where it fits. 96K resident (0 staged) is 11.41 t/s against 9.86 t/s with 8
staged layers.

### 3.8 Short context (no MTP, no staging)

| Context | Decode |
|---|---:|
| 40,379 | 15.5-15.7 t/s (63.58-64.18 ms) |
| 52,671 | 14.47 t/s (69.13 ms) |
| 60,931 | 14.00 t/s (71.45 ms) |
| 97,581 | 11.41 t/s (87.63 ms) |

---

## 4. What works

| Item | Effect | Evidence |
|---|---|---|
| Staged host KV, 2 slots, 2 streams | 11.3 t/s at 96K vs 0.97 t/s stock | section 3.5 |
| FA scratch moved to the backend pool | freed 372 MiB compute + 425 MiB draft | section 7 |
| Residency over staging where it fits | 11.41 vs 9.86 t/s at 96K | section 3.5 |
| **Vector kernel for quantized KV** (`LLAMA_FA_VEC_QUANT_KV=1`) | **+49% decode at 200K, enables L=9** | 3.1, 3.4 |
| Deeper staging at 200K | 5.63 t/s at L=9 vs 3.79 at L=14 | 3.1 |
| KV snapshot to disk | 4,047 MiB save in 5.4 s, restore ~5 s | section 8 |
| Snapshot restore across staging depths | restores at L=9..14 from one file | section 8 |
| MTP at 96K | part of the 11.31 t/s default | 3.5 |

---

## 5. What fails, and why

| Attempt | Symptom | Root cause | Verdict |
|---|---|---|---|
| L=8 at 200K, 2 slots | slots do not fit | 450 MiB of slots, 33 MiB free | impossible without a slot change |
| L=8 at 200K, 1 slot | loads, but 4.63 t/s | copy no longer hides behind compute | rejected, see 3.1.1 |
| L=9..12 at 200K, default kernel | OOM in prefill | f16 scratch 838 MiB + >1,189 MiB needed | fixed by vector kernel |
| The 4-token remainder hitting MMA | OOM at L=12 despite 1,189 MiB free | `n_q=4 > 2` routes to MMA, which allocates the full 838 MiB f16 K/V to process 4 tokens | vector kernel fixes it |
| MTP at 200K | 3.63-3.64 t/s | draft forwards each pay 200K attention | net loss |
| MTP below L=13 at 200K | load fails | draft ~764 MiB + 450 MiB slots | does not fit |
| `--spec-draft-ngl 0` | no effect | ignored for `draft-mtp` | dead end |
| MTP prefill after a slot rewind (128K) | CUDA error at ~30K tokens | draft context not rewound with the target (`memory_seq_rm`), suspected desync | open bug, unproven |
| `LLAMA_KV_STAGE_NODMA` | garbage output | slots never initialised | removed from the tree |
| Batching / alignment / chunk / slot-count fixes | 0% | cost is proportional to payload bytes | see `staged-kv-root-cause.md` |
| Write-allocate, DRAM-write, throttling hypotheses | 0% | falsified by controlled experiment | `staged-kv-root-cause.md` |

### Why the vector kernel costs 3.8x on prefill

`cols_per_block` is 1 when `n_q == 1` and **2 otherwise** (`fattn-vec.cuh:552-564`), so a
128-token ubatch spawns 64 blocks per head, each sweeping the entire KV with no tensor
cores. Smaller ubatches make it worse (83.0 t/s at ub=32), so it is not tunable.

### Why no cheaper kernel could be used

| Kernel | Needs f16 K/V | Site |
|---|---|---|
| MMA | yes, `launch_fattn(..., true, true, ...)` | `fattn-mma-f16.cuh:2104` |
| tile | yes, `launch_fattn(..., true, true, ...)` | `fattn-tile.cuh:1182` |
| vector | `need_f16_K = type_K == GGML_TYPE_F16` | `fattn-vec.cuh:538` |

`flash_attn_ext_f16` and `flash_attn_tile` carry no K/V type parameters, so they are
f16-only at runtime. There is a compiled `(Q4_0, Q4_0)` vector instance for head sizes
64/128/256 (`fattn.cu:445`), which is what makes the opt-in switch possible.

---

## 6. Cost model

Fitted and validated in `staged-kv-root-cause.md`. Holds while DMA is far from
saturation:

```
pass = resident pass + 3.50 ms + 0.0104 ms per MB of staged payload
```

At long context the model breaks down and the pass becomes transfer-bound:

```
pass ~= payload / 13.3 GB/s / 0.94
```

Both forms agree with measurement to within 1% at 96K, and the transfer form predicts
263 ms for L=14 at 200K against 263.5 ms measured, and 160 ms for L=9 against 174.8 ms.

Compute-only cost is linear in context, from the 52.7K and 97.6K points:

```
compute(ctx) = 47.43 ms + 0.4119 ms per 1K tokens      (131.8 ms at 204.8K)
```

---

## 7. The flash-attention scratch: two rounds

### 7.1 Round one (96K era): the 387 MiB per-layer reservation

`ggml_backend_cuda_buffer_type_get_alloc_size` special-cased `GGML_OP_FLASH_ATTN_EXT`,
charging the scheduler for a full-precision copy of K and V after the output tensor:

```
ggml_cuda_flash_attn_ext_get_alloc_size() -> ggml_nbytes(dst) + n_kv*1024*2 (K) + n_kv*1024*2 (V)
```

At 96K that is 3 MiB + 192 MiB + 192 MiB = 387 MiB per attention layer. Those copies are
only needed by the MMA/TILE prefill kernel, but the scheduler reserves for the worst-case,
prefill-shaped graph, so 387 MiB sat in VRAM permanently.

Fix: the copies are pure op temporaries, so they moved to the backend pool
(`fattn-common.cuh`), and `get_alloc_size` now returns the plain tensor size. Net diff over
four files was +17/-92 lines. Target compute buffer 433.27 -> 61.39 MiB; draft 428.27 ->
2.83 MiB.

```cpp
ggml_cuda_pool_alloc<half> K_f16_alloc(pool);
ggml_cuda_pool_alloc<half> V_f16_alloc(pool);
half * K_f16 = need_f16_K && K->type != GGML_TYPE_F16 ? K_f16_alloc.alloc(ggml_nelements(K)) : nullptr;
```

### 7.2 Round two (200K era): the scratch is still real at prefill time

After round one the buffer is a pool temporary rather than a permanent reservation, but it
is still allocated at prefill and at 200K it is **C x 4096 B = 838 MiB** (K and V in f16
for the whole prefix). With `NO_VMM = 1` the pool keeps freed blocks, so it also shows up
after a prefill.

This is what made L=9..12 impossible. The fix is the opt-in vector kernel switch in
`fattn.cu`:

```cpp
} else {
    // the MMA and tile kernels copy the whole KV to f16, the vector kernel dequantizes it on the fly
    static const bool vec_quant_kv = getenv("LLAMA_FA_VEC_QUANT_KV") != nullptr;
    if (vec_quant_kv) {
        return BEST_FATTN_KERNEL_VEC;
    }
```

Default off, so stock behaviour is unchanged. Set by `serve.ps1 -VecQuantKv`.

**The real fix, still open**: teach the MMA kernel to dequantize tiles into shared memory
so the full-context f16 buffer never exists. That would give the ~1.1 GiB VRAM saving with
MMA prefill speed instead of 3.8x slower.

### 7.3 The CUDA context residual (unchanged)

The ~820 MiB labelled "CUDA context + allocator overhead" is real and is not ours:

1. Disabling CUDA graphs changed nothing: the staging banner reported exactly 809.00 MiB
   free either way.
2. In the same machine state, `nvidia-smi` reports 12,115 MiB free while a bare
   `cudaMemGetInfo` reports 11,255 MiB. The 860 MiB gap is the WDDM CUDA context, booked
   outside nvidia-smi's process accounting and allocated before our first byte.

It is why a "12 GiB card" yields roughly 11.4 GiB of budget.

---

## 8. KV snapshot cache (disk)

Snapshots make repeat experiments cost seconds instead of a prefill.

| Item | Detail |
|---|---|
| Save | `POST /slots/0?action=save`, body `{"filename": "x.bin"}` |
| Restore | `POST /slots/0?action=restore`, same body shape |
| Requirement | server started with `--slot-save-path` |
| 200K file | `results\.kv_cache\kv-200k-q4_0-L14.bin`, 4,047 MiB, 204,365 tokens |
| Save time | 5.4 s (4,243,261,512 bytes) |
| Restore time | ~5 s |
| Staging-depth agnostic | yes, restores at L=9..14 |
| Prompt binding | **exact token sequence only** (`big.prompt.txt`, sha256 `4607209938ADAF9D`) |

Two bugs were found and fixed on the way:

1. `serve.ps1` built the slot path with `Join-Path` and no trailing separator, while the
   server does `slot_save_path + filename` as plain concatenation, producing
   `...\.kv_cachekv-....bin`. Fixed.
2. The filename is a **JSON body field**, not a query parameter. A query parameter returns
   HTTP 400 with an empty body.

---

## 9. Leverage, ranked

### 9.1 Closed (tested, no gain)

| Idea | Result | Why |
|---|---|---|
| One staging slot to reach L=8 | **-0.98 t/s** | loads at L=8 (4.63) but serialises copy and attention; see 3.1.1 |
| Smaller batch to fit L=8 | no effect | compute buffer 85.77 -> 15.51 MiB moves `cudaMemGetInfo` free only 33 -> 61 MiB; L=8 needs 450; see 3.5 |
| MTP at 200K | -0.16 t/s | draft forwards each pay 200K attention; see 3.3 |
| `--spec-draft-ngl 0` | no effect | ignored for `draft-mtp` |
| KV quant below q4_0 in the staging path | not possible | no 3-bit or lower K/V type reaches a flash-attention kernel |
| Weight streaming, 340 -> 360 GB/s | ~0% | already hidden behind compute |

### 9.2 Open (ranked by payoff)

| # | Idea | Expected at 200K | Cost | Effort |
|---|---|---|---|---|
| 1 | ~~Smaller weight quant~~ **Rejected on quality grounds.** It was the only lever that changed the staging arithmetic by whole layers, and would have reached the 7.59 t/s ceiling | n/a | model quality | ruled out |
| 2 | **CPU-assisted attention, chunk split** (section 13). Uses the ~7 GB/s of spare host DRAM; measured feasible, CPU kernel is memory-bound at 20 GB/s | ~6.3 t/s (+12%), ceiling 8.3 | needs a CPU q4_0 kernel plus per-layer sync | High |
| 3 | **Bounded f16 scratch inside MMA** (tile-wise dequant). Keeps 219 t/s prefill and returns the vector kernel's VRAM. Worthless alone at L=9 because PCIe still binds; only pays combined with #2 | no gain alone | none | High |
| 4 | Close the 15 ms pipeline gap: pass is 174.8 ms against 160 ms of link time | up to +0.5 t/s | needs more slots, so VRAM | High |
| 5 | Fix the MTP draft desync on slot rewind (CUDA error at ~30K tokens) | makes MTP testable at long context again | none | Medium, unproven |
| 6 | `--spec-draft-n-max 3` at 96K | +10-15% if mean accepted rises | candidate count | Low |
| 7 | Double-buffer K and V within a single slot to recover overlap at 1 slot | unknown | none | High |

**Notes**

- The slot count is not a free knob. Two slots are what hide the copy behind compute, and
  at 200K copy and compute are the same order of magnitude (160 ms against 132 ms), so any
  change that reduces overlap costs more than the VRAM it frees.
- The **7.59 t/s ceiling** is compute-bound (`47.43 + 0.4119 x 204.8` ms) while KV is
  staged. With a CPU assist the DRAM pool sets a lower floor (121 ms, 8.3 t/s) but the GPU
  compute chain keeps the practical ceiling near 7.
- Items 2 and 3 are complementary, not alternatives: see 13.5.

---

## 10. Recommended server settings

### 200K (decode-focused)

```powershell
.\serve.ps1 -Ctx 204800 -StagedLayers 8 -VecQuantKv -Mtp 0 -OverrideTensor 'output=CPU'
```

| Setting | Value | Why |
|---|---|---|
| `-Ctx` | 204800 | 225 MiB/layer, 3,600 MiB total KV |
| `-StagedLayers` | **8** | best measured. L=8 only fits because of the output offload; L=9 is 5.67 t/s, L<8 buys nothing (14.4) |
| `-ot output=CPU` | **on** | frees 644 MiB for 0.9 ms; nothing else frees that much (14.3) |
| `-t` | 6 (default) | 4, 6, 8, 12 threads all measured the same, the GEMV is memory-bound |
| `-VecQuantKv` | **on** | no 838 MiB f16 scratch; required below L=14 |
| `-Mtp` | **0** | MTP is a net loss at 200K |
| `-TypeK` / `-TypeV` | q4_0 | 1,152 B/token/layer |
| `-Batch` / `-UBatch` | 512 / 128 | prefill throughput |

### 32K to 64K (speculation pays here)

MTP 
_max=2 is worth **1.27-1.30x** on real code at 32K (18.68 -> 23.78 t/s, acceptance
72.9%) and output stays byte-identical. Do **not** stack 
gram-mod: it drops the gain from
1.27x to 1.04x. Do not use 
gram-mod alone (0.87x). See report 15.2.

### 96K (general use, shipped defaults)

```powershell
.\serve.ps1
```

12 staged, 4 resident, MTP n_max=2, q4_0 draft, draft ubatch 8 -> 11.31 t/s.

### Anything under ~60K

Leave KV resident (0 staged) and let llama.cpp keep everything on the GPU: 15.5 t/s at
40K, 14.47 t/s at 53K.

---

## 11. Reproduction

```powershell
# rebuild (vcvars64 is required, bare nvcc fails with "Cannot find compiler cl.exe")
cmd /c tests\_rebuild_server.bat

# 200K run from scratch
$env:LLAMA_KV_STAGED_LAYERS='9'; $env:LLAMA_FA_VEC_QUANT_KV='1'
llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf -c 204800 -ngl 99 -fa on `
  -ctk q4_0 -ctv q4_0 -b 512 -ub 128 -np 1 --fit off --no-context-shift `
  --slot-save-path results\.kv_cache\    # trailing separator is required

# restore instead of prefill (5 s instead of 931 s)
# POST /slots/0?action=restore   {"filename": "kv-200k-q4_0-L14.bin"}
```

Helper scripts used for measurement (recreate as needed): `_p1.py` builds a prompt and
times prefill, `_p2.py` / `_dec.py` time decode, `_slot.py` drives save/restore,
`_run.py` / `_txt.py` run a completion and report timings.

---

## 12. Housekeeping

| Item | State |
|---|---|
| Server | not running, GPU idle at 12,115 MiB free |
| Disk | 8.62 GB free, 3 snapshots (200K 4,047 MiB, 128K 2,600 MiB, 96K 1,996 MiB) |
| Fork changes | `fattn.cu` (vector switch, env-gated), `staged-kv.cuh` (n_slots may be 1, default still 2), `fattn-common.cuh`, `fattn.cuh`, `ggml-cuda.cu` |
| Removed diagnostics | `LLAMA_KV_STAGE_NODMA`, `LLAMA_KV_STAGE_COPYFRAC`, `LLAMA_KV_STAGE_1STREAM` (ineffective) |
| Kept diagnostics | `GGML_KV_STAGE_PROF` (off by default), `LLAMA_KV_STAGE_SLOTS`, `LLAMA_KV_STAGE_TRACE`, `LLAMA_KV_STAGE_JOIN` |
| Benchmarks (sources kept) | `host_dram_under_dma.cu`, `cpu_q4_attn_bench.cu`, `tests/cpu_attn_probe.cpp`, `bw_test.cu`, `kv_dma_bench.cu` |
| VCS | nothing committed or pushed |

---

## 13. CPU assist: exploration and rejection

The hypothesis: the DMA uses 13.3 GB/s of host DRAM but the machine has more, so a CPU-side
consumer could use the difference without touching PCIe, freeing VRAM for more resident KV.

The premise about spare bandwidth is correct. **The idea still fails**, because for this
workload the CPU is compute-bound, not bandwidth-bound. Everything below is measured.

### 13.1 Measured: the spare DRAM is real

`host_dram_under_dma.exe`, 2 GiB pinned buffer, DMA streaming pinned H2D while AVX2 workers
read the same buffer:

| workers | CPU read idle | DMA alone | CPU read under DMA | DMA under CPU | combined |
|---:|---:|---:|---:|---:|---:|
| 6 | 19.33 GB/s | 13.19 GB/s | 11.27 GB/s | 8.86 GB/s | 20.13 GB/s |
| 3 | 20.04 GB/s | 13.25 GB/s | 11.10 GB/s | 9.71 GB/s | 20.80 GB/s |
| 1 | 19.33 GB/s | 13.18 GB/s | 8.59 GB/s | 12.42 GB/s | 21.01 GB/s |

1. Host DRAM tops out at about **20 GB/s combined**.
2. The CPU share is **not free**: 11 GB/s of CPU reads throttle the DMA from 13.19 to
   8.86 GB/s. Ceiling is 20/13.3 = 1.5x, not additive.
3. Knee: at 8.59 GB/s of CPU reads the DMA still holds 12.42 GB/s; at 11.27 it collapses.

### 13.2 Measured: the CPU cannot convert that bandwidth into attention

First attempt (`cpu_q4_attn_bench.exe`) suggested 19 GB/s on the CPU, but that benchmark was
**misleading**: it did QK^T with 8 heads and no V pass. Qwen decode has **32 query heads over
8 KV heads** (GQA 4:1), so the real op does 4x the math plus the AV accumulation.

The authoritative number comes from **ggml's production CPU kernel**, driven through the
public API at our exact shape (`tests/cpu_attn_probe.cpp`, q4_0 K/V, 8 KV heads, 32 query
heads, 204,800 positions):

| threads | per layer | KV read |
|---:|---:|---:|
| 1 | 539.03 ms | 0.44 GB/s |
| 3 | 201.77 ms | 1.17 GB/s |
| 6 | 120.95 ms | 1.95 GB/s |
| 12 | **84.46 ms** | **2.79 GB/s** |

Against 5.30 ms per layer on the GPU, the CPU is **16x slower** at best.

**Why it cannot be fixed by better code.** Per q4_0 KV byte the op needs
`8192 MACs / 576 B` = 14.2 MAC/B = 28.4 FLOP/B after the GQA expansion. To consume the 20 GB/s
of DRAM it would have to sustain **568 GFLOP/s**. The Ryzen 5 4600G peak is
`6 cores x 8 lanes x 2 x 3.7 GHz` = 355 GFLOP/s, and realistic q4_0 dequant+FMA throughput is
70-100 GFLOP/s. The CPU is therefore capped near **3-5 GB/s** on this op, which is what the
84 ms measurement shows. Even a perfect kernel bounded by the 355 GFLOP/s peak would land
around 12.5 GB/s, i.e. ~45 ms per layer.

### 13.3 What the measured rate means for decode

Feeding the real CPU rate into the chunk-split balance from 13.4 (`(1-g) x C_cpu = 5.3g`):

| CPU per layer | GPU share g | PCIe payload | Chain | Pass | Decode |
|---:|---:|---:|---:|---:|---:|
| 84.46 ms (12 threads, measured) | 0.941 | 1,998 MB | 129 ms | **175 ms** | **5.61 t/s, no gain** |
| 45 ms (projected perfect kernel) | 0.895 | 1,900 MB | 127 ms | 167 ms | 5.99 t/s, +7% |
| 20 ms (4x better than ggml) | 0.790 | 1,677 MB | 122 ms | 149 ms | 6.7 t/s, +20% |
| 11 ms (theoretical memory floor) | 0.675 | 1,433 MB | 117 ms | 132 ms | 7.6 t/s |

The measured kernel delivers **exactly zero**: it shaves 5.9% of the PCIe payload, which is
94 MB out of 2,123 MB, about 7 ms, inside the run-to-run noise of a 175 ms pass.

### 13.4 Why layer-granular offload fails outright

Offloading whole layers, or whole weight matrices, puts CPU work on the serial critical path.
Layers are strictly sequential, so the GPU idles while the CPU works, and the CPU is far
slower per byte: host DRAM gives 8-20 GB/s against 350 GB/s of VRAM. Moving 1 GB of weights
to host costs 1 GB per token from DRAM, i.e. 120-250 ms serial, to save 17.7 ms of PCIe. That
is the stock `-ngl 48` cliff, and why stock is 0.97 t/s.

Whole-layer attention offload is worse still: one staged layer costs 17.7 ms of PCIe plus
5.3 ms of GPU compute, against 84.46 ms of serial CPU time.

**Weights on the CPU are ruled out.** KV is the only host-resident data, and 13.2 shows the
CPU cannot process it fast enough either.

### 13.5 The design that was tested on paper: intra-layer chunk split

KV positions inside one attention are independent, so a layer can be split into a GPU chunk
and a CPU chunk that run in parallel, merged by an online-softmax combine over ~4 KB of
partials. This is the only structure that avoids the serial penalty, and it is what the
numbers in 13.3 assume:

```
per staged layer:  max( 5.3 ms x g , (1-g) x 225 MiB / r_cpu )
```

It does not rescue the idea, because the CPU side of that `max` is 16x the GPU side.

### 13.6 Interaction with the attention kernel

At 200K the GPU attention is itself slow: 84.4 of the 131.8 ms compute comes from the vector
kernel running at ~42 GB/s effective. That is the real target, not the CPU.

| Configuration | Pass | Decode |
|---|---:|---:|
| today (L=9, PCIe only) | 175 ms | 5.61 t/s |
| CPU assist, measured kernel | 175 ms | 5.61 t/s, **no gain** |
| faster GPU attention alone | 175 ms | 5.61 t/s, **no gain**, still PCIe-bound |
| both | would have been ~7.0-8.3 | not reachable, CPU leg is 16x too slow |

### 13.7 Verdict

**Rejected.** The spare host DRAM bandwidth (about 7 GB/s beyond the DMA) is real but not
usable for this model: attention over KV needs 568 GFLOP/s to consume 20 GB/s, and this CPU
peaks at 355 GFLOP/s. The bottleneck on the CPU side is arithmetic, not the memory channel
that was hoped to be free.

Artifacts kept: `host_dram_under_dma.cu`, `cpu_q4_attn_bench.cu` (the misleading 8-head
microbenchmark, kept as a caution), `tests/cpu_attn_probe.cpp` (the authoritative ggml
measurement), and their build scripts.

The remaining levers are unchanged: a faster GPU attention kernel (9.2 item 3) and weight
quantisation (9.2 item 1, ruled out on quality). At L=9 the pass is 91% link-limited, so
only reducing staged bytes helps.

---

## 14. Output layer on the CPU: a working lever

Follow-up to section 13. The guess was that some large, low-arithmetic-intensity part of
the model could run on the CPU in parallel. The tensor table says the vocabulary tensors
are the only candidates.

### 14.1 Where the bytes actually are

From the GGUF (866 tensors, 9,948.57 MiB):

| tensor | size | type | access | traffic per token |
|---|---:|---|---|---:|
| `token_embd.weight` | 388.4 MiB | IQ2_S | gather, one row | **1.6 KB** |
| `output.weight` | 644.1 MiB | IQ4_XS | full GEMV | **644 MiB** |
| MTP `blk.64` | 332.3 MiB | Q6_K | - | not loaded |
| GDN layers, 48 of them | 7,452.8 MiB | mixed | dense GEMV | all |
| attention layers, 16 | 1,130.9 MiB | mixed | dense GEMV | all |

### 14.2 token_embd is already off the GPU

`-ot token_embd=CPU` changes nothing, and the buffer accounting explains why:

```
CPU_Mapped model buffer size = 388.38 MiB      <- the embedding, in host RAM already
CUDA0 model buffer size     = 9227.87 MiB
```

llama.cpp already keeps the gather-only embedding in mapped host memory. 9,616.3 MiB of
non-MTP weights minus 388.38 = 9,227.87, exactly the CUDA0 figure. **No gain available.**

### 14.3 output.weight offloads at almost no cost

`output.weight` is 644.1 MiB read by a full GEMV every token, i.e. **3.76 FLOP/byte**,
which is the one intensity this CPU can sustain. Measured end to end at 200K:

| staged | output on | free VRAM | decode | pass |
|---:|---|---:|---:|---:|
| 9 | GPU | 513 MiB | 5.64 t/s | 174.5 ms |
| 9 | **CPU** | 1,333 MiB | 5.67 t/s | 173.6 ms |
| **8** | **CPU** | 1,107 MiB | **5.77 t/s** | **170.6 ms** |
| 7 | CPU | 881 MiB | 5.75 t/s | 171.1 ms |
| 6 | CPU | 657 MiB | 5.53 t/s | 178.1 ms |
| 5 | CPU | 431 MiB | 5.74 t/s | 171.6 ms |
| 4 | CPU | - | load fails, slots do not fit | - |
| 7 | GPU | - | **load fails**, slots do not fit | - |

Two things worth noting:

1. Handing 644 MiB to the CPU costs only **0.9 ms** (174.5 -> 173.6 at the same L). The
   GEMV is largely hidden behind the pipeline's tail DMA. This also settles the question
   from 13.4: an offloaded GEMV is *not* automatically fatal, because it is not the only
   work in flight when it runs.
2. **The offload is what makes L<=8 reachable at all.** L=7 with the output on the GPU does
   not load. L=8 gives 5.77 t/s, a 2.3% gain over the 5.64 baseline.

Thread count is not a lever: `-t 6` 5.77, `-t 8` 5.76, `-t 4` 5.58. The GEMV is
memory-bound and six threads already saturate it.

### 14.4 The plateau matters more than the gain

Between L=8 and L=5 the payload falls from 1,888 MB to 1,180 MB, i.e. DMA time from 142 ms
to 89 ms, yet the pass stays at 171 +/- 1 ms. **The pipeline is no longer PCIe-bound.** It
is chain-bound, and the chain decomposes as:

```
47.4 ms  weights and GDN          (9,227 MiB at ~350 GB/s, efficient)
84.4 ms  attention, 16 layers     (225 MiB each at ~42 GB/s effective, the vec kernel)
24.0 ms  output GEMV on the CPU   (644 MiB, partly hidden)
------
~156 ms  chain, against 142 ms of DMA at L=8
```

### 14.5 Consequence: the attention kernel is now the whole game

With the chain at 156 ms, every further staging reduction is wasted: the DMA is already
85% of the chain and falling. The 84.4 ms of attention is 54% of the chain and is running
the vector kernel at 42 GB/s on a card with 350 GB/s.

| Change | Chain | Pass | Decode |
|---|---:|---:|---:|
| today (L=8, output on CPU) | 156 ms | 171 ms | **5.77 t/s** |
| attention 3x faster, staging unchanged at L=8 | 99 ms | 157 ms (DMA-bound) | 6.4 t/s |
| attention 3x faster, then stage down to L=5 | 99 ms | 114 ms (chain-bound) | 8.7 t/s |

So the ordering is now clear: **fix the attention kernel first.** It is the only change that
lowers the chain, and the chain is the binding constraint. Reducing residency below L=8
yesterday would have been wasted work; it only pays once the chain is under the DMA time
again, and then it pays at roughly 1 t/s per 100 ms of DMA removed.

### 14.6 MTP now fits at L=9, and is still a loss

The 644 MiB released by the output offload is enough for the MTP draft head
(332.3 MiB of weights plus one 225 MiB draft KV layer), so the question was worth
testing. Measured at 200K:

| staged | output | MTP | free VRAM | decode |
|---:|---|---|---:|---:|
| 9 | CPU | off | 1,333 MiB | 5.67 t/s (173.6 ms) |
| 9 | CPU | **on** | 495 MiB | **4.56 t/s** (215.8 ms) |
| 8 | CPU | on | - | load fails, needs one more 225 MiB layer |

So the offload did buy something real here: MTP at 200K used to need L>=13 and now loads at
L=9. It is still a **21% loss**, and the chain accounting shows why:

```
one target pass, no MTP : 173.6 ms -> 1.00 token
one target pass, MTP    : ~388 ms  -> 1.80 tokens (acceptance 0.4058, mean length 1.80)
break-even needs        : ~312 ms  -> the extra 214 ms buys only 0.8 token (139 ms worth)
```

MTP pays a full attention sweep over 204,800 positions for every drafted token, and the
attention sweep is now 54% of the chain (84.4 ms of 156 ms). Break-even would need a mean
accepted length of 2.23, i.e. acceptance near 0.6 instead of 0.41.

**MTP and plain decode therefore want the same fix.** Projected at a 3x faster attention
kernel, both become attractive at once:

| Config | Chain | Pass | Decode |
|---|---:|---:|---:|
| no MTP, L=8 | 99 ms | 157 ms | ~6.4 t/s |
| no MTP, staged down to L=5 | 99 ms | 114 ms | ~8.7 t/s |
| MTP on, L=9 | ~157 ms per pass | 157 ms | ~11.5 t/s equivalent |

The last row is the interesting one: MTP's arithmetic only works once the per-pass attention
cost is small relative to the pass itself. That is the strongest argument yet for building
the attention kernel rather than continuing to shuffle residency.

---

## 15. Speculation at 200K: measured honestly, and rejected

Prompted by the AMD fork in `amd_ref/` (section 5 of `amd_borrow_plan.md`), whose README
reports MTP giving 1.9x at 1k and 3.3x at 262k. Our fork already implements every piece of
their spec configuration (`ngram-mod`, `--spec-draft-sampling probabilistic`,
`--spec-ngram-mod-n-*`), so this needed no code.

### 15.1 The artifact that almost got reported as a win

Their config (`--spec-type draft-mtp,ngram-mod`, n-match 8, n-max 64, probabilistic) at
200K on our existing snapshot:

| config | decode | vs baseline | acceptance |
|---|---:|---:|---:|
| L=9, no speculation | 5.68 t/s (175.1 ms) | 1.00x | - |
| L=9, ngram-mod | **8.43 t/s** (118.1 ms) | **+49%** | 100% |

Repeatable, and byte-identical output. It looked like a step change.

It is an artifact. **Our 200K snapshot's continuation is a run of newline characters**
(`head=' \\n\\n\\n\\n...'`), because `big.prompt.txt` is synthetic repetitive prose that
ends mid-thought. A run of newlines is the best case n-gram matching can ever have, so the
matcher predicted every token and every draft was accepted.

### 15.2 The control that settles it

Same model, real C++ source as the prompt (`code_prompt.txt`, 50 KB of this repo's own
llama.cpp sources plus a code-writing task), 32K context, KV fully resident, 192 tokens,
greedy, output byte-identical in every row:

| config | decode | vs baseline | acceptance |
|---|---:|---:|---:|
| no speculation | **18.68 t/s** (53.3 ms) | 1.00x | - |
| MTP `n_max=2` | **23.78 t/s** (41.8 ms) | **1.27x** | 72.9% |
| MTP `n_max=3` | **24.28 t/s** (41.0 ms) | **1.30x** | 62.0% |
| MTP + ngram-mod | 19.38 t/s (51.3 ms) | 1.04x | 36.5% |
| ngram-mod alone | 16.25 t/s (61.2 ms) | **0.87x** | 10.3% |

Three conclusions:

1. **ngram-mod does not generalise.** 100% acceptance on a newline run, 10.3% on real code,
   where it is a 13% *loss* because a wrong draft costs a verify pass.
2. **Stacking ngram with MTP destroys MTP** (1.27x -> 1.04x). Their fork's headline config is
   tuned for repetitive SQL boilerplate; on real code the wrong ngram drafts shorten the
   effective verify batch.
3. **MTP itself is genuinely good on code: 1.27-1.30x.**

### 15.3 Why MTP still loses at 200K

At 200K an MTP draft is not cheap: the draft layer must attend over all 204,800 positions,
i.e. it pays the full attention sweep that dominates our chain.

```
target pass at 200K, no spec      : 175 ms -> 1 token
target pass + one draft pass      : ~264 ms -> 1.24 tokens    (acceptance 72.9%, measured)
break-even needs mean accepted    : >= (175 + 89) / 175 = 1.51
```

Measured at 200K: MTP `n_max=2` gives 4.69 t/s against 5.68 without, i.e. **0.83x**. It needs
mean accepted length 1.51 and delivers 1.24, so the draft pass costs more than the tokens it
saves. `n_max=3` makes it worse (mean falls to 0.96).

**So MTP is a win below roughly 64K and a loss at 200K, and the crossover is set by the cost
of one attention sweep.** Their fork gets 3.3x at 262k because their sweep is cheap relative
to a PCIe 5.0 pass; ours is 84.4 ms of chain.

Two corollaries:
- Below ~64K, MTP `n_max=2` is a real +27% and should be on. It already is, in the 96K profile.
- At 200K, MTP can only win after the attention kernel (15.4) makes the draft pass cheap.

### 15.4 Long drafts crash at 200K, and that is diagnostic

`--spec-ngram-mod-n-max 128` at L=9, and `64` at L=8, both die mid-decode:

```
CUDA error: out of memory
  in function alloc at ggml-cuda.cu:546
```

A verify batch of more than two tokens leaves our vector kernel (`cols_per_block` is 1 only
for `n_q == 1`), and the tile/MMA fallback tries to allocate the 838 MiB f16 K/V scratch.
That is the same root cause as section 14.5, now with a second symptom: **it does not merely
slow speculation down, it caps draft length at two tokens.**

The AMD fork's `fattn-paged-decode` exists precisely to remove this ceiling (1-8 query
tokens, split-K, no f16 scratch). Its constants are hardcoded for their shape, so it is a
reference design rather than a port, but it is now the highest-value item on the list.

### 15.5 Benchmarking discipline, learned the hard way

Any speculation measurement on `big.prompt.txt` is invalid: its continuation is degenerate
and flatters every drafter. Throughput baselines are fine on it (decode cost is
content-independent), but draft acceptance must be measured on `code_prompt.txt`, which is
what the target workload actually is. `_cfg.py --prompt-file code_prompt.txt --no-restore`
is the representative path.

---

## 16. Tier 1: the catch-up merge, and the n_max=1 discovery

Porting the AMD fork's MTP catch-up merge (`c288a0fac`) was the next item.

### 16.1 The port

`git apply` accepted their patch unmodified: our fork is the same llama.cpp base, has
`llama_process` with `LLAMA_PROCESS_TYPE_DECODE`, and its
`common_speculative_impl_draft_mtp` carries every member the patch expects
(`i_batch_beg`, `is_mem_shared`, `chain_heads`, `verify_h`, `pending_h`, `set_embd_state`).
One edit was needed: the patch bumps `cparams.n_tokens_decode`, a field that comes from their
KV-stream phase switch. This fork has no phase-switched compute buffers, so the extra slot is
unnecessary and the line became a no-op with a comment (`common/speculative.cpp:2756`).

Effect: at 200K with `n_max=2`, 4.69 -> 4.80 t/s (+2.3%). At 32K on real code, unchanged
within noise. Output byte-identical. The patch does exactly what it says - it removes one MTP
pass over the verify rows - but that is the **cheap** pass. Kept, because it also stops
rejected drafts from entering the draft KV (a quality fix worth more than its 2.3%).

### 16.2 One draft token beats two, at both ends of the context range

Measuring the marginal cost of drafts produced the useful result:

| context 32K, real code | decode | vs no-spec |
|---|---:|---:|
| no speculation | 18.68 t/s | 1.00x |
| MTP `n_max=1` | **24.44 t/s** | **1.31x** |
| MTP `n_max=2` | 23.83 t/s | 1.28x |
| MTP `n_max=3` | 23.74 t/s | 1.27x |

| context 200K | decode |
|---|---:|
| no speculation | 5.81 t/s |
| MTP `n_max=1` | 5.74 t/s (parity) |
| MTP `n_max=2` | 4.80 t/s |
| MTP `n_max=3` | 4.69 t/s |

The second draft pass costs a full attention sweep at 200K: 5.74 -> 4.80 t/s. At 32K the draft
is nearly free, so `n_max=2` is close but still behind, because the extra draft rarely returns
a third token (acceptance falls from 84% to 73%).

**`n_max=1` is optimal at both ends**, so `serve.ps1` now defaults `-Mtp` to 1 instead of 2.
At 32K this is worth 1.31x over no speculation; at 200K it is neutral, so speculation can stay
on rather than being switched off by context.

### 16.3 Where the remaining 200K speculation cost sits

Per cycle at 200K: one verify pass (n_q = 2, still in the vector path and cheap) plus one
draft pass, which is a **full attention sweep over 204,800 positions** because the MTP layer
attends over the whole context. That draft sweep is what keeps MTP from paying at 200K, and
it is the same 84.4 ms of attention that bounds plain decode. Nothing in Tier 1 can remove
it: it is real work, not redundant work.

Hence T2 is the only path that makes speculation profitable at 200K, and it helps plain
decode by the same amount.

### 16.4 T1.2 deferred

The graph-arena patches (`e54b93bc0`, `ec12b5fc6`) build on their phase-switched arena
infrastructure, which this fork does not have; adopting them means adopting their KV-stream
scheduler first. Decode here is DMA-bound, so graph rebuild is not the constraint. Revisit
after T2.

---

## 17. Offline layer-group prefill and a validated 200K snapshot (2026-10-19)

Priority: stop paying full 200K prefill for each decode experiment. The separate
`group_prefill.exe` producer runs all prompt chunks through layers 0-31, exports
their raw FP32 boundary activations and complete sequence state to host RAM, unloads
that model/context, loads layers 32-63 on the GPU, and runs the second group. Active
KV stays resident. Inactive weights/KV stay on the host. Both groups use MMA attention.
No new approximation, weight quant or KV quant was introduced.

### 17.1 Conservative full-size measurement

Same `big.prompt.txt` as the historical 200K study; Q4_0 K/V, split 32, ubatch 512.
The producer deliberately leaves the final prompt token uncached to recover logits
without a recurrent rewind when the snapshot is restored.

| Metric | Measured |
|---|---:|
| Original prompt tokens | 204,365 |
| Cached tokens | 204,364 |
| First group compute | 398.6 s |
| Second group compute | 391.1 s |
| Export/unload/load/import between groups | 6.65 s |
| Producer wall time before final save | 801.315 s, 255.04 tok/s |
| Final snapshot save | 5.149 s |
| Total including save | 806.464 s, 253.41 tok/s |
| Snapshot size | 3,929,454,216 bytes |

The historical MMA baseline was 931.8 s / 219.3 tok/s. The first implementation is
about 1.16x faster before save, a modest improvement, not a multi-fold measured PP
speedup. It also avoids the forced-vector prefill path, which was 3.8x slower than
MMA in the earlier 32K comparison. The completed larger-microbatch measurement is below.

### 17.2 Correctness and consumer validation

- At 2,048 tokens, split vs ordinary full-model prefill gives **byte-identical whole
  snapshots**, not merely matching generated text. Validated at ubatch 128 and 2048.
- Restoring those snapshots with zero/eight staged layers gives identical logits
  across all 248,320 vocabulary entries and identical 16-token greedy continuations.
- At 200K, restoring the new snapshot with nine/ten staged layers gives **max logit
  difference 0.0** and identical 16-token continuations.
- The actual server slot API restores all 204,364 tokens in **2,556.901 ms**. The next
  completion reports `cache_n=204364`, `prompt_n=1`. The 16-token readiness check was
  5.70 tok/s; this short check is not a new sustained decode benchmark.
- Its continuation is coherent repetitive prose, not the old snapshot's newline
  stream. The prior staging-depth discrepancy is absent on this new snapshot; this
  does not establish which earlier code or snapshot defect caused the old discrepancy.

Artifact: `results/.kv_cache/kv-200k-group-q4.bin`, SHA256
`eac1d02ad341a4211622681d422bd9e68e1d9687dc595adeed0dced950e258cf`.
Its JSON manifest records model/prompt/binary identities. Full logs and consumer
checks are in `results/group-prefill-200k.log` and `results/group-prefill-200k-check/`.
Producer binary timestamps/hashes are provenance, not an automatic reason to redo
prefill after each decode-kernel build. Compatibility must follow state semantics.

`_cfg.py` now defaults to the validated ubatch-2048 snapshot below and fails on restore failure or a
manifest prompt mismatch, rather than silently paying for another full prefill. Its
old `mean` label was generated/drafted tokens, not tokens committed per verify round;
it is now labelled `tokens_per_drafted_token`.

### 17.3 MTP boundary

The sequence file is **target-only**, with all attention and GDN state. It does not
contain initialized MTP draft/speculative state. Restored MTP acceptance from an empty
draft cache must not be used as an acceptance benchmark. `_cfg.py` rejects that
combination for a target-only manifest.

The producer can optionally retain every normalized target hidden state using
`--nextn-out`, or `--save-nextn` in the Python runner. This companion is intended to
let a future initializer build the one-layer MTP state without redoing the 64-layer
target prefill. It is not itself a complete MTP server snapshot.

See `dev/group-prefill.md` for build, production and restore commands.

### 17.4 Final ubatch-2048 measurement, including retained target hidden states

| Metric | Measured |
|---|---:|
| First group | 378.8 s |
| Second group, including hidden-state writes | 380.6 s |
| Export/unload/load/import between groups | 6.659 s |
| Wall time before final snapshot save | 770.698 s, 265.167 tok/s |
| Final snapshot save | 4.765 s |
| End-to-end process wall time including save/shutdown | 776.642 s, 263.138 tok/s |
| Target snapshot | 3,929,454,216 bytes |
| Target hidden-state companion | 4,185,374,744 bytes, 204,364 x 5,120 F32 rows plus header |
| Sampled peak GPU memory | 11,659 MiB |
| Actual server restore | 2,405.613 ms |

This is about 1.21x the historical 219.3 tok/s MMA rate before save, or about 17%
less wall time using the conservative end-to-end comparison. It is only about 4%
faster than the new ubatch-512 producer. The large iteration win is reusing the
validated cache in 2.4 seconds, not a claimed multi-fold improvement in cold prefill.

Recommended artifact: `results/.kv_cache/kv-200k-group-u2048-q4.bin`, SHA256
`314a08d1ff42db0467c3502f15549865a609b896c817f69c9858fe5985ec7b1f`.
Companion: the same filename plus `.nextn.f32`, SHA256
`626cf1338308babbea9e5f5a2cf07d0d63ded2b6a9876c4eeb27d3ac7b64e18b`.
Its JSON manifest holds the producer identity and full progress summary.

Consumer check: `results/group-prefill-200k-u2048-check/summary.json`. Nine/ten
staged layers give max logit difference 0.0 and identical 16-token output. The
server reports `cache_n=204364`, `prompt_n=1`, and the same coherent continuation as
the ubatch-512 snapshot. The runner now defaults to ubatch 2048; ubatch 512 remains
a tested conservative setting. No further batch sweep was run.

### 17.5 Tiny-tail regression and exact output-head overrides

An additional 2,051-token check (2,048 plus a three-token tail) initially failed.
Tensor tracing showed identical attention Q/K/V, gated attention output, projection
weights and projection input. The output projection itself differed because the
producer's unanchored `output.*` override also matched `blk.3.attn_output.weight`
and the other attention projections. For tiny batches this selected CPU math;
large batches offloaded the same operation to the GPU.

The producer now uses `^output\.weight$`. `_cfg.py` uses that same anchored pattern,
and `serve.ps1` normalizes its `output=CPU` shorthand to an exact head match. The
2,051-token whole snapshot and hidden-state history now match the full-model
reference byte for byte. Restored logits match exactly at zero/eight staged layers,
as do all 16 generated tokens. Evidence: `results/group-prefill-check-1792405238/`.

The measured 200K artifacts predate the placement correction. Their final batches
have 76/1,612 tokens, so the projections used GPU math; the full-batch golden checks
and 200K consumer checks pass. No new full 200K timing was paid for this small
placement-only correction. The broad historical `-ot output=CPU` setting must not
be interpreted as offloading only the 644 MiB vocabulary head.

The corrected `_cfg.py` restore path was also checked with the anchored CPU head:
`cache_n=204364`, identical 16-token text, 4.84 tok/s in that short check. The GPU-head
consumer check was 5.70 tok/s. Therefore the default experiment driver keeps the head
on CUDA; CPU output is an explicit memory tradeoff, not a demonstrated free default.
The old section 14 "nearly free" attribution needs fresh anchored-pattern measurements
before reuse. None of these short checks establish a new sustained decode benchmark.

Final default-driver check: `python _cfg.py --staged 9 --spec none --n 16` reused
204,364 cached tokens and returned the same 16-token continuation at 5.71 tok/s.
The driver now keeps the head on CUDA by default. All test servers and producer
processes were stopped after validation; the GPU was idle at completion.

---

## 18. Historical snapshot discrepancy: observations retained, cause unresolved

### 18.1 How it showed up

While sweeping staged depth with MTP, the generated text changed with `LLAMA_KV_STAGED_LAYERS`:

| staged layers | output (192 tokens, same prompt, same restored snapshot) |
|---|---|
| L=9 | newline run, `sha256 e48bf1f5b3f1d43e` (reproduced twice) |
| L=10 | `"\n\n. the job.\n\nHere is the text extracted from the image:` `sha256 64acdcb2f1a7da19` (twice) |
| L=12 | ` ---\n\n\n\n...` `sha256 2a7cd88fe98115b9` |

Each depth was deterministic, but the depth must not matter at all: staging changes where a layer's
KV lives, not its value. So something was wrong, and every 200K number in sections 15-16 was
measured on text that was not the model's real continuation.

### 18.2 Short-context controls

32K capacity, about 14.9K filled with real code, fresh prefill, no snapshot:

| config | output |
|---|---|
| L=0 (all KV resident) | `sha256 9924fc0bca5e5c21` |
| L=4 | `sha256 9924fc0bca5e5c21` |
| L=8 | `sha256 9924fc0bca5e5c21` |

Identical in this control. The layer partition is right (Bresenham over
the 16 KV layers, exactly `staged` selected - `llama-kv-cache.cpp:294`).

Save and restore are also depth-invariant on this short prompt: a snapshot written at L=8 and restored at
L=0, L=4 and L=8 gives `9924fc0bca5e5c21` in all three cases, equal to a fresh prefill.

### 18.3 Timestamps did not establish a cause

```
kv-200k-q4_0-L14.bin     19-10-2026 07:50:23
staged-kv.cuh            19-10-2026 09:04:53     <- 74 minutes newer
```

The snapshot predates a source modification. This alone does not show incompatible state,
corruption, or why the outputs differed. The earlier report incorrectly called this a
confirmed root cause. No decisive fresh-snapshot comparison had completed in that session.

The old file is unsuitable as a correctness or acceptance baseline because of its observed
depth-dependent continuation. The new section 17 snapshots pass direct 200K logit checks
across staging depths, but those checks do not identify the original defect.

### 18.4 Compatibility rule, corrected

Record the model, prompt, KV types, producer build and snapshot identity. Reuse the snapshot
across decoder builds when its representation and semantics remain compatible. A different
binary timestamp is not sufficient grounds to repeat prefill. Revalidate state/logit parity
when changing representation or state handling. A prompt mismatch is a separate token-prefix
issue and cannot be attributed to file age without evidence.

---

## 19. Prefill throughput at 2K: the vector gate was costing 37%

### 19.1 Objective and the arithmetic ceiling

Target: 2K context, KV staging off, the whole prompt in one pass. The stated ideal was
"load the weights once": 9.73 GiB at 360 GB/s = 29 ms for 2,048 tokens, i.e. 70k tok/s.

That ideal is not reachable, and not because of bandwidth. The model is **27.32 B
parameters at 3.06 bits/weight** (`_params.py`), so one 2,048-token pass costs

```
2 * 27.32e9 * 2048 = 111.9 TFLOP
intensity          = 111.9 TFLOP / 10.4 GB = 10,716 FLOP/byte
RTX 3060 balance   = 51.2 TFLOPS(fp16, fp16 acc) / 360 GB/s = 142 FLOP/byte
```

Arithmetic intensity is about **75x above the machine's balance point**, so this workload is
compute-bound by construction, and batching *more* tokens only pushes it further that way.

| scenario | time | required |
|---|---:|---:|
| weights read once, math free | 0.029 s | 4,145 TFLOPS |
| 2K before this change | 4.98 s | 22.5 TFLOPS |
| 2K now | 3.63 s | **30.9 TFLOPS** |

The "27 ms" fantasy needs **81x this card's 51.2 TFLOPS fp16 peak**. Measured throughput is
already about 60% of the 51.2 TFLOPS figure, so configuration has little headroom left.

### 19.2 What actually moved it

`LLAMA_FA_VEC_QUANT_KV` forced the **vector** attention kernel for every quantized-KV
attention, including prefill, where it is the slow path (`report` 3.4 measured 3.8x at 32K).
The gate is now bounded by `LLAMA_FA_VEC_MAX_NQ`, default 4:

```c
// vector prefill is 1.2x slower at 2K and 3.8x at 32K, so only decode-sized batches take it
if (vec_quant_kv && Q->ne[1] <= vec_max_nq) return BEST_FATTN_KERNEL_VEC;
```

Decode keeps the vector kernel (`n_q <= 4` covers 1-token decode and the 2-4 token MTP/ngram
verify batches), and prefill (`n_q = ubatch`) falls through to `MMA_F16`.

2048 tokens, ctx 4096, `big`-style code prompt, steady state (3 cold repeats):

| config | PP tok/s |
|---|---:|
| before: `-ub 128`, vector prefill | 411 |
| before: `-ub 2048`, vector prefill | 449 |
| **now: `-ub 2048`, MMA prefill** | **557-566** |
| now: `-ub 1024` | 551 |
| now: `-ub 512` | 550 |
| `-fa off -ctk f16 -ctv f16` | 531 |
| `-ub 4096`, 4096-token prompt | 539 |

Net **+37%** at 2K. Every configuration produced byte-identical output (`sha256 b25aee1d`),
so this is a pure speed change. The 4096-token row shows the curve flattening, i.e. 2K is
already past the amortisation knee.

### 19.3 Regression check on the 200K staged decode

The gate change must not touch decode. Verified with the validated group-prefill snapshot,
L=9, no environment overrides:

| metric | before (17.5) | after |
|---|---:|---:|
| cached tokens | 204,364 | 204,364 |
| decode, 16 tokens | 5.71 tok/s | 5.66 tok/s (-0.9%, noise) |
| continuation | coherent "Paragraph 2894..." | identical prefix |

`n_q <= 4` still returns `VEC`, so the 200K staged decode path is unchanged by construction.
Only `n_q >= 5` behaviour differs: those now take `MMA_F16`, which needs the 838 MiB f16
scratch and therefore does not fit at 200K anyway. Long ngram drafts at 200K were already
outside the supported set (`x128` OOM), so nothing supported regressed.

### 19.4 Knobs that did nothing

At 2048 tokens, MMA path: `GGML_CUDA_MMQ_PREC=q8` 550, `GGML_CUDA_GRAPH_OPT=1` 549,
`GGML_CUDA_PREFER_MMQ=1` 555, `-ctk/-ctv f16` 550, `-ctk/-ctv q8_0` 543. All within noise of
the 557-566 baseline. Prefill is not sensitive to KV type at 2K because the KV is only
~0.3 GiB there.

### 19.5 Where the remaining headroom is

Power stayed at **35-50 W of a 170 W limit** with 25-40% SM utilisation during prefill, which
looks like stalling. Section 20 disproves the obvious explanation (dequantize instruction
pressure): every weight type from 2.5 to 16 bits/weight reaches the same 30-43 TFLOPS, so the
kernels are not limited by dequant work at all. They sit at a uniform GEMM efficiency wall
of roughly 73-80% of the card's 51.2 TFLOPS fp16 peak. The 81x gap to "weights read once" is
not reachable on this GPU at any batch size.

### 19.6 Reproduce

```
python _cfg.py --ctx 4096 --staged 0 --b 2048 --ub 2048 --spec none --n 8 \
    --no-restore --repeat 3 --prompt-file prompt_2048.txt
```

`_cfg.py` now takes `--b`, `--ub`, `--ctk`, `--ctv`, `--fa`, `--no-vec`, `--repeat` and
prints prompt throughput alongside decode.

---

## 20. Pre-dequantized weights for prefill: measured, and capped at 1.2x

### 20.1 The idea and how it was tested

If prefill is limited by the cost of dequantizing 2.5-3 bit weights inside the matmul kernel,
then dequantizing once into a cheaper format (Q8_0, F16) and running a plain GEMM should be
faster. Dequantized weights do not fit: F16 is 54.6 GB and Q8_0 is 29.0 GB against 12 GiB of
VRAM, so either stream them from host memory a few layers at a time, or dequantize on the GPU
into a per-layer scratch buffer.

Before building either pipeline, measure whether the weight format even matters.
`tests/matmul_bench.cpp` runs `ggml_mul_mat` on a real layer shape (5120 x 17408, 89.1 M
parameters, the size of this model's `ffn_up`/`ffn_gate`/`ffn_down`) on the CUDA backend for
each type, reports the device the scheduler actually chose, and takes the best of six
steady-state repeats.

### 20.2 Result: throughput is almost independent of weight type

Single matrix, steady state, best of 6:

| weight type | bits/weight | 2048 tok ms | TFLOPS | 4096 tok ms | TFLOPS |
|---|---:|---:|---:|---:|---:|
| iq2_s | 2.5 | 11.96 | 30.5 | 24.09 | 30.3 |
| iq3_xxs | 3.1 | 9.75 | 37.4 | 20.40 | 35.8 |
| q4_0 | 4.5 | 9.77 | 37.4 | 17.98 | 40.6 |
| q8_0 | 8.5 | 9.44 | 38.7 | 17.72 | 41.2 |
| f16 | 16.0 | 9.97 | 36.6 | 17.07 | 42.8 |

Six and a half times the weight bytes buys **1.20x** at 2K and 1.41x at 4K. The kernels are
not dequantize-bound.

Measurement trap worth recording: the first three or four repeats are 5-10x slower than
steady state (5-100 ms versus 9-12 ms, varying by type). A single-shot benchmark reports
iq3_xxs at 4.1 TFLOPS and q8_0 at 38.8 TFLOPS, which would have "proved" a 9x win from
dequantizing. Only the steady-state numbers are real.

### 20.3 Why every variant of the idea loses anyway

Compute-only ceiling is 1.20x (iq2_s -> f16, 2K). Against that:

| variant | arithmetic | outcome |
|---|---|---|
| host dequant, stream F16 | 54.6 GB / 13.2 GB/s = 4.14 s per pass | slower than today's 3.63 s |
| host dequant, stream Q8_0 | 29.0 GB / 13.2 GB/s = 2.20 s, compute 111.9/38.7 = 2.89 s | 2.89 s if perfectly overlapped, 706 t/s (+25%) |
| GPU dequant to scratch, then Q8_0 GEMM | +38.7 GB of VRAM traffic (108 ms) + 54 ms more GEMM read traffic | 3.05 s, 671 t/s (+19%) |
| partial residency, ~1.5 GB of Q8_0 | ~7 of 65 layers upgraded (11%) | ~2% |
| GPU dequant to Q4_0 (4.5 bpw) | 25.1 GB traffic (70 ms) + 37.4 TFLOPS | 3.06 s, 669 t/s (+18%) |

So the whole family is worth at most ~1.2x, the host variant is a net loss because PCIe 3.0
is too slow for 29-55 GB per pass, and the on-GPU variant needs a new dequant-into-scratch
GEMM path in `ggml-cuda` to collect ~19%. Upward dequantization is quality-safe (Q8_0 and F16
are higher fidelity than IQ2_S), so this is purely a speed question, and the answer is no.

### 20.4 The useful side finding

The full model at 2K measures 30.9 TFLOPS. A single IQ2_S matrix at 2K measures 30.5 TFLOPS.
The model is running at *exactly* the per-matrix rate, which means attention, the GDN
recurrence, the norms and all launch overhead together cost approximately nothing measurable.
There is no overhead left to reclaim outside the GEMMs: prefill is the GEMM wall, end of
story. That also retires the "SMs are stalling, so there must be headroom" reading of the
35-50 W power figure in 19.5.

### 20.5 Reproduce

```
tests\build_matmul_bench.bat
cd llama_cpp_src\build-cuda\bin
matmul_bench.exe 2048 6
matmul_bench.exe 4096 6
```

Read the last reps only. Verify the device column says CUDA0; a silent scheduler fallback to
the CPU backend produces plausible-looking but meaningless numbers.

---

## 21. MMQ occupancy sweep: no gain. Vendor int8 GEMM: 2.4x available

### 21.1 The sweep

`_mmq_sweep.py` (external suggestion) changes the J=128 CASE entries of
`mmq-config-ampere.cuh`: nthreads, targeted occupancy, and I (the output-row SRAM tile).
Rationale offered: every Ampere entry is one block per SM at 254 registers, no double
buffering, so loads never overlap math.

| variant | nthreads | occupancy | I | 2K PP tok/s | iq2_s TFLOPS | text |
|---|---:|---:|---:|---:|---:|---|
| base (shipped) | 256 | 1 | 128 | **563.6** | 30.5 | b25aee1d |
| o2 | 128 | 2 | 64 | 546.8 | 31.2 | b25aee1d |
| o1 (control) | 128 | 1 | 64 | 547.0 | 31.1 | b25aee1d |
| o3 | 128 | 3 | 64 | 530.6 | 28.4 | b25aee1d |

Nothing beats the shipped config, and every variant is 3-6% slower end to end. Numerics are
unchanged throughout (identical 2K output hash), so this is a pure performance result.

The decisive row is **o1 versus o2**: 547.0 against 546.8 with occupancy 1 versus 2. Granting
two co-resident blocks per SM changes nothing measurable, so the load/compute overlap theory is
falsified. Halving the tile does make an isolated `mul_mat_q` call 2% faster (iq2_s 30.5 ->
31.2 TFLOPS), but end to end the doubled block count costs more than it gains.

### 21.2 The diagnosis that was right

Independently re-derived from `nsys_bench32.sqlite`:

- **98.6% GPU busy** across the traced 32K prefill (274.3 s of kernel time in a 278.2 s span).
  The earlier "27% busy, 40 W" reading in 19.5 is a Windows/WDDM reporting artifact, full stop.
- `mul_mat_q` is **66.6%** of kernel time, `flash_attn_ext_f16` 14.3%, then rms_norm, unary_gated_op,
  quantize_mmq_q8_1 and k_bin_bcast at 1-3% each.
- Every `mul_mat_q` launch is **254 registers/thread, 256 threads, 56-68 KB shared memory**:
  65,024 of 65,536 registers means exactly one block, 8 warps, per SM.

So the diagnosis was accurate. Only the proposed fix does not work.

### 21.3 What the hardware can actually do

Same shape (M=17408, N=2048, K=5120, 365 GFLOP), vendor cuBLAS against ggml's MMQ:

| engine | ms | rate |
|---|---:|---:|
| cuBLAS int8 (x int8 -> int32) | 4.08 | **89.5 TOPS** |
| cuBLAS int8 + per-row scale epilogue | 4.95 | 73.8 TOPS |
| cuBLAS fp16 | 8.51 | 42.9 TFLOPS |
| MMQ q8_0 (best MMQ type) | 9.38 | 38.9 TFLOPS |
| MMQ iq2_s (the model's bulk type) | 11.95 | 30.5 TFLOPS |

At 4096 tokens the ratio holds: cuBLAS int8 + epilogue 9.82 ms versus MMQ iq2_s 24.09 ms.

**2.4x is available on the matmul from a vendor library, at the epilogue-inclusive rate.**
The 30-43 TFLOPS plateau in section 20 was the ceiling of the MMQ kernel, not of the card.

### 21.4 What it would take, and what it would buy

int8 weights for the whole model are 25.4 GiB and do not fit in 12 GiB, so the route is to
dequantize each layer from IQ2_S to int8 **on the GPU into a scratch buffer** (read 27 MB,
write 89 MB per matrix, ~0.3 ms, no PCIe at all), then run cuBLAS int8 with a scaled epilogue.

Projecting at 2K with the traced 80/20 GEMM/other split of 3.63 s:

```
GEMM      2.9 s / 2.4 = 1.21 s
dequant   ~0.10 s
other     0.72 s
total     2.03 s -> about 1000 tok/s   (1.8x)
```

Long context gains more, because the GEMM share rises and the epilogue is amortised over more
tokens. This is the only route to four digits found so far, and it needs a new int8 `mul_mat`
path in `ggml-cuda`: dequant-to-int8 with per-row scales, per-token activation quantization,
cuBLAS int8 call, scaled epilogue, gated to large batches only so decode is untouched.

### 21.5 The cost: it changes arithmetic

The speedup comes precisely from dropping MMQ's per-32-element block scales in favour of one
weight scale per row and one activation scale per token. Weight fidelity actually improves
(IQ2_S is 2.5 bits; int8 per row is finer), and int8 x int8 -> int32 accumulation is exact.
The new error source is activation quantization, which is standard practice but is still a
numerical change: output cannot stay byte-identical, so it needs a divergence check (logit
max/RMS against the current path) before adoption. It does not touch quantization depth, so it
is not a smaller-weight-quant trade.

### 21.6 Reproduce

```
python _mmq_sweep.py                 # 4 rebuilds, ~14 min, restores + rebuilds on exit
cd llama_cpp_src\build-cuda\bin
cublas_gemm_bench.exe 2048 6
cublas_gemm_bench.exe 4096 6
```
The sweep leaves `mmq-config-ampere.cuh` byte-identical (verified by SHA256 against a backup).
`tests/cublas_gemm_bench.cu` needs only the CUDA toolkit, not ggml.

---

## 22. Activation outlier survey: the int8 path needs rotation first

### 22.1 Method

`GGML_CUDA_MM_STATS=1` (new, env-gated) computes max|x| and RMS(x) of the src1 activation of
every MUL_MAT with a prefill-sized batch (>= 256 tokens), and prints one line per matmul with
the weight tensor name. Run on the 2K code prompt, one pass, 573 matmuls captured.

The relevant number is `ratio = max/RMS`. A single int8 scale per token sets the step at
`max/127`, so `step/RMS = ratio/127`. Uniform quantization noise is `step/sqrt(12)`, so
`ratio` maps directly to relative noise: ratio 13 gives 2.9% of RMS, ratio 127 gives 29%.

### 22.2 Result: nothing in this model is safe

| matmul kind | count | median ratio | min | max | step/RMS | noise |
|---|---:|---:|---:|---:|---:|---:|
| **ffn_down** | 63 | **259.2** | 115.0 | **1248.1** | 2.04 | 59% |
| ssm_out | 48 | 174.2 | 50.4 | 981.7 | 1.37 | 40% |
| attn_q / attn_k / attn_v | 48 | 70.2 | 45.6 | 105.1 | 0.55 | 16% |
| attn_output | 16 | 68.2 | 21.4 | 229.3 | 0.54 | 16% |
| attn_qkv, ssm_alpha, ssm_beta, attn_gate | 192 | 64.9 | 32.6 | 97.5 | 0.51 | 15% |
| ffn_gate, ffn_up | 126 | 48.0 | 25.5 | 78.0 | 0.38 | 11% |
| attn_inp_k_rot, attn_inp_v_rot (rotated) | 80 | **10.3** | 3.8 | 46.2 | **0.08** | **2.3%** |

Worst single tensors: `blk.3.ffn_down` ratio 1248 (step 9.8x the RMS), `blk.21.ssm_out` 982,
`blk.0.ffn_down` 819. `ffn_down` is uniformly bad across all 63 layers (min 115), so this is not
a few unfortunate layers.

A target of step/RMS <= 0.1 (about 3% noise) is met by exactly one row of that table: the
already-rotated KV store path.

### 22.3 Rotation is the fix, and the fork already has the kernel

The last row is the proof: the same class of data, rotated, has a 6.8x lower ratio than unrotated
(`attn_k` 70.2 versus the rotated store path 10.3). That is the whole argument for QuaRot-style
rotation.

For this model the rotation is feasible because every reduction dimension divides by 1024:

| K | blocks of 1024 |
|---:|---:|
| 5120 | 5 |
| 6144 | 6 |
| 17408 | 17 |

Folding a block-diagonal normalised Hadamard into the weights is exact, not an approximation:
with H orthogonal and symmetric, `(x H) . (W H)^T = x H H^T W^T = x W^T`. The fork already
carries a complete CUDA FWHT (`ggml-cuda/fwht.cu`, widths 64 to 8192, block and warp variants),
so the transform itself does not need writing.

### 22.4 Consequence for the int8 plan

Naive per-token int8 activation quantization is dead on arrival for this model: 11-59% noise on
every matmul kind. The int8 route becomes viable only as rotate-then-quantize:

1. dequantize IQ -> float on the GPU, fold the block-1024 Hadamard into the weight, quantize to
   int8 with one scale per row (the rotation rides along with work already being done);
2. rotate the activation with the same Hadamard and quantize to int8 per token, fused into one
   kernel (read 142 MB float, write 35 MB int8 for the K=17408 case, about 0.5 ms);
3. cuBLAS int8 plus the scaled epilogue (74 TOPS measured in 21.3);
4. KLD gate against the current path before adoption.

That keeps the 2.4x GEMM gain mostly intact (rotation overhead is ~10%) and puts activation noise
at about 2-3% instead of 11-59%. It is more work than dequantize-and-go, and it is the only version
of this idea that can pass a quality gate.

Correcting 21.5: converting IQ values to int8 adds rounding error on top of the source, it does
not improve weight fidelity. For a row whose weights span a wide range the error is about
row_max/254 per element, which is small next to the 2.5-bit source error but is not zero.

### 22.5 Reproduce

```
set GGML_CUDA_MM_STATS=1
python _cfg.py --ctx 4096 --staged 0 --b 2048 --ub 2048 --spec none --n 1 --no-restore \
    --prompt-file prompt_2k.txt --tag mmstats
python _outliers.py
```
The hook is `ggml_cuda_mm_input_stats` in `ggml-cuda.cu`, inert unless the env var is set.

---

## 23. Rotated outlier survey: the gate passes, with two exceptions

### 23.1 Method

Same hook as 22, extended with a block-1024 Hadamard applied to src1 before measuring, so every
matmul is logged twice: raw and rotated. The transform is verified in-band by a roundtrip check
(apply twice, compare to the input): `max|diff| = 0.000e+00`, so `k_fwht_block` is exact.

### 23.2 Result

Bar is rotated ratio <= 13 (`step/RMS <= 0.1`, about 3% quantization noise).

| kind | n | K | raw median | rotated median | rotated min-max | verdict |
|---|---:|---:|---:|---:|---|---|
| ffn_down | 63 | 17408 | 259.2 | **13.78** | 8.22 - 53.51 | **FAIL** (marginal) |
| attn_output | 16 | 6144 | 68.2 | **16.68** | 12.12 - 24.03 | **FAIL** (marginal) |
| ssm_out | 48 | 6144 | 174.2 | 11.95 | 7.50 - 34.77 | PASS |
| attn_q, attn_k, attn_v | 48 | 5120 | 70.2 | 5.25 | 3.71 - 5.97 | PASS |
| attn_qkv, ssm_alpha, ssm_beta, attn_gate | 192 | 5120 | 64.9 | 5.49 | 4.92 - 7.71 | PASS |
| ffn_gate, ffn_up | 126 | 5120 | 48.0 | 6.26 | 5.21 - 10.47 | PASS |

Rotation reduces the ratio by 13x (attention) to 18.8x (ffn_down). Ten of twelve kinds land at
5-6, which is comfortable. The two exceptions are marginal, not catastrophic: 13.78 and 16.68
against a bar of 13.

A larger rotation block cannot fix ffn_down: K=17408 = 1024 x 17, and 17 is odd, so 1024 is the
largest power-of-two block that divides it. For K=6144 (ssm_out, attn_output) a 2048 block exists
(6144 = 2048 x 3) if that kind later needs help.

### 23.3 Decision: per-layer allowlist, not per-kind exclusion

ffn_down rotated ratios across its 63 layers: 17 layers <= 10, 14 in 10-13, 16 in 13-20, 16 above
20 (worst: layer 0 at 53.5, layer 3 at 47.2, then 8, 2, 54, 5, 10 at 23-30).

Excluding only the 16 layers above 20 keeps 6.0% of total FLOPs on MMQ; excluding attn_output
entirely adds 2.1%. So the safe baseline pays under 8% of FLOPs for a large margin on quality:

```
converted share 0.921 -> 0.921 / 2.4 + 0.079 = 0.463  (2.16x on the GEMM phase)
total at 2K     3.625 * (0.8 * 0.463 + 0.2)  = 2.07 s -> about 990 tok/s
```

Allow rotation overhead of about 10% on the converted matmuls and this is ~950 tok/s, at the top of
the 850-930 estimate and essentially at the 1000 target, while staying far away from the two
measured weak spots.

### 23.4 Reproduce

```
set GGML_CUDA_MM_STATS=1
python _cfg.py --ctx 4096 --staged 0 --b 2048 --ub 2048 --spec none --n 1 --no-restore \
    --prompt-file prompt_2k.txt --tag mmstats_rot
python _outliers_rot.py
```

---

## 24. Lossless leads, measured: the conversion trick is dead, the clock cap is real

### 24.1 Faster int-to-float in MMQ (hypothesis: 1.3-1.8x)

The hypothesis was that every 32-element k step ends in an int32 to float conversion, that I2F runs
at 16/clk/SM against 128 for FMA, and that this is why MMQ reaches 38.9 TOPS where cuBLAS int8
reaches 89.5.

The conversion sites are real and were confirmed by reading the code, not guessed. For our model the
sites are not the tensor core path at all: every IQ type dispatches to dp4a
(`mmq.cuh:620-653`), so the conversions are one to two per 32-element block inside
`vec_dot_q8_0_q8_1_impl` and `vec_dot_q8_0_16_q8_1_impl`:

```
int sumi = 0;  for (i < vdr) sumi = dp4a(v[i], u[i], sumi);   // 8 dp4a per 32-block
return d8_0*d8_1 * ((T) sumi);                                // 1 I2F   (IQ3_*, IQ4_XS)
sumf += d8_0[...]*sumi;                                       // 2 I2F   (IQ2_S, IQ2_XS)
```

The bound is satisfied: one 32-element block sums at most 32*127*127 = 516128, well under 2^22, so
`__int_as_float(x + 0x4B400000) - 12582912.0f` is exact. It was installed as `ggml_cuda_i2f` in
`common.cuh` (with `__fsub_rn` to block reassociation) and applied to both IQ impls, then to the four
tensor core sites that serve Q2_K and Q6_K.

| build | rep0 | rep1 | rep2 | output hash |
|---|---:|---:|---:|---|
| baseline | 553.0 | 567.0 | - | b25aee1d9f4577b0 |
| IQ impls (92% of weights) | 548.8 | 565.1 | 561.9 | b25aee1d9f4577b0 |
| IQ impls + MMA sites (Q2_K, Q6_K) | 551.2 | 565.6 | 563.2 | b25aee1d9f4577b0 |

Bit-identical, as predicted, and no faster. So the conversion is not the bottleneck for either path.

Why: the conversion sits on a pipe the dequant work does not use, so it was already hidden behind
it. This is the same conclusion the tile sweep reached from a different direction (o1, o2, o3 all
within 6%): the IQ inner loop is bound by block dequant arithmetic, which is invariant to tile shape
and to conversion rate. That also explains the 38.9 TOPS ceiling better than the conversion story:
39 TOPS is close to what dp4a can deliver at all, and cuBLAS int8 wins because it uses tensor cores
and needs no per-block dequant, not because MMQ wastes cycles on I2F.

The change is kept: exact (hash proven), no regression, three hunks, revert by deleting the helper
and the six call sites.

### 24.2 Clock cap: real, about 10%, but outside our code

Sampled `clocks.sm, clocks.max.sm, power.draw, utilization.gpu, temperature.gpu` at 0.9 s intervals
through a 2K prefill:

```
  210 MHz  idle
 1777-1920 MHz  under load, mean 1884 MHz   (max 2100)
 155.8-169.7 W  (card cap 170 W)
 100%           utilization
 58-60 C        temperature
```

The card is **power capped, not thermally capped**: 165-170 W held, 60 C, and the SM sits at 89.7%
of its 2100 MHz maximum. There is no thermal headroom to reclaim, but there is ~10% of clock
headroom at a fixed power budget, which is exactly what an undervolt curve buys. That is a driver
level change, not a code change, and it would move 567 tok/s toward ~630 if the kernel scales with
clock.

### 24.3 The fuse-and-reuse lead: smaller than stated, and unsafe in the obvious form

The old trace attributes 10-11% of GPU time to non-matmul ops (rms_norm 2.5%, SiLU gate 2.6%,
quantize_mmq_q8_1 2.1%, add 1.1%, concat 1.0%, conv 0.9%). Two of those are worth attacking, but the
ceiling is lower than 5-10%.

The survey data does confirm the redundant quantize: `attn_qkv`, `ssm_alpha`, `ssm_beta` and
`attn_gate` all show identical activation statistics at K=5120 in every layer (192 matmuls), so they
read one shared input that gets quantized four times. `ffn_gate` and `ffn_up` do the same at K=5120
(126 matmuls). Removing the redundant quantizes is worth about 6 GB of traffic for the attention
group and 8.9 GB for the FFN group, roughly 2.3% of a 2K pass.

The catch is how to do it safely. Caching the quantized activation keyed by tensor address is not
safe, because ggml's allocator gives different intermediate tensors the same address at different
points of one graph evaluation. Doing it correctly means a real graph change (a quantize node that
the matmuls take as src1), which touches every backend. 2.3% does not pay for that. Fusing the
quantize into the producer (RMS norm or SiLU-gate) hits the same graph problem.

### 24.4 Long context lead: correct, but it is a long-context win

The f16 conversion of the whole KV per ubatch makes total conversion work grow as ctx^2/ub, so a
prefill-only ubatch of 2048 instead of 512 cuts it 4x. That is real for the 200K case and irrelevant
at 2K, where the KV is only 2K tokens. Dense context checkpoints are lossless and valuable for agent
workloads, but they reduce re-prefill work, not prefill throughput.

### 24.5 Net

Nothing in the lossless list moves 2K prefill. The conversion trick is ruled out by measurement on
both paths, the fusions cap at about 2-3% and need invasive graph changes, long-context work does not
apply, and the remaining ~10% is a power limit that only an undervolt can touch. The int8 path with
rotation remains the only lever that is worth its cost.

---

## 25. Long context is the real target: measured curve, and where the time goes

### 25.1 The curve (current build, q4_0 KV, fa on, staged 0)

| ctx tokens | ub | PP t/s | seconds | note |
|---:|---:|---:|---:|---|
| 2,048 | 2048 | 565.0 | 3.62 | |
| 16,925 | 2048 | 537.7 | 31.5 | f16 KV instead: 536.4, see 25.3 |
| 33,939 | 2048 | 504.1 | 67.3 | ub 512 instead: 483.5 |
| 67,966 | 512 | 416.6 | 163.2 | ub 2048 OOMs, see 25.4 |
| 131,072 | - | ~380 projected | ~345 | if ub 2048 fit |
| 200,000 | - | ~323 projected | ~619 | if ub 2048 fit |

Fit through the two clean ub-2048 points (2K and 32K):

```
T(n) = 1.7562 ms/token * n  +  6.7023e-9 s * n^2
```

The linear term is exactly the 2K throughput, so 2K remains a valid proxy for the matmul phase. The
quadratic term is the attention phase and it crosses 25% near 100K: 11% at 32K, 33% at 128K, **43% at
200K**. The 64K point lands 8.5% above the fit, which is the ub-512 penalty, not a fitting error.

The earlier 801 s for 200K was a pre-vec-gate build. Current build projects 619 s, so ~1.29x is
already banked there.

### 25.2 Attention is near its ceiling, so the int8 win shrinks with context

Attention FLOPs at 200K, 16 full-attention layers, 24 q heads, head_dim 256, causal:

```
2 * (24 * 256 * 2) * n^2/2 * 16 = 7.86 PFLOP      at 268 s -> 29.3 TFLOPS
```

That is 57% of the 51.2 TFLOPS fp16 tensor peak, which is normal for causal flash attention. There is
at most ~1.4x in the attention kernel, worth ~14% overall.

So the int8 projection has to be recomputed for the real target. Using the 2K decomposition (80%
GEMM, 20% everything else) and 2.16x on the GEMM phase, the linear coefficient falls from 1.756 to
1.006 ms/token:

| ctx | now | after int8+rotation | gain |
|---|---:|---:|---:|
| 2K | 565 | ~994 | 1.76x |
| 128K | ~380 | ~500 | 1.32x |
| 200K | ~323 | ~426 | 1.32x |

That is the honest number for the long-context target. At 200K, 43% of the time is attention that
int8 cannot touch.

### 25.3 The KV conversion theory is dead

At 16,925 tokens the same prompt with q4_0 KV gave 537.67 t/s and with f16 KV 536.41 t/s. If the
per-ubatch q4_0 to f16 conversion of the whole cache cost anything, f16 KV would have been faster.
It is not: the conversion is free at this scale. Whatever the ubatch effect is (25.4), it is not
conversion.

### 25.4 ubatch: +4.3%, and it is attention tiling, not conversion

32K, same prompt: ub 2048 gives 504.12 t/s, ub 512 gives 483.54 t/s. The gain is attention tiling
efficiency, since conversion was just shown to be free and the matmul phase is ubatch independent in
this range.

The problem is that ub 2048 needs 142 MB class compute buffers and **does not fit at 64K**: the
server dies with `CUDA error: out of memory` in `cublas_handle` at 67,966 tokens with 9.73 GiB of
weights and 1.18 GiB of q4_0 KV. ub 512 fits and costs the 8.5% above. So beyond ~48K the ub-2048
advantage is not available without moving KV off the device.

### 25.5 What this means for the plan

Long context has three walls, in order of size at 200K:

1. attention, 43%, already at 57% of peak, so at most 1.4x on its own share;
2. matmul, 57%, where int8 with rotation gives 2.16x, so 1.32x end to end;
3. VRAM, which already blocks ub 2048 at 64K and will block 128K and 200K outright without staging.

Point 3 is the one nobody had measured: at 128K the cache alone is 2.36 GiB against 9.73 GiB of
weights on a 12 GiB card, so prefill at those sizes has to run either with a small ubatch (5-9%
penalty) or through the staged path, which so far has only been validated for decode.

---

## 26. 200K decode with MTP: GQA-packed q4_0 decode attention, replay rollback, MTP window (2026-10-20)

All measured on the code snapshot `results/.kv_cache/kv-200k-code-u2048-q4.bin` (200,368 tokens of
`results/code200k.prompt.txt`), RTX 3060, head on CUDA, q4_0 K/V, `-b 512 -ub 128`. Every new path is opt-in
and off by default. With all flags off, decode is unchanged: 5.80-5.83 t/s at L9.

### 26.1 The finding: each KV head was read 6x

The vector FA kernel runs one block per Q head. With 24 Q / 4 KV heads, each q4_0 KV head is read and dequantized
6 times. Single-layer microbench at n_kv=204800 (`tests/dec_fa_bench.cpp`): 24/4 heads 4.98 ms, 4/4 heads 0.89 ms.
For verify batches it is worse: the vec kernel takes 12.13 ms at n_q=2 and 24.13 ms at n_q=4, so an MTP verify
pass cost 2-5x a decode step. MMA on F16 K/V takes 2.46 ms at every n_q, which is pure DRAM bandwidth (838 MB at
340 GB/s). The q4_0 bytes (236 MB) bound the kernel at ~0.69 ms.

### 26.2 LLAMA_FA_DECODE_Q4=1: split-K decode attention, CUDA port of the AMD paged-decode design

`ggml/src/ggml-cuda/fattn-decode-q4.{cu,cuh}`, dispatched ahead of VEC in `fattn.cu` only when the env is set.
Gate: D=256, GQA 6, q4_0/q4_0, 1-8 query tokens, mask, no sinks/softcap/alibi, n_kv % 64 == 0.
- One block = one KV head x a range of 64-token chunks, 8 warps, ~7 splits per head (1 block/SM). A combine kernel
  merges the splits.
- Rows (query token, GQA head) are packed into 1, 2 or 4 16-row tiles. Each K/V row is read once for all 6 heads
  and all verify tokens.
- cp.async 2-stage pipeline of raw q4_0 rows. mma.sync m16n8k16 f16->f32 for S = QK^T and O += PV. K and V are
  dequantized in the fragment builders. Head dims are permuted so that each lane reads one 32-bit quant word.
- The softmax row sum uses one fixed order (16 groups of 4, butterfly) for every tile count. A row's output is
  therefore bit-identical whatever n_q is (tested nq 1..8). The vec kernel is not batch-invariant.
- 170-220 registers, no spills.

| n_kv 204800, ms/layer | nq=1 | 2 | 3 | 4 | 5 | 6-8 |
|---|---:|---:|---:|---:|---:|---:|
| vec (QUANT_KV) | 4.98 | 12.13 | 25.34 | 24.13 | 36.9 | - |
| vec + GQA pack | 3.61 | 6.62 | - | 12.43 | - | - |
| DECODE_Q4 | 0.95 | 0.95 | 1.53 | 1.53 | 1.53 | 2.6-2.8 |

Accuracy vs an fp64 CPU reference on the exact dequantized K/V: max rel 6.6e-4..1.1e-3 (vec: 1.7e-3..4.2e-3).
Teacher-forced decode on an 8,828-token code prompt (128 positions, `tests/dq4_kld.py`): top-1 128/128 and mean KLD
4.0e-4 vs vec. That is the same size as stock MMA vs vec (3.3e-4).
No-MTP effect at L9 is only 168.9 -> 161.4 ms, because that pass is DMA-bound: 9 x 225 MiB over PCIe = ~160 ms.

### 26.3 MTP state for snapshots

`tests/init_mtp_cache.py` builds the target+MTP snapshot from the `.nextn.f32` hidden-state companion in 42 s.
Validated on an 8,828-token prefix: the restored offline-initialized state and a fresh server prefill give identical
tokens and identical drafted/accepted counts. Longer coding tasks are appended after the snapshot prompt, prefilled
once at L16, and saved by the server (`kv-200k-code-task{2,3}-mtp.bin`).

### 26.4 Memory is the ceiling: pass = 17.7 ms x L + ~30 ms

The verify pass streams the staged layers once, whatever n_q is. Measured ms/pass follows `17.7 x L + ~30`
(L13: 262.8 vs 260; L11: 212-230). So every MiB that lets one more layer stay resident is worth 17.7 ms per pass.
The recurrent (GDN) rollback state was the biggest consumer: 150 MiB x (n_max+1) snapshot planes.

**LLAMA_RS_REPLAY=1** (`src/llama-memory-recurrent.{h,cpp}`, `src/models/delta-net-base.cpp`, a sync hook in
`src/llama-context.cpp`). Memory layout:
- S keeps the current state and the base state of the last ubatch (2 planes). The conv state keeps 1 plane.
- For a ubatch of at most n_max+1 tokens, the per-token k/v/gate/beta and the conv input are stored (23 MiB).
- seq_rm with a rollback replays m GDN steps from the base, with the same AR kernel, on a private stream after
  syncing the context's work (2.4 ms per call). It then copies the matching conv window, so nothing is ever pending.
  Save/restore, checkpoints and the file format are unchanged.

RS buffer: 598 MiB -> 313 MiB at n_max 3, flat in n_max. Checks (`tests/rs_replay_check.cpp`, `tests/dq4_saverestore.py`):
- The state blob, next-token logits and save/restore logits are bit-identical to the snapshot-plane path for
  accept-all, partial and reject-all at n_max 1..4.
- For n_max <= 3, both also equal a plain decode of only the kept tokens.
- Server task 2, 256 tokens: identical tokens at n_max 1-4.
- Mid-decode slot save/restore: identical tokens.

**LLAMA_MTP_WINDOW=W** (`src/llama-model.cpp`, `src/llama-kv-cache.{h,cpp}`). The MTP head's KV becomes a
sliding-window cache of about W cells, which the target's verification makes output-lossless. Restoring a longer
sequence keeps the newest cells, so the existing snapshots load. At W=8K, acceptance is unchanged vs the full
window (155/300 vs 155/297) and tokens are identical. `--spec-draft-n-ubatch 16` shrinks the MTP compute buffer.

Lowest staged L that loads:

| | n_max 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| before | 12 | 13 | 14 | 15 | 16 |
| + replay | - | 13 | 13 | 13 | 13 |
| + draft ubatch 16 | - | 12 | 12 | 12 | - |
| + MTP window 8K | - | - | 11 | 11 | 11 |

L10 does not load. An equivalent-consumption probe (L11 + 81 MiB) already fails, so the GDN fp16 state (144 MiB)
would not reach L10 alone and was not implemented. The MTP head Q4 GGUF waits for the user's own confirmation:
it is a smaller weight quant.

### 26.5 Results, 512 greedy tokens, two coding tasks

Task 2 = kv_page_lru class; task 3 = bounded MPMC queue. "best" = L11, DECODE_Q4 + RS_REPLAY + MTP_WINDOW=8192 +
draft ubatch 16. Raw data: `results/dq4/F-*.json`, `comm/files/seq34_final.txt`.

| config | task 2 t/s | task 3 t/s | tok/pass (t2/t3) | ms/pass |
|---|---:|---:|---:|---:|
| no MTP, L9, flags off | 5.80 | 5.80 | 1 | 172 |
| no MTP, L9, DECODE_Q4 | 6.09 | 6.09 | 1 | 164 |
| MTP n_max 3, L14, flags off | 5.34 | 5.91 | 2.93 / 3.26 | 548-551 |
| best, n_max 2 | 10.93 | 12.68 | 2.33 / 2.69 | 212 |
| best, n_max 3 | 11.86 | 14.93 | 2.60 / 3.26 | 218 |
| best, n_max 4 | 15.41 | 17.02 | 3.44 / 3.79 | 222 |
| best, n_max 5 | 15.89 | 17.67 | 3.66 / 4.06 | 230 |

Caveats:
- Task 2's n_max 4/5 runs drift after ~250 tokens into a `<think>` block that restates the prompt. That part is
  easy to draft, so those two numbers are inflated. Task 3 is real code for all 512 tokens.
- The honest range on code is ~12-17 t/s, with n_max 4-5 best. On 256 tokens, n_max 2-4 varied by ~10% with content.
- Speculation does not change tokens in these checks:
  - flags-off MTP == flags-off no-MTP (512/512, both tasks);
  - DECODE_Q4 n_max <= 3 == no-MTP over 256 tokens (L13-L15);
  - task 3, n_max 3 at L11 == no-MTP over all 512 tokens.
- Open items:
  - In the L11 runs, n_max 2/3 diverge from no-MTP L9 late (tokens 372-479); the cause is not found.
  - n_max >= 4 diverges early (verify batch >= 5: non-attention GEMV numerics differ by batch).
- At L11 there is no VRAM headroom for a new long prompt suffix: prefilling 226 tokens on top of the snapshot ran
  out of memory, so the task snapshots were made at L16. For interactive use with changing prompts, use L12-13.

### 26.6 Interactive numbers: fresh prompt suffix at L12/L13

Each run restores `kv-200k-code-u2048-q4-mtp.bin`, sends a new 226-token user turn (task 2 or 3) that is prefilled
on top, and then generates 512 tokens. Flags: DECODE_Q4 + RS_REPLAY + MTP_WINDOW=8192 + draft ubatch 16.

| L | n_max | task 3 t/s | task 2 t/s | suffix prefill |
|---:|---:|---:|---:|---:|
| 12 | 3 | 13.78 | 11.00 | 22.8-23.1 s |
| 12 | 4 | 15.93 | 14.32 | 23.0 s |
| 12 | 5 | 16.41 | 14.79 | 22.9-23.1 s |
| 13 | 3 | 12.89 | 10.25 | 23.2 s |
| 13 | 4 | 14.86 | 13.36 | 23.0-23.1 s |
| 13 | 5 | 15.37 | 13.83 | 22.9-23.1 s |

Drafted/accepted counts and tokens per L match the snapshot-restored L11 runs. Two configs crash with OOM during
the suffix prefill: L11 with MTP, and no-MTP L9 (DECODE_Q4, VEC_MAX_NQ=128). So the old no-MTP L9 config is also not
usable for a new prompt at 200K; it only works for continuing a restored context. **L12 is the lowest L that works
interactively.** The suffix prefill runs at ~10 tok/s (vec FA for 128-token ubatches at 200K). That is the next
bottleneck for interactive use.

Open-item check (split-K invariance): DEC_FA_INV at n_kv = 200448 / 200704 / 200960 with nq 1..5 gives bit-identical
rows. The split count depends only on n_kv and occupancy (1 block/SM for every tile variant), so dq4's partition is
ruled out as the cause of the late n_max 2/3 divergence. Still open; the remaining suspects are outside attention.

### 26.7 MTP output equals no-MTP output (exact)

Two causes made speculative greedy output differ from plain greedy output. Both were found by flag bisection on an
8.8K prompt (`comm/files/seq37_bisect.txt`):
1. **dq4 split partition.** It depended on the current padded n_kv, and a verify batch can see a larger n_kv than a
   decode at the same position. Fixed: the chunks per split now come from the whole cache tensor (the K view root),
   which is constant per context.
2. **MMVQ warp count.** MMVQ uses 4 warps for 1-4 columns but 2 for 5-8, so verify batches of 5 or more
   (n_max >= 4) reduce K in a different order. New opt-in `LLAMA_MMVQ_BATCH_INVARIANT=1` gives 5-8 columns the
   4-column warp count.

Result at 200K, L12, all flags, restored task snapshots, 512 tokens. MTP n_max 3/4/5 equals no-MTP with the same
flags, token for token, on both tasks.

| n_max | task 3 t/s (with / without MMVQ flag) | task 2 t/s (with flag) |
|---:|---:|---:|
| 3 | 14.05 / 14.12 (both exact) | 10.86 |
| 4 | 15.93 / 16.00 (without the flag: diverges at 152) | 11.59 |
| 5 | 16.65 / 16.75 (without the flag: diverges at 152) | 11.58 |

The flag costs <= 0.6% at 200K (4.5% at short context). n_max 3 is exact even without it. Task 2's earlier n_max
4/5 numbers came from a different, repetitive greedy path; on the exact path task 2 tops out at ~11.6 t/s.

### 26.8 New message at ~199K: prefill 10x faster

Profile of a 226-token turn at L12 (`GGML_FA_TIMING`): vec FA took 618 ms per layer at n_q=94 and 773 ms at
n_q=128, which is essentially the whole ubatch (8.05 s + 12.1 s). Staging DMA (~0.2 s per ubatch) and the matmuls are
minor.

`LLAMA_FA_DECODE_Q4` now also takes 9..512 query tokens:
- Row tiles of 10 query tokens (60 rows) go on grid z, using the 4-m-tile variant. The combine indexes the tiles.
- n_q <= 8 is one tile running the unchanged path: decode output rows are byte-identical to before (bench dumps), and
  t3 MTP d4 == no-MTP still holds over 512 tokens.
- `LLAMA_FA_DECODE_Q4_MAXQ=N` caps the n_q it takes (for A/B tests).

| n_kv 204800, ms/layer | n_q=16 | 64 | 94 | 128 |
|---|---:|---:|---:|---:|
| vec | - | - | - | 782.6 |
| DECODE_Q4 | 5.5 | 19.1 | 25.2 | 32.7 |

Accuracy:
- vs fp64: rel <= 1.26e-3 at n_q 9..128. The bench mask is causal per row, and n_q=128 includes the partial last
  tile of 8 tokens.
- vs vec at 204800: max rel 8.3e-4 / 9.4e-4 / 1.3e-3 for n_q 9 / 94 / 128.
- End to end, prefill vec vs prefill dq4 (both decode with dq4), task 3: greedy identical for 479 of 512 tokens.
  Over those 479 positions, top-1 agrees 479/479, with mean KLD 1.6e-3 on the top-20 support (first token 8.8e-3).
  vec is the less accurate of the two, since it quantizes Q to q8_1.

End to end at L12 with the full recommended config:

| turn | prefill before | prefill now | decode |
|---|---:|---:|---|
| 226 tokens | 22.8 s | 2.2-2.4 s | d4 15.5 t/s; no-MTP 4.57 t/s |
| 1,184 tokens | ~110 s (est. 10 ubatches x ~11 s) | 8.7-9.1 s | d4 15.64 t/s, tokens == no-MTP 512/512 |

`-ub 256` and `-ub 512` still OOM (a 178.5 MiB pool allocation in the batched path), so `-ub 128` stays.
L11 and no-MTP L9 also still OOM on a new turn for the same reason. L12 is the interactive floor.

Recommended 200K decode command (restore a target+MTP snapshot through the slot API, then chat). Add
`LLAMA_MMVQ_BATCH_INVARIANT=1` so that output is identical to non-speculative decoding at n_max >= 4:

```
LLAMA_KV_STAGED_LAYERS=12 LLAMA_KV_STAGE_SLOTS=2 LLAMA_FA_VEC_QUANT_KV=1 LLAMA_FA_VEC_Q4_COLS=2
LLAMA_FA_VEC_MAX_NQ=128 LLAMA_FA_DECODE_Q4=1 LLAMA_RS_REPLAY=1 LLAMA_MTP_WINDOW=8192 LLAMA_MMVQ_BATCH_INVARIANT=1
llama-server -m Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf -c 204800 -ngl 99 -fa on -ctk q4_0 -ctv q4_0
  -b 512 -ub 128 -np 1 --fit off --no-context-shift --jinja -ot "^output\.weight$=CUDA0"
  --spec-type draft-mtp --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max 4
  --spec-draft-n-ubatch 16 --slot-save-path results/.kv_cache/
```

(`tests/dq4_e2e.py` runs exactly this.) Not implemented, by decision: the lossy CPU-retrieval KV approach used by
ninfer ("KVMem", ~28K tokens of history in VRAM; it fails a 250K needle test). It is not comparable to exact
full attention.

### 26.9 MMVQ "unpack once" for 2-8 columns: bit-exact, but no speedup

The idea: `mmvq.cu` calls `vec_dot_q_cuda` once per column j, so a weight block's unpack (grid lookup, sign
assembly, scale) is redone for every column of a speculative verify batch. Opt-in
`LLAMA_MMVQ_UNPACK_ONCE=1` unpacks the block once into registers (`mmvq_unpack`) and runs a per-column dp4a
(`mmvq_dot_unpacked`) against each column's q8_1, keeping each column's accumulate order unchanged. Implemented
for IQ3_XXS and IQ3_S, 2..8 columns.

Coverage, by weight bytes of this model (`tests/gguf_type_bytes.py`): IQ3_S 21.7%, IQ3_XXS 19.6% (41.3%
together), IQ2_S 19.2, IQ4_XS 13.9, Q2_K 6.0, IQ2_XS 5.9, IQ2_XXS 5.4, Q4_K 3.7, Q6_K 3.3, rest <1.5.

Bit-exactness, `tests/mmvq_cols_bench.cpp` writing f32 outputs at ncols 1, 2, 4, 5, 8 for 7 types and both main
shapes: 140 dumps, all bit-identical between the per-column and the unpack-once kernel (`tests/cmp_mmvq_dumps.py`;
the same-binary A/B is in `results/mmvq_cols/old2_*` vs `new2_*`).

Per-tensor time, median of 50 graph runs of 16 matmuls, us per matmul:

| type | shape | nc=1 | nc=2 | nc=4 | nc=5 | nc=8 |
|---|---|---:|---:|---:|---:|---:|
| IQ3_XXS | 5120x17408 old | 151.0 | 170.0 | 218.8 | 236.3 | 321.2 |
| IQ3_XXS | 5120x17408 new | 151.4 | 170.1 | 219.5 | 231.3 | 326.1 |
| IQ3_S | 5120x17408 old | 159.8 | 180.7 | 232.5 | 262.3 | 349.9 |
| IQ3_S | 5120x17408 new | 159.7 | 178.9 | 234.7 | 245.9 | 347.1 |

Mean change over the two IQ3 types / over all 7 types: nc=1 -0.4 / -0.5%, nc=2 -0.0 / +0.0%, nc=4 +0.2 / +0.3%,
nc=5 **-3.0** / -0.7%, nc=8 +0.3 / +0.3%. The five types without the new path scatter within -0.6..+0.7%, so
that is the noise band; the only real effect is ~-3% at 5 columns on the covered types, i.e. under 1.2% of
MMVQ time once weighted by their 41% byte share.

End to end, 8,828-token code prefix, ctx 16384, all layers resident, 512 greedy tokens, full exact-output flags
(`LLAMA_FA_VEC_MAX_NQ=128 LLAMA_FA_DECODE_Q4=1 LLAMA_RS_REPLAY=1 LLAMA_MTP_WINDOW=8192
LLAMA_MMVQ_BATCH_INVARIANT=1`, `--spec-draft-n-ubatch 16`, restored `dq4-short-q4-mtp.bin`):

| n_max | old t/s | new t/s | delta | accepted/drafted |
|---:|---:|---:|---:|---:|
| none | 19.925 | 19.947 | +0.1% | - |
| 3 | 39.001 | 38.867 | -0.3% | 364/438 (0.831) |
| 4 | 37.947 | 37.528 | -1.1% | 386/500 (0.772) |
| 5 | 35.003 | 34.826 | -0.5% | 399/559 (0.714) |

All 8 runs produce the same 512 tokens, and the same first-64 hash as the seq 37 bisection
(`17026ab43c3dcb10`): the flag changes nothing observable except its own (absent) speedup.

Why there is no gain: at 1 column MMVQ is weight-load bound and the unpack hides behind the load latency; as
columns are added the weight read amortizes, so the unpack should surface - and it does, barely, at 5 columns.
At 8 columns it does not surface at all (-0.0..+1.5%), because there the warp count drops from 4 to 2 and
latency dominates again. Conclusion: unpacking is not on the critical path at these column counts on this part.
The flag stays opt-in and off by default; the code is left in place, not reverted. Its switch statements for
ncols 2..8 still carry a dead `const bool inv = false; if (inv) {} else {}` pair that should be collapsed if
the path is kept.

### 26.10 DFlash vs MTP at 8K

Drafter `Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf` (535 MiB; `tests/gguf_meta.py`): 5 layers, 5120 embd, 17408 ffn,
`dflash.block_size` 8, `selector_top_k` 16 (so it is a DFlash2: it reads its selector lattice from
`h_nextn` and never consumes raw logits), `target_layers` [6, 20, 34, 48, 62], `sliding_window` 2048 on all 5
layers, declared context 262144.

Server args that work: `--spec-type draft-dflash --spec-draft-model <drafter> --spec-draft-ngl 99
--spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max N`. No snapshot is needed: a fresh
prefill is enough, because the drafter is fed target layer inputs extracted from the same prefill. `n_max` is
clamped to `block_size - 1` = 7 with a warning ("requested draft size (n_max=12) exceeds the trained block
size 8 -- clamping to 7"); `--spec-draft-n-max 12` then behaves exactly like 7.

Same setup as 26.9, all runs a fresh prefill, 512 greedy tokens, one rep each (`tests/dflash_8k.ps1`):

| spec | n_max | t/s | ms/tok | PP t/s | accepted/drafted | acc | tokens/step | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | - | 20.58 | 48.50 | 490.0 | - | - | - | 10,057 |
| MTP | 4 | 37.79 | 26.41 | 447.8 | 387/492 | 0.787 | 4.13 | 10,603 |
| DFlash | 3 | 38.67 | 25.81 | 452.1 | 362/445 | 0.813 | 3.44 | 10,813 |
| DFlash | 4 | **39.17** | 25.48 | 453.7 | 386/496 | 0.778 | 4.10 | 10,817 |
| DFlash | 5 | 37.30 | 26.76 | 452.9 | 401/544 | 0.737 | 4.65 | 10,819 |
| DFlash | 6 | 36.26 | 27.53 | 453.2 | 411/592 | 0.694 | 5.12 | 10,823 |
| DFlash | 7 | 34.95 | 28.56 | 451.7 | 419/631 | 0.664 | 5.57 | 10,827 |

DFlash beats MTP by 3.7% at the same draft length (39.17 vs 37.79), its best length is also 4, and every run is
token-for-token identical to the no-speculation baseline (512/512, `tests/cmp_tokens.py`). Acceptance falls
steadily with draft length, so tokens/step rises while t/s falls: the extra drafted positions cost more than
they earn past 4. Prefill pays 7.4% for the target feature extraction (490.0 -> 453.7 t/s); MTP pays 8.6%.

Exactness never breaks, and cannot with this drafter: `block_size` 8 caps the verify batch at 8 columns, which
is exactly MMVQ's limit, so the >8-column MMQ path is unreachable. For contrast, the same prompt with the
batch-invariant flag removed diverges at token 152 for MTP n_max 4 (`M8k-d4-nobinv`) and not at all for
n_max 3, which is the 5-column warp-count effect of 26.7; the flag also costs 8.3% at 8K there (41.38 ->
37.95 t/s).

Memory, and the 200K question. Peak device memory with the same prompt at ctx 16384 and 32768:

| ctx | no spec | DFlash d4 | MTP d4 | DFlash - no spec |
|---:|---:|---:|---:|---:|
| 16384 | 10,057 | 10,817 | 10,603 | +760 |
| 32768 | 10,349 | 11,111 | 10,895 | +762 |

The target's KV costs ~292 MiB per 16K tokens, but the drafter's marginal cost is flat at 760 MiB: its KV is
bounded by the 2048-token sliding window, so it does not grow with context. At 200K the drafter still costs
~0.76 GiB (535 MiB of weights plus ~225 MiB of windowed KV and buffers), and DFlash needs ~216 MiB more than
MTP d4, so a 200K DFlash config likely needs one more staged layer than the working MTP L12 setup (L13).
Not yet run at 200K.

Exactness is per context size: the 16K and 32K runs of the same prompt produce different tokens from each other
(`027d2bab...` vs `7d46062f...`), which is the split-partition ordering of 26.7 item 1, not speculation.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,
`tests/summ_spec.py`; data in `results/mmvq_cols/` and `results/dq4/{M8k,D8k,D32k}-*.json`.

## 27. T1: where an MTP decode step goes (8K, 64K, 128K)

Measured 2026-10-20 on IQ3_XXS-mtp, exact-output flags (`LLAMA_FA_DECODE_Q4=1`,
`LLAMA_FA_VEC_MAX_NQ=128`, `LLAMA_RS_REPLAY=1`, `LLAMA_MTP_WINDOW=8192`,
`LLAMA_MMVQ_BATCH_INVARIANT=1`), restored snapshots, 128 greedy tokens. nsys was unusable
(`nsys launch`/`profile` lose the agent connection even for `cmd /c echo`), so the fork's
env-gated CUDA-event timers were extended to decode-sized batches: `GGML_MM_TIMING` now times
`n <= 8` instead of only prefill (`ggml-cuda.cu`, instrumentation only), and `GGML_FA_TIMING`
already wrapped all flash-attention calls. Timer runs use `GGML_CUDA_DISABLE_GRAPHS=1` because
event sync inside graph capture is illegal. Clean runs use graphs on.

### 27.1 Clean throughput and step cost

| config | t/s | ms/token | verify steps | tok/step | ms/step | drafted | acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K MTP d4 base | 39.32 | 25.23 | 30 | 4.27 | 107.7 | 120 | 0.808 |
| 8K MTP d4, `--samplers temperature` | 39.13 | 25.35 | 30 | 4.27 | 108.2 | 120 | 0.808 |
| 8K MTP d4, + `--backend-sampling` | 40.10 | 24.74 | 30 | 4.27 | 105.6 | 120 | 0.808 |
| 8K MTP d4, graphs off | 38.09 | 26.05 | 30 | 4.27 | 111.1 | 120 | 0.808 |
| 8K no-spec | 20.86 | 47.55 | - | 1 | 47.6 | - | - |
| 8K no-spec, backend sampling | 21.05 | 47.14 | - | 1 | 47.1 | - | - |
| 64K MTP d4 (budget 1678) | 37.64 | 26.36 | 30 | 4.27 | 112.5 | 120 | 0.808 |
| 64K no-spec (budget 2238) | 19.18 | 51.72 | - | 1 | 51.7 | - | - |
| 128K MTP d4 (budget 1422) | 29.32 | 33.84 | 30 | 4.27 | 144.4 | 120 | 0.808 |
| 128K no-spec (budget 1982) | 13.51 | 73.43 | - | 1 | 73.4 | - | - |

All ten runs produced the same 128 tokens (`sha256` prefix `d0fb7092955c105a`). CPU sampling is not
the gap: the minimal chain is -0.5%, backend sampling +2.0% (2.1 ms/step). CUDA graphs are used and
worth 3.1% at 8K; the server reported `graphs reused = 29` for 30 verify steps (one build, one realloc),
so verify and draft both replay captured graphs.

### 27.2 8K step anatomy

Verify pass (n=5, all 64 layers, GPU kernel-time sum): **81.7 ms** last pass, 81.9 ms over two passes.
FFN 52.9-54.4 (65%), attention weights 18.0-18.7 (22%), GDN 5.9-6.1 (7.5%), LM head 2.69 (3.3%),
flash-attn 2.15 (2.6%). Per tensor: `ffn_down` 0.319, `ffn_up` 0.317, `ffn_gate` 0.255,
`attn_qkv` 0.186, `output.weight` 2.69 ms.

Each draft step is one MTP layer + one full LM head: `output.weight` 1.99 ms and the timed MTP block
0.39 ms; the n=1 timer misses the block's `ffn_down` and most `attn_output` matmuls, so from weight
bytes it is ~1.2 ms, i.e. ~3.2 ms/draft. Four drafts ~13 ms of the 107.7 ms step, and 8.0 ms of that
is the four LM heads (62%). The draft does use the full IQ4_XS `output.weight` (644 MiB) every step:
blk.64 has only `nextn.shared_head_norm`, no output tensor of its own.

The remaining ~13 ms/step is host/graph/sync; sampler A/B shows <1 ms of it. Confirmed by the n=1
path: logged no-spec matmuls 30.3 ms + the unlogged `ffn_down` (~8 ms from the MMVQ microbench) +
`attn_output` (~1.4 ms) ~= 39.7 ms of the 47.6 ms token.

`LLAMA_MMVQ_BATCH_INVARIANT`: verify matmul sum 81.9 -> 73.1 ms when removed (-8.8 ms, 10.8%;
including FA 84.1 -> 75.3). FFN 54.4 -> 47.2, attention weights 18.7 -> 17.6, GDN 6.1 -> 5.8,
LM head 2.69 -> 2.55. So the e2e ~8.3% is the 5-column MMVQ FFN path (down/up), not only the warp count.

### 27.3 Why 64K -> 128K costs ~30 ms/step

64K (budget 1678) keeps the whole KV resident: verify 88.2 ms, 16/16 FA logged, FA 7.80 ms.
All-resident attention therefore grows only 8K -> 64K -> 128K by ~5.7 ms and then ~8 ms, as expected.

128K at budget 1422 **stages 9 of 16 KV layers** (evenly spaced ordinals 1,3,5,7,8,10,12,14,15 ->
layers 7,15,23,31,35,43,51,59,63), two device slots x 144.56 MiB. `GGML_KV_STAGE_PROF=1` reports
DMA 102-123 ms per eval and attention 6-17 ms; the staged layers' attention bypasses the FATIME hook
(direct `ggml_cuda_flash_attn_ext` call), which is why only the seven resident layers (3,11,19,27,39,
47,55) appear in the timer log. The 64K -> 128K step growth (112.5 -> 144.4 ms, +31.9 ms) is therefore
mostly the 1.27 GiB/eval staged-KV DMA over PCIe (about 95 ms at 13.4 GB/s, partly overlapped), not the
q4 attention kernel. The no-spec step shows it more clearly: 51.7 -> 73.4 ms, because there is less
compute to hide the same DMA.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,
data in `results/prof/t1c_*.json`, `t1_*_clean.json`; summary in `comm/files/seq45_t1_summary.txt`.

## 28. T2: DFlash at 128K and 200K

DFlash has no `get_state`/`set_state` in `common/speculative.cpp`, so a server-saved snapshot cannot
rebuild its 2048-token drafter window. Workaround: prefill a long base once and save it, then for each
run restore the base and send the full prompt; the server prefills only the suffix (~3.5K tokens) with
DFlash active, which rebuilds the drafter window. Base sizes: 127,542 tokens at 128K, 196,916 at 200K;
suffix 3,523 / 3,453 tokens. The 200K runs use ctx 204,800 because the 200K prompt at ctx 200,704
leaves only 335 tokens.

| config | KV budget | t/s | verify steps | tok/step | ms/step | drafted | accepted | acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K DFlash d4 | 1000 | 24.45 | 125 | 4.10 | 167.2 | 500 | 385 | 0.770 |
| 128K DFlash d5 | 1000 | 27.58 | 108 | 4.74 | 171.6 | 540 | 403 | 0.746 |
| 128K DFlash d6 | 1000 | **27.72** | 99 | 5.17 | 186.2 | 591 | 412 | 0.697 |
| 128K DFlash d7 | 1000 | 26.86 | 94 | 5.45 | 202.4 | 654 | 417 | 0.638 |
| 128K MTP d4 | 1000 | 22.86 | 132 | 3.88 | 169.4 | 527 | 379 | 0.719 |
| 128K MTP d4 | 1422 | 29.32 | 30 | 4.27 | 144.4 | 120 | 97 | 0.808 |
| 128K no-spec | 1982 | 13.36 | - | 1 | 74.7 | - | - | - |
| 200K DFlash d4 | 1000 | 16.02 | 116 | 4.41 | 275.0 | 464 | 394 | 0.849 |
| 200K DFlash d5 | 1000 | 18.15 | 101 | 5.07 | 278.7 | 501 | 410 | 0.818 |
| 200K DFlash d6 | 1000 | 19.53 | 92 | 5.57 | 284.5 | 552 | 419 | 0.759 |
| 200K DFlash d7 | 1000 | **21.06** | 84 | 6.10 | 288.9 | 588 | 427 | 0.726 |
| 200K MTP d4 | 1000 | 15.55 | 118 | 4.34 | 278.6 | 472 | 392 | 0.831 |
| 200K no-spec | 1982 | 5.50 | - | 1 | 181.5 | - | - | - |

The 128K MTP 1422 row is the 128-token run from 27.1 (30 steps); the other rows are 512-token runs. Both are steady-state and share the same snapshot.

Exactness: all four DFlash lengths produce 512 tokens identical to the no-spec reference at 128K and
at 200K (`d1838211...` at 128K, `2815707d...` at 200K). The drafter never changes the target's tokens.

At 200K, longer drafts are nearly free: d4 -> d7 raises the step from 275.0 to 288.9 ms (+5%) while
tok/step goes 4.41 -> 6.10, so d7 is +31% over d4. At 128K the step grows faster (167 -> 202 ms, +21%),
so d6 is the best there.

Fit limits: at 128K DFlash OOMs at budget 1422 even with `-ub 64`; only budget 1000 (more KV layers
staged) fits, while MTP at 1422 fits. Thus at 128K the best feasible MTP config (1422, 29.32) still
beats the best DFlash config (1000, 27.72), but at equal budget 1000 DFlash is +21% (27.72 vs 22.86).
At 200K DFlash at budget 1000 beats MTP at the same budget by 35% (21.06 vs 15.55), and beats the
chart's MTP 200K number (15.68).

## 29. MMQ vs MMVQ at decode column counts

`tests/mmvq_cols_bench` (40 reps, 9 types x 4 real shapes x ncols 1,2,4,5,8; MMQ forced with
`GGML_CUDA_PREFER_MMQ=1`, which routes to the smallest MMQ tile J=8):

| ncols | MMQ/MMVQ mean | range |
|---:|---:|---:|
| 1 | 1.50 | 0.13-2.69 |
| 2 | 1.51 | 1.17-2.16 |
| 4 | 1.10 | 0.80-1.50 |
| 5 | 1.03 | 0.80-1.34 |
| 8 | 0.71 | 0.52-0.94 |

At 5 columns MMQ is only neutral: FFN up 0.91, down 1.11, qkv 0.94, attn_output 1.14. It is not near
the MMVQ 1-column time (MMQ@5 / MMVQ@1: up 1.47, down 1.96, qkv 1.31, attn_output 0.67). At 8 columns
MMQ is 26-48% faster, so the MMA tile only pays once the verify batch reaches 8.

MMQ J=8 is column-invariant: for every type/shape, column j's output is bit-identical across ncols
1,2,4,5,8 (36/36 combos, `tests/cmp_cols_columns.py`). An exact mode through MMQ would therefore be
exact by construction, but it is not free: at n=1 MMQ is ~1.5x MMVQ overall (FFN up 1.2-1.6x, down
1.4-2.3x), so no-spec decode would pay 50%+ for it.

n=1 per-op breakdown (no-spec, one token, GPU kernel times): ffn gate/up 14.83 ms, attention weights
10.25, GDN 4.37, LM head 1.99, flash-attn 1.57, rot 0.16; the n=1 timer misses ffn_down (~8.0 ms from
the MMVQ bench) and most attn_output (~1.1 ms), giving ~42.3 ms of the clean 47.6 ms token. Forcing
MMQ does not make those ops appear, so the gap is not the MMVQ path (scheduler/graph replay).

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,
`comm/files/seq46_t2_mmq.txt`.

## 30. T5: all-MMQ decode (PREFER_MMQ) and its exactness limit

Config: `GGML_CUDA_PREFER_MMQ=1` (MMQ J=8 for every n<=8 matmul) + `LLAMA_MMVQ_BATCH_INVARIANT=1`,
exact-output flags, base+suffix restores, 512 greedy tokens. The old prefer-MMQ branch sent an
MMQ-unsupported type (IQ1_M ffn_up) to cuBLAS; added an explicit MMVQ fallback and an `GGML_MM_ROUTE`
log. Route census at n=1: every MMQ-capable type takes `path=mmq`; IQ1_M takes `path=mmvq-fallback`;
none take cuBLAS.

| config | t/s | steps | tok/step | ms/step | drafted | acc |
|---|---:|---:|---:|---:|---:|---:|
| 8K no-spec | 15.66 | - | 1 | 63.7 | - | - |
| 8K MTP d1 | 23.07 | 261 | 1.96 | 84.9 | 261 | 0.954 |
| 8K MTP d2 | 30.67 | 182 | 2.81 | 91.6 | 364 | 0.904 |
| 8K MTP d4 | 40.12 | 124 | 4.13 | 102.7 | 496 | 0.780 |
| 8K MTP d5 | 42.29 | 110 | 4.65 | 109.8 | 547 | 0.733 |
| 8K MTP d6 | 43.28 | 102 | 5.02 | 115.7 | 612 | 0.667 |
| 8K MTP d7 | 45.07 | 94 | 5.45 | 120.6 | 657 | 0.635 |
| 8K DFlash d4 | 42.58 | 125 | 4.10 | 96.0 | 499 | 0.774 |
| 8K DFlash d5 | 47.25 | 110 | 4.65 | 98.3 | 546 | 0.734 |
| 8K DFlash d6 | 51.09 | 100 | 5.12 | 100.0 | 600 | 0.685 |
| 8K DFlash d7 | **54.34** | 93 | 5.51 | 101.1 | 649 | 0.644 |
| 64K DFlash d7 | 38.57 | 117 | 4.38 | 113.2 | 810 | 0.485 |
| 64K DFlash d6 | 37.01 | 123 | 4.16 | 112.2 | 732 | 0.529 |
| 64K no-spec | 14.56 | - | 1 | 68.5 | - | - |
| 128K DFlash d6, budget 1200 | 35.47 | 94 | 5.45 | 153.2 | 564 | 0.739 |
| 128K DFlash d6, budget 1300 | **35.81** | 94 | 5.45 | 151.8 | 564 | 0.739 |
| 200K DFlash d7, budget 1000 | 17.58 | 104 | 4.92 | 279.5 | 723 | 0.563 |

The 128K budget sweep found 1300 as the largest feasible (1400 OOMs; 1200 also fits). At 8K the win over
the previous best is +34-38% (DFlash d7 54.34 vs 39.32 MTP d4 / 40.52 DFlash d4); at 128K it is +22%
(35.81 vs 29.32). At 64K it is +2.5% (38.57 vs 37.64), and at 200K it is worse (17.58 vs 21.06) because
the PCIe DMA dominates and the n=1 draft steps are slower under MMQ.

Exactness: every spec run is identical to every other spec run (8K `a4f10fc6`, 64K `08d45794`,
128K `7f8469f9`), but the no-spec MMQ reference differs at token 151 (8K) and 203 (64K). The divergence
is width-independent: MTP d1 (2 columns), d2, d4, d7 and DFlash all differ at the same token. The MMQ
matmuls, including the 248,320-row IQ4_XS output head, are column-invariant (0 bit diffs), so the break
is not the matmul path. The no-spec MMQ run is reproducible, and a fresh full-prefill no-spec run differs
from the base+suffix no-spec at token 390, so the reference is prefill-scheme-sensitive; but spec and
no-spec in the table use the same scheme. The remaining explanation is a non-matmul batch effect
(attention/GDN) whose ULP differences are exposed by the MMQ trajectory: MMVQ mode happened not to flip
at 151, MMQ mode does. "Exact by construction" from MMQ column invariance alone is therefore false
end-to-end; making all-MMQ exact needs batch-invariant attention/GDN too.

## 31. T4: IQ3_S best configs (output head on GPU)

`D:\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, MTP d4, output.weight on CUDA0, all 16 KV layers staged,
`-ub 32`, PREFER_MMQ off:

| ctx | CPU GDN layers | t/s (128 tok) | chart config |
|---:|---:|---:|---:|
| 8K | blk.0-1 (2) | 23.38 | 19.17 |
| 64K | blk.0-2 (3) | 15.85 | 15.05 |

Fewer CPU layers OOM (1 or 0 at 8K; 2 at 64K); the chart used 3 CPU layers and the output head on CPU.
Keeping output.weight on the GPU is worth ~+22% at 8K and ~+5% at 64K.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,

## 32. T5 addendum: the n=1 exactness break is the fused mul_mat_vec_q path

With `GGML_CUDA_PREFER_MMQ=1`, the n=1 decoder still routed ffn_down, attn_output and some gate/up
through the fused `ggml_cuda_mul_mat_vec_q` paths in `ggml_cuda_try_fuse`, which bypass the routing and
use MMVQ; at n>=2 those nodes are unfused and use MMQ. That is the exact split: all spec widths agree
with each other, only n=1 differs. `GGML_MM_ROUTE` diff at n=1 (fusion on -> off) shows the fused ops:

| op | K | fusion on | fusion off |
|---|---:|---:|---:|
| ffn_down IQ3_XXS | 17408 | 0 | 68 |
| ffn_down IQ3_S | 17408 | 0 | 64 |
| ffn_down IQ2_S | 17408 | 0 | 60 |
| attn_output IQ3_S | 6144 | 104 | 140 |
| gate/up IQ2_S | 5120 | 168 | 208 |
| gate/up IQ3_XXS | 5120 | 192 | 224 |

So ffn_down + residual add (all 64 layers), attn_output + residual add (16 layers) and some gate/up + GLU
are fused at n=1. `GGML_CUDA_DISABLE_FUSION=1` makes every n<=8 matmul take the MMQ path, and all-MMQ then
becomes exact:

| config | t/s | 512 tokens identical to no-spec |
|---|---:|---|
| 8K no-spec | 13.06 | - |
| 8K MTP d4 | 39.52 | yes |
| 8K DFlash d7 | 53.09 | yes |
| 64K no-spec | 12.28 | - |
| 64K DFlash d7 | 37.86 | yes |
| 128K no-spec | 9.51 | - |
| 128K DFlash d6, budget 1300 | 34.73 | yes |

Cost of disabling fusion: 1.5% (MTP d4), 2.3% (8K DFlash d7), 1.8% (64K d7), 3.0% (128K d6); the
no-spec path pays 17% (15.66 -> 13.06), which does not matter in spec mode. The proper fix is one line of
routing: make the fused paths honor PREFER_MMQ (or skip fusion when it is set), which should recover the
~2% and keep exactness.

Exact all-MMQ vs the previous best MMVQ mode: 8K DFlash d7 53.09 vs 39.32 (+35%), 64K d7 37.86 vs 37.64
(+0.6%), 128K d6 34.73 vs 29.32 (+18%). 200K stays MMVQ + DFlash d7 (21.06).

## 33. PREFER_MMQ now skips only the mul_mat_vec_q fusions; exact per-context configs

The `GGML_CUDA_PREFER_MMQ` routing now skips exactly the `mul_mat_vec_q` fusions (ffn_down+add,
attn_output+add, gate/up+GLU, shared expert) and keeps every other fusion. `ggml_cuda_should_fuse_mul_mat_vec_q`
returns false when the flag is set, with an optional `GGML_MM_ROUTE` line (`path=skip-fused-mmvq`);
when the flag is unset the helper behaves and logs exactly as before. Default mode is unchanged:
8K MTP d4, 128 tokens -> 39.82 t/s, token hash `d0fb7092955c105a` (pre-patch 39.32, same hash).

After the fix, `GGML_CUDA_PREFER_MMQ=1` alone is exact; no `DISABLE_FUSION` needed. 512 greedy tokens,
no-spec vs spec in the same mode:

| ctx | config | t/s | exact | previous best MMVQ |
|---:|---|---:|---|---:|
| 8K | all-MMQ, DFlash d7 | 54.19 | yes | 39.32 MTP d4 (+38%) |
| 64K | all-MMQ, DFlash d7, budget 1678 | 38.51 | yes | 37.64 MTP d4 (+2.3%) |
| 128K | all-MMQ, DFlash d6, budget 1300 | 35.56 | yes | 29.32 MTP d4 (+21%) |
| 200K | MMVQ, DFlash d7, budget 1000 | 21.06 | yes (T2) | 15.55 MTP d4 (+35%) |

Crossover: all-MMQ wins clearly at 8K and 128K, is nearly flat at 64K, and appeared to be **worse at
200K** (17.58 vs 21.06). Sections 38-42 resolved this: both modes are exact at 200K, and over five task
prompts MMQ is +4.7% (4/5 wins, equal acceptance, step 281.9 vs 292.4 ms). So **MMQ is the right mode at
200K too**, and the 200K row below is superseded by section 42. **160K was not measured**; the only 160K decode data is the
IQ3_S chart point (6.69 t/s, different model/config). Measuring 160K would need a fresh 160K base.

### Recommended exact configs

Common server command (per context; change `-c`, the budget, and `--spec-draft-n-max`):
```
llama-server.exe -m Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf -c <ctx> -ngl 99 -fa on -ctk q4_0 -ctv q4_0
  -b 512 -ub 128 -np 1 --fit off --no-context-shift --jinja --port <port>
  --slot-save-path results\.kv_cache -ot "^output\.weight$=CUDA0"
  --spec-type draft-dflash --spec-draft-model Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf --spec-draft-ngl 99
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-max <7|6> --spec-draft-n-ubatch 16
```
Common environment:
```
LLAMA_FA_DECODE_Q4=1 LLAMA_FA_VEC_MAX_NQ=128 LLAMA_RS_REPLAY=1 LLAMA_MTP_WINDOW=8192
LLAMA_MMVQ_BATCH_INVARIANT=1 LLAMA_MMVQ_UNPACK_ONCE=0
```
Per context:
- 8K: ctx 16384, `--spec-draft-n-max 7`, `GGML_CUDA_PREFER_MMQ=1`, no KV budget (all resident).
- 64K: ctx 65792 (or 66560 for 512 tokens), `--spec-draft-n-max 7`, `GGML_CUDA_PREFER_MMQ=1`,
  `LLAMA_KV_GPU_BUDGET_MIB=1678` (all resident).
- 128K: ctx 131584, `--spec-draft-n-max 6`, `GGML_CUDA_PREFER_MMQ=1`,
  `LLAMA_KV_GPU_BUDGET_MIB=1300` (largest feasible; 1400 OOMs).
- 200K: ctx 204800, `--spec-draft-n-max 7`, **`GGML_CUDA_PREFER_MMQ=1`** (section 42: +4.7% over five
  prompts, exact), `LLAMA_KV_GPU_BUDGET_MIB=1000`.

DFlash has no get_state/set_state, so a saved snapshot cannot rebuild its 2048-token window. For
repeatable runs use the base+suffix recipe: prefill a long base once and save it; then restore it and
send the full prompt, so the server prefills only the ~3.5K suffix with DFlash active. The 200K prompt
needs ctx 204800 to leave room for 512 decoded tokens. A fresh full prefill also works and gives the
same tokens, at the cost of the full prefill time.

MTP remains available as the fallback when DFlash is not wanted: 8K 40.31, 64K ~37.6, 128K ~29.3,
200K ~15.6 t/s in the same modes.

## 34. Step-1 gate for a small-batch (1-16 column) matmul kernel: the bar, and why it is not worth it

Measured 2026-10-20 with a new standalone microbench (`tests/mmsb_bench.cu`, no model code): 4 weight
types x 5 real shapes x ncols 1,2,4,8,12,16, kernels: STREAM (contiguous), LOAD (planned 16-row
tile pattern), UNPACK (load + full unpack to int8, no mma), and seven full kernels (unpack +
`mma.m16n8k32.s8` + fp32 block scaling). All full kernels are bit-exact per column for ncols 1..16
and match ggml's own dequantizers + a double-precision dot to <= 1.6e-7.

Same tensors, same machine, in-tree kernels, x of each tensor's own measured stream floor
(341 GB/s peak, 280-341 GB/s on the smaller tensors; `results/ref/mmvq.txt`, `results/ref/mmq.txt`,
head from `results/prof/cols/head[q|v].txt`):

| type | tensor | MMVQ@1 | MMVQ@8 | MMQ@8 | best new kernel |
|---|---|---:|---:|---:|---:|
| IQ3_S | ffn_up 5120x17408 | 1.33 | 2.92 | **2.07** | 3.3-3.9 |
| IQ3_XXS | ffn_up 5120x17408 | 1.42 | 2.97 | **2.04** | 3.3-4.5 |
| IQ2_S | ffn_up 5120x17408 | 1.21 | 2.78 | **1.71** | 3.2-5.7 |
| IQ4_XS | ffn_up 5120x17408 | 0.85 | 1.69 | **1.08** | 2.3-3.5 |
| IQ4_XS | output head 5120x248320 | 1.00 | 2.01 | **1.23** | - |

The load path is not the problem (planned pattern 1.04-1.47x of a contiguous stream). The unpack is:
measured alone, with no mma, it costs 2.4-3.2x the floor for IQ3_S/IQ3_XXS, 2.7-4.0x for IQ2_S and
1.2-2.1x for IQ4_XS (ablations: grid lookups ~40 us of a 329 us pass, sign-mask table ~8 us, the
remaining ~180 us is the per-lane metadata byte scatter - 18 LDG/step with 16 sectors each).

The decisive number is the bar, not our kernels: **IQ4_XS (13.9% of the weight bytes, including the
644 MiB output head) is already at 1.08-1.40x with MMQ@8, and the head is at 1.23x** - there is
nothing to win there. The three grid types (60.5% of the bytes) sit at 1.71-2.07x, so the maximum
headroom is ~1.6x on 60% of the bytes; MMQ pays the same per-byte unpack cost and is already close to
unpack-bound, and all seven of our designs are 1.2-2.4x *worse* than MMQ@8 (best: v1, the plain
direct-register unpack, at 3.3-3.9x; the MMQ-shaped versions v2/v6 at 4.4-6.6x; a 32-row warp tile
v7 at 5.1-5.3x).

Verdict: the "verify pass at n=8 within 1.2-1.4x of the weight-read floor" target is **not reachable**
for IQ3_S/IQ3_XXS/IQ2_S by this route, and unnecessary for IQ4_XS. A perfect 1.0x-floor kernel would
take the 8K verify step from ~101 ms to ~75 ms (54 -> ~73 t/s); a realistic one that beats MMQ@8 by
25% on the grid types buys 8-9 ms (+9%). Step 2 (production kernel) is cancelled.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,
`results/mmsb_sweep.txt`, `results/ref/*`; tables and ablations in `comm/files/seq50_*`.

## 35. T3: where the 8K DFlash d7 verify step actually goes

Measured 2026-10-20 on the exact 8K config from section 33 (`GGML_CUDA_PREFER_MMQ=1`, DFlash
`--spec-draft-n-max 7`, ctx 16384, all layers resident, exact-output env, fresh 8,829-token code
prefill). 512 greedy tokens: 50.6 t/s, 101.1 ms/step, 5.11 tok/step (a 400-token run of the same
config reaches 6.22 tok/step and 61.3 t/s - t/s is acceptance-bound, the step time is not).

Instruments: `GGML_MM_TIMING` (matmuls, n<=8), `GGML_FA_TIMING` (flash attention), plus two new
env-gated timers added for this measurement and default-off (verified: same tokens, 50.55 vs 50.67 t/s
on the unmodified path): `GGML_GDN_TIMING` (the GDN recurrent kernel, plain and fused-snapshot path)
and `GGML_GRAPH_TIMING` (one backend graph compute). Timer runs use `GGML_CUDA_DISABLE_GRAPHS=1`;
clean runs use graphs.

### 35.1 Step budget (per step, 101 ms)

| bucket | ms | share | how measured |
|---|---:|---:|---|
| FFN matmuls (gate/up/down, 64 layers) | 52.2 | 52% | MMTIME (target 49.3 + drafter 3.3) |
| attention matmuls (q/k/v/o/gate + GDN qkv) | 20.9 | 21% | MMTIME (target 18.8 + drafter 1.7) |
| GDN matmuls (ssm_alpha/beta/out) | 7.5 | 7% | MMTIME |
| LM head (target 2.44 + drafter 2.44) | 4.4 | 4% | MMTIME |
| flash attention (16 + 5 calls) | 4.5 | 4.5% | FATIME (target 3.5 + drafter 1.4) |
| GDN recurrent kernel (AR, 48 + 24 calls) | 2.6 | 2.6% | GDNTIME (new) |
| draft head + conv projections | 0.9 | 0.9% | MMTIME |
| norms, rope, adds, sampling | ~1 | 1% | remainder |
| host / launch / gaps | ~9 | 9% | wall (graphs on) minus GPU, split by LLAMA_STEP_TIMING (section 36) |

GPU kernel time is therefore ~89 ms and the matmul share 85%. The per-op timer sums carry ~4-5 ms of
launch latency after each sync (each op is event-timed and synced), so the bucket shares are the solid
part and the totals are upper bounds.

The FFN number reconciles exactly with the per-tensor microbench: 128 x ffn_up (247.5 us) + 64 x
ffn_down (313.1 us) = 51.6 ms. Nothing is hiding in the model - the step is the sum of the per-tensor
MMQ@8 numbers.

### 35.2 The two passes

| | target verify pass (n=8) | DFlash drafter pass (n=8) |
|---|---:|---:|
| matmuls | 78.1 | 8.3 |
| flash attention | 3.5 | 1.4 |
| GDN recurrent | 1.9 | 0.8 |
| sum | 83.5 | 10.5 |
| wall per step | ~87-88 (incl. ~5 gap) | ~16.4 (incl. ~6 gap) |

The drafter runs 0.86-1.0 times per step and costs ~16 ms of wall time (~16% of the step); its own
breakdown is ffn 3.3, attn 1.7, LM head 2.4, draft head (fc/selector_hidden) 0.4, conv projections
0.5, FA 1.4, GDN 0.8.

### 35.3 GDN at n=8: which kernel, and how it scales

`select_gdn_mma_path` (`gated-delta-net-mma.cu`) requires `n_tokens >= 64` (plus Ampere GA10x and
head-count checks), so **both the n=1 and the n=8 paths use the autoregressive kernel**
(`launch_gated_delta_net`), never the chunked/MMA one. Measured per op: 0.0256 ms at n=1, 0.0307 at
n=4, 0.0389 at n=8 - i.e. +52% for 8x the tokens, so it is fixed-cost dominated (launch + state IO),
not per-token. 48 layers = 1.9 ms per verify pass (2%). The GDN *matmuls* are column-invariant
(7.2 ms at n=1 vs 7.5 at n=8), like every other MMQ matmul.

### 35.4 Graphs, launches, and the host gap

- The scheduler reserves a 751-node decode graph (`sched_reserve: graph (pp bs=16, tg bs=1): nodes =
  751 / 531`); a request reports `graphs reused = 37, graph reallocs = 1` for 7 steps - one capture,
  then replays. Graphs-on vs graphs-off is 13 ms/step (11%): 100.9 vs 114.0 ms.
- Per step there are ~550 matmuls, 21 FA calls, 72 GDN calls and ~500 elementwise kernels: ~1100 GPU
  ops, issued as 2 graph replays (verify + draft).
- The host gap is zero for a plain n=1 token (no-spec wall 75.0 ms vs 75.4 ms of timed GPU), so all of
  the ~11 ms sits in the speculative loop: ~5 ms around the verify eval and ~6 ms inside the draft
  call (16.4 ms wall vs 10.5 ms GPU).
- `LLAMA_RS_REPLAY` is not part of it: `RS_REPLAY=0` is 1.6 ms/step *slower* and produces the same
  tokens.

### 35.5 Levers, and what is left

Ranked by measured size (see `comm/files/seq52_step_anatomy.txt` for the numbers):

1. **The matmul bulk, 73 ms (73%)** - no lever: section 34 shows MMQ@8 is at 1.08-2.07x of the floor
   and no reachable kernel is faster.
2. **Host gap, ~11 ms (11%)** - the only item above 5 ms. Needs one more measurement (host-side timers
   around the spec loop's phases: draft call, verify call, sampling, logits D2H) before a code change;
   realistic recovery 5-8 ms (+5-8%).
3. **Drafter LM head, 2.4 ms/step** - the drafter runs the full 248,320-row IQ4_XS head every step;
   whether DFlash2's lattice head needs it at all is a code question worth 2.4 ms (2.4%).
4. **GDN AR kernel, 2.6 ms** - a small-n chunked path could halve it (1.3 ms, 1.3%); not worth it on
   its own.
5. Flash attention 4.5 ms and the LM head 2.4 ms on the target are already at their best known
   settings (q4 decode attention, output head on GPU).

**Conclusion: there is no large lever left in the 8K verify step.** The step is 85% matmul at rates
that section 34 shows are at or near the achievable limit for these quant formats, 5% GDN+FA, and 11%
host-side speculative overhead. The only code lever worth trying is the host gap (~3-4 ms of it is
recoverable, +3-4%). Section 36 has the phase split (9 ms: submit 1.8, sampling+bookkeeping 3.3,
inter-step 3.5, drafter host 0.3) and section 37 the 200K idle-window measurements.
Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,

## 36. T3 follow-up: the 8K host gap, phase by phase

`LLAMA_STEP_TIMING=1` (new, default off) turns the server's dormant compile-time `DEBUG_TIMINGS`
scaffold into runtime phase timers and prints one line per eval step:

```
STEPTIME draft=.. pre=.. eval=.. sync=.. post=.. total=.. ops=<batch>
```

`draft` = `common_speculative_draft()` (the whole drafter call), `pre` = `pre_decode()` which contains
it, `eval` = the CPU submit of the target verify graph, `sync` = `llama_synchronize` after it (the
verify pass's GPU time), `post` = sampling + accept + bookkeeping, `total` = wall between step prints.

8K DFlash d7 (PREFER_MMQ, 200 tokens, 50 steps, graphs on, median ms/step): draft 9.85, pre-draft 0.08,
eval 1.78, sync 82.14, post 3.34, unattributed 3.5, total 100.7. The drafter call is GPU-bound (its GPU
is ~9.5 ms, host ~0.3). The host share is therefore ~9 ms, and the largest single item is the sampling
and bookkeeping in `post` (3.3 ms), then the graph submit (1.8 ms) and the rest of the step loop (3.5).
So the host is not the lever: at most ~3-4 ms is recoverable. This corrects the "~11 ms" estimate of
section 35, which came from wall minus per-op timer sums (those inflate the GPU side by ~4 ms).

## 37. T3 follow-up: the 200K PCIe idle window, and why MTP cannot use it

Measured on report.md 33's 200K row (DFlash d7, MMVQ, `LLAMA_KV_GPU_BUDGET_MIB=1000`, ctx 204800, base
snapshot restored + full prompt so only the ~3.4K suffix prefills), `GGML_KV_STAGE_PROF=1`,
`GGML_GRAPH_TIMING=1`, `LLAMA_STEP_TIMING=1`.

| phase | ms/step (DFlash d7, 512 tok, 84 steps) |
|---|---:|
| draft (drafter call) | 14.7 |
| eval (submit) | 136.7 |
| sync (verify GPU wait) | 131.1 |
| post | 4.2 |
| **step** | **289.4** (21.02 t/s, 6.08 tok/step) |

Staged-KV profile per eval: **span 234.2 ms**, staged attention 36.8 ms. In-graph GPU per step ~17 ms
(750-node graph 13.6 + 88-node 1.1 + 576-node 2.2). GPU busy ~= 17 + 36.8 + 14.7 = 68 ms of 289, so
**~220 ms/step (76%) is GPU idle under the PCIe KV stream**.

Width sweep at 200K with MTP d7/11/15/23/31 (verify width 8/12/16/24/32):

| n_max | width | t/s | ms/step | tok/step | draft ms | eval ms | VRAM peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 8 | 13.64 | 329.8 | 4.43 | 28.0 | 242.3 | 11015 |
| 11 | 12 | 13.31 | 338.1 | 4.50 | 40.8 | 247.1 | 11227 |
| 15 | 16 | 12.84 | 350.6 | 4.50 | 54.6 | 249.7 | 11241 |
| 23 | 24 | 11.37 | 395.9 | 4.50 | 84.6 | 269.1 | 11269 |
| 31 | 32 | 9.99 | 450.5 | 4.50 | 111.0 | 316.5 | 11299 |

The verify is nearly free to width 16 (+7.4 ms for 8 more columns, +2.5%), which confirms the idle-window
hypothesis. But MTP's sequential drafting costs +13.5 ms per four drafts and its acceptance saturates at
4.50 tok/step, so MTP t/s falls with width. The bottleneck is producing candidates, not verifying them.
DFlash is better on both counts at 200K (6.08 tok/step, 14.7 ms to draft a whole block).

Exactness at width > 8 (width > 8 leaves MMVQ for MMQ, so n=1 must be MMQ too): DFlash d7 at 200K,
512 tokens, identical tokens to index 158 then divergence:
MMVQ 21.02 t/s / 289.4 ms/step / 6.08 tok/step; MMQ 17.60 t/s / 279.2 ms/step / 4.91 tok/step.
Same speed per step (MMQ is 10 ms faster); **the whole difference is acceptance**, i.e. ULP differences in
the target hidden states feeding the drafter, not the matmul route's speed. At 8K the drafter is also
faster on MMQ (9.9 vs 14.7 ms) because its own pass runs at n=8. So the section 33 note that PREFER_MMQ
"loses at 200K because the n=1 draft steps are slower" is wrong: it loses on acceptance, and per-model
routing would not fix it.

Free VRAM at 200K with DFlash: ~835 MiB (peak 11453 of 12288; MTP peaks 11015). One GDN state copy is
293.62 MiB, so at most one extra branch state fits safely; the replay buffers (~60 MB for 8 tokens over
48 GDN layers) are the only scalable per-branch mechanism. The tree/parallel-path design that follows
from this is in `comm/files/seq54_tree_design.txt` (dq4 already takes per-query-row masks, so attention
is easy; the GDN recurrent state is the limiting resource).

## 38. Correction: MMQ at 200K is not slower - the earlier comparison was confounded

Section 33 concluded that all-MMQ is worse at 200K (17.58 vs 21.06 t/s) and recommended MMVQ there. That
comparison was between two runs that generate **different text**: MMQ and MMVQ produce identical tokens
up to index 158 and then diverge, so the acceptance figures (4.91 vs 6.08 tok/step over 512 tokens)
describe two different continuations, not the effect of the route. Measured on *identical* text instead
(runs short enough to stay inside the common prefix, so both modes emit exactly the same tokens):

| prompt / length | mode | t/s | steps | tok/step | ms/step | drafter call |
|---|---|---:|---:|---:|---:|---:|
| p200k code, 128 tok | MMVQ | 17.31 | 25 | 5.080 | 293.4 | 14.70 |
| p200k code, 128 tok | **MMQ** | **18.81** | **24** | **5.292** | **281.3** | 9.84 |
| p200k code, 150 tok | MMVQ | 17.16 | 30 | 4.967 | 289.5 | - |
| p200k code, 150 tok | **MMQ** | **18.35** | **29** | **5.138** | **280.0** | - |
| t200k_suf512, 150 tok | MMVQ | 15.97 | 32 | 4.656 | 291.5 | - |
| t200k_suf512, 150 tok | **MMQ** | **16.55** | **32** | **4.656** | **281.4** | - |

All three pairs emit identical tokens. On the same text MMQ accepts the same number or slightly more
(fewer steps for the same token count) and is 9-12 ms/step faster, because its drafter call is faster
(9.84 vs 14.70 ms: the DFlash drafter's own pass runs at n=8, where MMQ wins). The long-run 21.02 vs
17.60 is post-divergence text luck, one draw each.

Exactness at 200K with MMQ (new): no-spec MMQ (n=1) vs DFlash d7 MMQ (width 8), 200K, 512 greedy tokens:
both sha `05d4554fd13683fd`, token-identical. So **the 200K config is exact with `GGML_CUDA_PREFER_MMQ=1`**,
and it should replace the MMVQ recommendation for 200K. On these three identical-text samples that is
17.2-18.8 t/s instead of 16.0-17.3 t/s at the same text, and it makes 200K consistent with 8K/64K/128K.

Consequence for the wider-verification (tree) design of `comm/files/seq54_tree_design.txt`: there is **no
acceptance tax for verify widths above 8**, so the exact mode is free at width 16-24, and per-model routing
(target MMQ, drafter MMVQ) is unnecessary - the drafter is faster on MMQ anyway.

Caveat: the identical-text samples are 2 prompts x 1-2 lengths. Long runs still diverge and their
acceptance cannot be compared directly; the claim is that there is no measurable mode effect on the same
text, which is what the three pairs show.

## 39. Forest-of-paths (B=2) gate at 8K: the lattice does not contain the token often enough

Instrumented the DFlash2 lattice (opt-in `LLAMA_DFLASH_LATTICE=1`, token stream unchanged) to dump, per
step, the 16 candidates and 16x16 score matrix at every block position, plus the target's sampled token
next to each draft token. Joined per step (alignment verified: the greedy chain reconstructed from the
lattice equals the drafted tokens for all 96 steps).

8K DFlash d7, 64 steps (n=400), 38 full accepts, 26 failures:

| score rank of the target token at the mismatch position | share of failures |
|---|---:|
| 1 (the runner-up) | 54% |
| 3..8 | 27% |
| not in the top-16 | 19% |

- Oracle bound (branch at the *true* mismatch position, top-2): 54% of failures rescued. Top-3 at that
  position adds nothing, so extra candidates at the right position do not help.
- Implementable policy (branch at the least-confident position by margin, runner-up): 31% of failures
  (36% in an n=200 sample; the least-confident position is the mismatch position in 58% of failures).
  Expected token gain +0.35-0.45 tok/step = **+5.6-7.2% of accepted tokens**.
- Cost of verifying 16 instead of 8 columns, measured: **+25 ms/step at 8K** (verify 94 -> 119.5 ms,
  MTP d7/d15 with phase timers) but only **+7.4 ms at 200K** (hidden under the PCIe staging stream).

Net: **-15% t/s at 8K**, **+3% at 200K**. The forest is not worth building: 46% of the drafter's failures
cannot be repaired by one extra candidate at the right position and 19% are not in the lattice at all.
The 200K item that needs no code is the section 38 correction (switch to MMQ: exact, +4-7% at the same
text), which is worth more than the forest.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,

### 38.1 Follow-up: the flip to MMQ at 200K is rejected - both modes are exact, and MMVQ is 20% faster

Section 38 proposed switching 200K to `PREFER_MMQ=1` on the strength of identical-text step times. That
was wrong, and the re-validation the reviewer asked for shows why:

1. **MMVQ is exact at 200K too.** no-spec MMVQ (n=1) vs DFlash d7 MMVQ (width 8), 200K, 512 greedy
   tokens: both sha `2815707d781dd35c`, token-identical. (MMQ is exact as well, section 38.) So exactness
   does not force MMQ at 200K.
2. **The MMQ trajectory is reproducibly slower in t/s.** Two independent 512-token MMQ runs both gave
   sha `05d4554fd13683fd`: 17.60 and 17.49 t/s (4.91 tok/step, 279-281 ms/step). MMVQ: 21.02 t/s
   (6.08 tok/step, 289 ms/step). The step times are equal; the difference is that the MMQ run's text,
   after the divergence at token 158, is 19% less draftable. It is deterministic, so it is not noise - it
   is a property of the trajectory the mode takes on this prompt.
3. Therefore "equal acceptance on identical text" (section 38) does **not** mean "equal t/s on a long
   run": the two modes decode different text after the first divergence, and the draftability of that text
   is what sets tok/step. On the 200K code prompt, MMVQ wins by 20%.

Conclusion: keep MMVQ at 200K (section 33's row stands), and the general statement is: mode choice changes
the decoded text at 200K, so t/s must be compared on full runs of the actual prompt, not on the identical
prefix. The identical-prefix comparison is only valid for "is there a per-step or acceptance difference at
the same text" - and there is none.

## 40. Drafter quant is not the acceptance limiter: Q4_K_M == Q2_K_S on 8K output

The 8K DFlash drafter (`Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`, 0.56 GB) was compared against the
higher-precision `D:\Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1.14 GB, same block_size 8 / selector_top_k 16).
8K, exact config (`PREFER_MMQ=1`, DFlash d7, fresh prefill, 8,829-token code prompt):

| drafter | tokens | t/s | ms/step | tok/step | acc % | drafter call | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2_K_S | 400 | 61.45 | 101.5 | 6.234 | 75.1 | 9.95 | 10833 |
| Q4_K_M | 400 | 60.20 | 103.6 | 6.219 | 76.3 | 10.23 | 11383 |
| Q2_K_S | 512 | 50.35 | 101.5 | 5.110 | 59.5 | 10.01 | 10833 |
| Q4_K_M | 512 | 50.12 | 101.9 | 5.110 | 59.5 | 10.26 | 11387 |

The two drafters produce **identical token streams** at both lengths, the same accepted counts
(691 drafted / 411 accepted at 512 tokens each), and the Q4 drafter is marginally slower (+0.25 ms per
call, +550 MiB). So drafter quantization is not what caps acceptance: the 19% of failures where the
target's token is outside the drafter's top-16 (section 39) are inherent to the model's lattice, not to
its 2-bit weights. No reason to try the Q4 drafter at 64K/128K/200K.

## 41. Mode decisions need several prompts: 8K settles MMQ, 200K needs the multi-prompt method

Section 38/38.1 compared MMQ and MMVQ at 200K on one prompt and got contradictory-seeming results
(identical-text parity vs a 20% gap over 512 tokens). The resolution is that mode choice changes the
decoded text after the first divergence, and the draftability of that text then dominates t/s whenever the
step is not matmul-bound. The method that settles it: several task prompts per mode, same base, 512 greedy
tokens each, report per-prompt t/s and tok/step plus the mean.

### 41.1 8K, five task prompts (fresh prefill, 11-12K-token prompts, DFlash d7, 512 tokens)

| prompt | MMVQ t/s | MMQ t/s | delta | tok/step MMVQ / MMQ | ms/step MMVQ / MMQ |
|---|---:|---:|---:|---:|---:|
| q1 | 22.33 | 32.42 | +45.2% | 3.57 / 3.36 | 160.0 / 103.7 |
| q2 | 29.72 | 43.22 | +45.4% | 4.82 / 4.48 | 162.2 / 103.7 |
| q3 | 26.88 | 41.10 | +52.9% | 4.40 / 4.33 | 163.9 / 105.4 |
| q4 | 21.96 | 32.91 | +49.9% | 3.57 / 3.45 | 162.7 / 104.9 |
| q5 | 20.20 | 34.78 | +72.2% | 3.30 / 3.67 | 163.2 / 105.7 |
| **mean** | **24.22** | **36.89** | **+53.1%** | 3.93 / 3.86 | 162.4 / 104.7 |

MMQ wins 5/5 with essentially equal acceptance (mean tok/step 3.93 vs 3.86): the whole difference is the
step time, 104.7 vs 162.4 ms, because a width-8 verify on MMVQ is ~1.5x slower per matmul than MMQ's J=8
tile (section 29). At 8K the mode decision is therefore not a lottery at all: MMQ, by a wide margin.

### 41.2 200K is different, and needs the multi-prompt method

At 200K the step is PCIe-staging-bound (section 37), so MMQ and MMVQ step times nearly coincide
(279 vs 289 ms, +3.5% for MMVQ) and the text lottery dominates: on one prompt MMVQ measured 21.02 t/s and
MMQ 17.5-17.6 (reproducibly, identical shas). Five ~2.7K-token task suffixes on the 200K base are being
measured in both modes for the mean; the result goes in section 42 when the sweep finishes.
The expected outcome, from the step times and identical-text parity, is a small MMQ edge (+3.5% per step)
with a per-prompt spread of +-15% from the text.

### 41.3 Drafter quantization: verified not to be the limiter

Section 40 found Q4_K_M and Q2_K_S giving identical tokens at both 400 and 512 tokens. The obvious
suspicion - that the Q4 drafter was never really used - was checked by diffing the lattice dumps
(`LLAMA_DFLASH_LATTICE=1`, 3 steps x 7 positions each): **0 of 21 rows are identical** (different candidate
ids *and* different scores, e.g. position 1 top-1 is token 3729/score 19.2 for Q2 and token 313/score 15.5
for Q4). So the two drafters really do propose different things, and their *accepted* paths still coincide -
which is expected, because the accepted tokens are the target's own, not the drafter's. The conclusion
stands: drafter quantization is not the acceptance limiter.

Method lesson for the report: any config choice that changes the matmul route changes the generated text
after the first divergence; compare t/s over several prompts of the real workload, and use identical-prefix
runs only to test per-step or per-position effects.

## 42. 200K mode decision, multi-prompt: MMQ wins 4/5 prompts, +4.7% mean - MMQ everywhere

Same method as section 41: five ~2.7K-token task suffixes appended to the 200K base (byte-identical prefix,
CRLF preserved, ctx 212992 so the restored base is reused and only the suffix prefills), 512 greedy tokens
each, both matmul routes. DFlash d7, budget 1000.

| prompt | MMVQ t/s | MMQ t/s | delta | tok/step MMVQ / MMQ | ms/step MMVQ / MMQ |
|---|---:|---:|---:|---:|---:|
| s1 | 14.43 | 15.49 | +7.4% | 4.22 / 4.37 | 292.7 / 282.0 |
| s2 | 15.76 | 16.47 | +4.5% | 4.60 / 4.65 | 292.1 / 282.0 |
| s3 | 14.01 | 13.84 | -1.2% | 4.09 / 3.90 | 291.8 / 281.9 |
| s4 | 11.46 | 12.52 | +9.3% | 3.36 / 3.52 | 293.4 / 281.5 |
| s5 | 12.32 | 12.76 | +3.6% | 3.60 / 3.60 | 292.0 / 282.0 |
| **mean** | **13.59** | **14.21** | **+4.7%** | 3.97 / 4.01 | 292.4 / 281.9 |

Acceptance is equal in expectation (3.97 vs 4.01 tok/step); the gain is the step time, 281.9 vs 292.4 ms
(MMQ 3.6% faster), exactly what the single-step measurement in section 37 predicted. The earlier
single-prompt result (MMVQ 21.02 vs MMQ 17.5 on the original code prompt) was the text lottery, not the
mode: on that prompt the MMVQ trajectory happened to be much more draftable (6.08 vs 4.91 tok/step).

**Conclusion: `GGML_CUDA_PREFER_MMQ=1` is the right mode at every context** - 8K +53% (section 41.1),
64K ~flat (+2.3%), 128K +21% (section 33), 200K +4.7% (above) - and it is the exact mode everywhere
(spec == no-spec on the same route at 8K, 128K and 200K, sections 33 and 38). The per-context config table
in section 33 is now one mode for all four contexts; only the draft length and the KV budget differ.

Note on absolute numbers: these task prompts are harder to draft than the original code question
(3.97-4.01 tok/step vs 6.08 there), so the t/s here (13.6-14.2) is lower than the 21.0 t/s of section 33's
200K row. The paired comparison is what matters.

160K (seq 61's item iii) is deferred: it needs a fresh 160K base and is no longer decision-critical
because the mode question is settled by the 8K and 200K multi-prompt results. Cost of that base depends on
the launch: at a 160K context (KV resident) a 160K prefill runs at ~250 t/s (~12 min); a *200K* prefill at
`-c 204800` (KV staged) is ~22 t/s (~2.5 h), because the staged-KV DMA per ubatch grows with the prefix
(section 46.1). Reported as deferred rather than silently dropped.

## 43. Q1: the CPU drafter at 200K is a clear loss (2x slower step)

The idea: run the DFlash drafter on the CPU (`--spec-draft-ngl 0`) to free its VRAM, then raise the KV
budget so fewer KV layers are staged. Measured at 200K on task prompt s1 (128 greedy tokens, MMQ, DFlash d7,
`LLAMA_STEP_TIMING` for the phases):

| config | t/s | ms/step | tok/step | drafter ms/call | verify sync ms | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|
| GPU drafter, budget 1000 | **21.33** | 283.5 | 6.05 | **10.0** | 131.3 | 11493 |
| CPU drafter, budget 1000 | 9.83 | 615.3 | 6.05 | 312.3 | 130.9 | 10933 |
| CPU drafter, budget 1500 | 10.37 | 583.1 | 6.05 | 313.2 | 115.0 | 11401 |
| GPU drafter, budget 1500 | OOM at load | - | - | - | - | - |

- The CPU drafter is numerically equivalent (same tokens, sha `a1d82487ec4c621d`, same 6.05 tok/step) but
  **31x slower per call**: 312 ms vs 10.0 ms. The drafter is a 5-layer model over a 2048-token window; on
  6 CPU threads that is 300 ms, and no KV-staging saving can pay for it.
- Freeing the drafter does buy KV: peak VRAM drops 11493 -> 10933 (560 MiB), and raising the budget
  1000 -> 1500 cuts the verify's staged-KV sync from 130.9 to 115.0 ms (-15.9 ms/step). That prices a staged
  KV layer at **~15-17 ms/step at 200K**, matching the earlier estimate.
- Net: 283.5 -> 583.1 ms/step, **21.33 -> 10.37 t/s**. Rejected for the 200K config.
- The GPU drafter cannot use the larger budget (OOM), so "GPU drafter + budget 1500" is not an option; the
  trade is exactly the one measured above and it goes the wrong way by 2x.

Where the CPU drafter could still matter: as the *enabler* for something else that needs ~0.5 GiB (for
example DFlash on IQ3_S, where the binding constraint is the bigger weight set - Q4). On its own it is
rejected.

## 44. Q2 Phase 0: the cost of one -c 204800 launch for every context

The dynamic-config idea needs to know what a single large launch costs at short contexts. Measured with the
final config (MMQ, DFlash d7, 128 greedy tokens) at three content lengths, against an all-resident launch:

| content | all-resident launch | t/s | ms/step | tok/step | one launch `-c 204800`, budget 1000 | t/s | ms/step | tok/step | penalty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | `-c 32768` | 44.89 | 108.8 | 4.88 | same content | 41.71 | 117.1 | 4.88 | **-7%** |
| 8K | `-c 32768` | 52.77 | 109.4 | 5.77 | same content | 37.12 | 122.2 | 4.54 | **-30%** |
| 32K | `-c 40960` | 35.25 | 112.6 | 3.97 | same content | 19.74 | 156.9 | 3.10 | **-44%** |

4K and 8K pairs produce identical tokens (sha `e422dffc8fe3b133`, `d0fb7092955c105a`); the 32K pair diverges
(`00598a22...` vs `229a0f1f...`) because the staged attention path has slightly different numerics, which is
also why the 8K pair shows a lower tok/step for the same token stream: the drafter sees ULP-different hidden
states and offers different drafts, so the same 128 tokens need 28 steps instead of 22.

Why the penalty grows with content: the staged layers' K/V copy volume per eval scales with the content
length, and so does the staged attention, so 4K pays almost nothing while 32K pays 39% more per step plus the
acceptance effect. Above ~64K the KV cannot be resident at all, so the single launch costs nothing extra
relative to the per-context best (that is the regime section 33's per-context table was measured in).

Pending rows: 64K and 128K under the one-launch setting need base snapshots to be affordable (a staged
prefill of that size runs at 200-340 t/s, so a fresh 64K/128K prefill is 3-10 minutes each and a base pair
would take ~1 h to build); the per-context bests from section 33 are 38.51 (64K) and 35.56 (128K). The
dynamic config is therefore worth roughly 30-45% at 4K-32K, ~10-20% at 64K and 0 at 200K.

## 45. Q4 Phase 0: IQ3_S VRAM accounting, and what one CPU GDN layer costs

Model: `D:\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (11.28 GiB of weights). Measured at 8K (ctx 16384, `-ub 32`,
MTP d4, `GGML_CUDA_PREFER_MMQ=1`, 16/16 KV layers staged, `blk.0`/`blk.1` on the CPU) with the loader's own
buffer accounting (verbosity 4):

| component | size |
|---|---:|
| CUDA0 model buffer (62 layers) | 10892.42 MiB |
| CPU_Mapped model buffer (2 layers, mmap) | 655.65 MiB |
| KV cache, pinned host (16 layers, 16384 cells, q4_0) | 288.00 MiB (K 144 + V 144) |
| KV device slots (2 slots x 18 MiB) | 36.00 MiB |
| recurrent state RS (R 5.62 + S 288.00) | 293.62 MiB |
| compute buffers CUDA0 + CUDA_Host | 42.15 + 3.77 MiB |
| MTP layer KV (1 layer, 8448 cells) + its compute | 9.28 + 8.41 MiB |
| output buffer | 0.95 MiB |

So one GDN layer is **~328 MiB of weights** (655.65 / 2), the whole 11.28 GiB is 10.9 GiB on the GPU plus
0.64 GiB on the CPU, and the non-weight GPU residents are KV slots 36 + RS 294 + compute ~60 MiB.

Cost of one CPU GDN layer (8K, MTP d4, 128 greedy tokens, everything else identical):

| CPU GDN layers | t/s | ms/token | VRAM peak | notes |
|---:|---:|---:|---:|---|
| 2 | 19.61 | 50.6 | 11671 | prefill 172 t/s |
| 3 | 14.77 | 67.2 | 11549 | prefill 151 t/s |
| 4 | 11.25 | 88.2 | 11417 | prefill 136 t/s |

Marginal cost: **+16.6 ms/token (2->3) and +21.0 ms/token (3->4), i.e. ~18.8 ms/token or ~89 ms per
speculative step** (4.75 tok/step) per CPU GDN layer. VRAM freed per layer is only ~122-130 MiB of peak, far
less than its 328 MiB of weights, because the freed bytes are partly absorbed by allocator slack and the
compute/staging buffers.

Weight staging (the reviewer's idea) priced against that: 328 MiB per layer per pass at the measured
~13 GB/s PCIe is **~25 ms**, against ~89 ms/step of CPU compute, so ~3.5x cheaper - but the DMA has to hide.
At 8K the step is ~215 ms with two CPU layers, of which the GPU is busy ~100 ms (verify 82 + draft 20) and
PCIe is nearly idle; two staged layers would add ~50 ms of DMA per step that can overlap that compute,
replacing ~180 ms of CPU time. Expected step ~120-140 ms instead of 215, i.e. **~19.6 -> ~30-35 t/s at 8K**
if the overlap works. At 64K/128K the KV staging already uses the PCIe stream (section 37: the step is
DMA-bound there), so weight staging must take lower priority than KV - which is exactly the dynamic-config
interaction in the queue.

(Prefill also slows with CPU layers: 172 -> 151 -> 136 t/s, i.e. the same layers cost prefill throughput; a
streamed GPU layer would restore it.)

## 46. Review R1-R3: the -c dependence is real, it is the dq4 split partition, and it is now fixed

### 46.1 R1: which prefill rate is right - 22 t/s or ~250 t/s?

Both, for different launches. A prefill at a context where the KV is resident runs at 250-460 t/s
(measured: 4K 374, 8K 467, 32K 406-460, and section 44's staged-but-short runs 339-374). A **200K** prefill
at `-c 204800` with the KV staged runs at **22 t/s**: the staged layer copies use "prefix-only DMA", so each
ubatch copies the whole prefix of every staged layer, i.e. the DMA per ubatch grows linearly with the prefix
and the total is quadratic. That is why 4K/8K/32K staged prefills look fine (339-374 t/s) while a 200K one
does not (22 t/s -> ~2.5 h). Section 42's sentence conflated the two; corrected there.

### 46.2 R2: the 8K acceptance loss was NOT staging noise - it was the split partition

At 8K, `-c 32768` vs `-c 204800` gave *identical tokens* but 5.77 vs 4.54 tok/step (22 vs 28 steps). The
drafter's lattice dumps show the difference from **step 0, position 1**, and it is large: the 256 score
values differ by up to 14.27 (mean 3.66) on a ~19 scale, and the candidate sets differ. That is not
accumulated rounding.

Isolating the two variables:
- Same context (`-c 16384`), resident vs forced-staged (`budget 100`): **identical tokens**
  (`5196aac2fac9597a` both), 54.5 vs 50.9 t/s - staging alone costs 6.6% and changes nothing.
- Same context (`-c 16384`) staging forced, and same 32K content at `-c 40960` vs `-c 65536` with
  `LLAMA_FA_DECODE_Q4_SPLITS=1` (same partition): **identical tokens**.
- And with the new fixed-in-tokens partition (46.4): `-c 32768` vs `-c 204800` at 8K gives identical tokens
  **and identical draft/accept (150/105)** instead of 150/105 vs 194/99, and the t/s penalty falls from
  -30% to **-8%** (49.86 vs 45.93).
So the mechanism is: the dq4 split partition is derived from the whole KV tensor (i.e. from `-c`), a
different partition changes the fp32 combine order of the attention partial sums, that changes the hidden
states materially enough to reorder the drafter's lattice, and the 8K penalty was that effect, not staging.

### 46.3 R3: `-c` alone changes the output (reproducibility bug), and the fix

32K content, all resident, no staging anywhere:
- `-c 40960` 27.57 t/s, 134 drafted / 43 accepted, sha `c8c29024f8407461`
- `-c 65536` 23.02 t/s, 145 drafted / 41 accepted, sha `423b1a16c9fd70af`  **<-- different output**
With `LLAMA_FA_DECODE_Q4_SPLITS=1` in both: identical (`f61f3e05a3530666`). So launching the same model and
prompt with a different `-c` changes the generated tokens, which breaks the expectation that `-c` only
reserves memory.

### 46.4 The fix: `LLAMA_FA_Q4_SPLIT_TOKENS`

New opt-in env in `fattn-decode-q4.cu`: when set, the split partition is fixed in **tokens**
(`chunks_per_split = ceil(N / DQ4_CHUNK)`), independent of the cache size, so the fp32 combine order - and
therefore the output - no longer depends on `-c`. Default 0 keeps the old behaviour byte-identically.

Verified:
| check | result |
|---|---|
| 8K, `-c 32768` vs `-c 204800` (staged), SPLIT_TOKENS=16384 | identical tokens `d0fb7092955c105a`, identical 150/105 draft/accept, 49.86 vs 45.93 t/s (-8%) |
| 32K, `-c 40960` vs `-c 65536`, SPLIT_TOKENS=16384 | identical tokens `c8c29024f8407461`, identical 134/43 |
| 8K, spec vs no-spec, 512 tokens, SPLIT_TOKENS=16384 | identical `f91fe75fd51c0f09` (exact) |
| cost where the default partition was already tuned | 8K -c 32768: 52.77 -> 49.86 t/s (-5.5%); 32K -c 40960: 27.57 -> 24.90 (-9.7%) |
| gain where it was not | 8K -c 204800: 37.12 -> 45.93 t/s (+24%) |

So the env buys -c-independent output for 5-10% where the launch context already matches the workload, and
+24% (and identical acceptance) when one large launch serves short contexts. That is the "one-launch penalty
at 8K" going from -30% to -8% without any residency switch, as the review predicted. 200K with the fixed
partition (13 splits instead of 2) is not measured yet.

## 47. Q4 weight staging, set selection (numbers before any code)

### 47.1 Tensor sizes (exact, from the GGUF tensor offsets)

`D:\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`: `output.weight` 682.0 MiB, `token_embd.weight` 388.4 MiB,
**largest per-layer tensor 69.7 MiB** (`blk.64.*` ffn; in the main stack `blk.9.ssm_out` 69.7, ffn tensors
mean 37.3 MiB, min 18.6), and **blk.0 + blk.1 together are 267.3 MiB over 28 tensors**.

### 47.2 Why the per-layer slot plan was wrong (review point a)

Two slots sized to a whole GDN layer would cost 2 x ~133 MiB = 267 MiB, exactly the VRAM that blk.0/blk.1
already free by sitting on the CPU - net zero. The fix is per-tensor slots:

| plan | slot size | streamed bytes | net VRAM freed | DMA per pass |
|---|---:|---:|---:|---:|
| per-layer (rejected) | 2 x 133 = 267 MiB | 267 MiB | ~0 | 21 ms |
| **per-tensor** | **2 x 69.7 = 139 MiB** | **417 MiB** (blk.0 124 + blk.1 144 + ~4 resident ffn tensors ~150) | **~278 MiB** | **32 ms** |

So the streamed set has to be *larger* than the two CPU layers: it must also take ~150 MiB of currently
resident ffn tensors, both to pay for the 139 MiB of slots and to end up with more free VRAM than the CPU
placement. Every byte of that set is then streamed at high speed instead of being slow-on-CPU or resident.

### 47.3 Overlap available (review point b), measured

From `STEPTIME` on the same IQ3_S config (8K, MTP d4, KV 16/16 staged), median per step:

| CPU GDN layers | step total | eval (CPU-bound) | sync | MTP draft | post |
|---:|---:|---:|---:|---:|---:|
| 2 | 213.7 ms | 161.7 | 24.7 | 19.9 | 3.2 |
| 3 | 283.0 ms | 229.5 | 26.8 | 19.7 | 3.9 |
| 4 | 349.2 ms | 295.4 | 26.6 | 19.7 | 3.5 |

So the GPU-side window per step is the verify pass (~95-100 ms for IQ3_S) plus the MTP draft phase
(19.7-19.9 ms), i.e. ~115-120 ms, against **32 ms of DMA** for the 417 MiB set: ~28% of the window, so the
copies can hide. During the draft passes the *main-layer* slots are idle, so the first ~20 ms of the next
verify pass's copies can start there (review point c).

### 47.4 What a CPU GDN layer really costs (review point d)

Direct per-step figures from the same table: **+69.3 ms/step (2->3 layers) and +66.2 ms/step (3->4), i.e.
~68 ms/step per CPU layer** - not the 89 ms I derived from ms/token x tok/step. It all lands in `eval`
(the call blocks while the CPU computes), while `sync` stays 25-27 ms.

Is it CPU compute or the host<->device activation round trip? Two CPU layers on adjacent blocks
(blk.0+blk.1, one round trip) vs two non-adjacent blocks (blk.0+blk.32, two round trips), same everything
else: **18.31 vs 18.09 t/s, 53.8 vs 54.4 ms/token** - a 1% difference. The cost is CPU compute, not the
round trip, so adjacency does not matter and a streamed set can be chosen on size alone.

### 47.5 Expected outcome

Per streamed GDN layer: 68 ms/step of CPU compute replaced by ~10-13 ms of overlapped DMA plus ~3-4 ms of
GPU compute. Replacing the two CPU layers at 8K: step 213.7 -> ~95-105 ms, i.e. **19.6 -> ~45-50 t/s**
(+130%), with ~278 MiB more free VRAM than the current CPU placement. At 64K/128K the KV staging already
uses the PCIe stream (section 37), so the weight-staged set must shrink or move behind KV there.

## 48. Q4 weight staging M1: the slot mechanism, its real DMA rate, and its overlap

Mechanism (opt-in, `LLAMA_W_STAGE_MIB=<MiB per eval>`, `LLAMA_W_STAGE_PINNED=1`,
`LLAMA_W_STAGE_PERIOD_MS=<ms>`): two device slots sized to the largest streamed tensor (69.7 MiB -> 2 x
70 MiB), a non-blocking copy stream, per-slot events, and a bounded queue (the next firing waits for the
previous copies, like the KV staging object). **M1 is a dry run: nothing is re-pointed, no tensor moves, so
the output must be - and is - byte-identical.**

Measured rate through the mechanism (417 MiB, 6 chunks alternating slots):

| source memory | copy time | rate |
|---|---:|---:|
| pageable (malloc, like the mmap'd weights) | 64.4-74.8 ms | **5.9-6.8 GB/s** |
| pinned (`cudaHostAlloc`) | 32.6-32.8 ms | **13.2-13.4 GB/s** |

So the 13 GB/s assumption holds only for a *pinned* source. The real streamed set would come from the
mmap'd GGUF (pageable) and reach only ~6.6 GB/s -> 64 ms per pass for 417 MiB; pinning the streamed set at
load (`--load-mode none` / mlock) or into a pinned mirror is required to get the 13 GB/s figure.

Overlap, on a GPU-bound config (IQ3_XXS, DFlash d7, 8K, 128 tokens, 106.8 ms/step):

| config | t/s | ms/step | tok/step | tokens |
|---|---:|---:|---:|---|
| no copies | 54.07 | 106.8 | 5.77 | `d0fb7092955c105a` |
| 417 MiB copied per step (32.7 ms of DMA) | 49.49 | **116.6** | 5.77 | `d0fb7092955c105a` (identical) |

32.7 ms of DMA costs **+9.8 ms of step time, i.e. ~70% of the copy hides** under an 8K verify step. Good
enough for the plan: with the two CPU layers replaced, the projected 8K step is
213.7 - 136 (CPU compute removed) + 7 (their GPU compute) + ~10 (unhidden DMA) ~= 95 ms, i.e. ~50 t/s.

Two things M1 taught that the design must respect:
1. **The slots do not fit on top.** At 8K with 2-3 CPU layers there are only ~600-740 MiB of headroom, and
   adding 2 x 70 MiB of slots OOMs during graph allocation. The residency change (M2) must therefore remove
   the streamed tensors' bytes in the same step that allocates the slots - which is what re-pointing does,
   and why the per-layer slot plan of section 47.2 was wrong.
2. **`graph_compute` is the wrong hook.** It is called once per scheduler *split*; with whole layers on the
   CPU one prefill ubatch produced hundreds of calls (15,745 firings in one 8K run, which collapsed the
   prefill to 15 t/s). The dry run had to be rate-limited for measurement; the real design must tie each
   copy to the layer that uses it, which is the natural boundary anyway.

## 49. S1: tuning LLAMA_FA_Q4_SPLIT_TOKENS (the split constant)

Decode t/s, DFlash d7, MMQ, 128 tokens (8K/32K) or the 200K s1 task prompt:

| N (split tokens) | splits at 8K | 8K, `-c 16384` | 8K, `-c 204800` | 32K, `-c 40960` | 200K, `-c 212992` |
|---:|---:|---:|---:|---:|---:|
| default (from cache size) | 2 | 52.77 | 37.12 | **35.25** | 15.49 |
| 1024 | 8 | **54.52** | 48.43 | - | - |
| 2048 | 4 | 54.38 | **48.95** | 22.29 | **21.06** |
| 4096 | 2 | 53.75 | 48.67 | - | - |
| 8192 | 1 | 50.51 | 45.97 | 32.05 | - |
| 16384 | 1 | 49.86 | 45.93 | 28.05 | - |

Readings:
- At 8K, N=1024/2048 are **equal or better than the default** (54.4-54.5 vs 52.8, +3%) and give
  **+31% at the one-launch setting** (48.4-49.0 vs 37.1). The default degenerates at `-c 204800` because the
  partition comes from the cache size: 1600 chunks per split means ONE split, i.e. 24 blocks for 28 SMs.
- At 200K, N=2048 is **+36%** over the default (21.06 vs 15.49): the default again gives only 2 splits for a
  200K prefix.
- At 32K the default wins and every constant is worse (N=8192 -9%, N=16384 -20%, N=2048 -37%):
  2 splits ≈ 16K tokens each is best there.
So there is **no single constant within 2% of the default everywhere**; N=2048 is the best single-launch
setting at 8K and at 200K, and 32K is the counter-example. The clean fix is a partition that follows the
*position* (which is -c independent and adapts), not the cache size and not a constant - that needs the
unpadded position plumbed into the kernel (today it uses `K->ne[1]`, which is padded, hence the whole-cache
workaround). Kept opt-in.

## 50. S2: prefill under one launch (staged KV) is 17-19% slower up to 32K, much worse at long content

Fresh-prefill throughput, DFlash d7, MMQ:

| content | matched launch | one launch `-c 204800` | penalty |
|---|---:|---:|---:|
| 4K | 460 t/s (`-c 32768`) | 374 t/s | -19% |
| 8K | 450 t/s (`-c 16384`) | 374 t/s | -17% |
| 32K | 406 t/s (`-c 40960`) | 339 t/s | -17% |
| 64K | 374 t/s (extrapolated; the matched run is invalid because the 67,966-token prompt exceeds `-c 65536`) | 294 t/s | ~-21% |
| 200K content | n/a (cannot be resident) | 22 t/s | ~-95% vs the ~450 t/s resident rate |

So the "quadratic prefix copy" is mild (17-21%) up to 64K and only collapses for a full 200K-token prefill,
where each ubatch copies the entire prefix of every staged layer.

Candidate fixes, measured where possible:
- **Raise the ubatch for staged prefill** (fewer ubatches -> fewer prefix copies): `-ub 512` at `-c 204800`
  **OOMs** during context creation, so this needs compute-buffer headroom that does not exist at 200K-class
  contexts.
- **Copy only the new tokens per ubatch** (resident window for the staged layers): impossible without
  bounding the attention window, since attention needs the whole prefix - unless the staged layers' attention
  is computed host-side from the pinned prefix (which is where the data already lives) instead of copied.
- **Stream the staged layers' weights instead of the KV** (weight staging, section 47): during prefill each
  layer's weights are read once per ubatch, which is the same order as the copy it replaces, and it does not
  grow with the prefix. This makes weight staging and the prefill fix the same mechanism.
Size of the fix: the first two are small changes with real costs (memory, or an attention-window change);
the third is the M2 weight-staging prototype. Recommendation: keep prefill penalties in mind but do not
start a separate prefill fix; fold it into M2.

## 51. S1 corrected: the split-constant t/s differences were mostly the text lottery

Review point (seq 69) was right: changing N changes the fp32 combine order, which changes the generated text,
and t/s then follows draftability. Redoing the sweep with kernel-isolating columns - ms/step, tok/step, and
the FA decode kernel time - from the same runs:

| cell (DFlash d7, MMQ) | t/s (earlier) | ms/step | tok/step |
|---|---:|---:|---:|
| 8K, `-c 16384`, default partition | 52.77 | 106.8 | 5.77 |
| 8K, `-c 16384`, N=1024 | 54.52 | **105.9** | 5.77 |
| 8K, `-c 16384`, N=2048 | 54.38 | 106.2 | 5.77 |
| 8K, `-c 16384`, N=8192 | 50.51 | 119.7 | 6.00 |
| 8K, `-c 204800`, default | 37.12 | 122.2 | 4.54 |
| 8K, `-c 204800`, N=2048 | 48.95 | 117.9 | 5.77 |
| 32K, `-c 40960`, default | 35.25 | **112.6** | 3.97 |
| 32K, `-c 40960`, N=2048 | 22.29 | 116.3 | 2.57 |
| 32K, `-c 40960`, N=8192 | 32.05 | 120.1 | 3.82 |
| 200K, `-c 212992`, default | 15.49 | **279.2** | 4.91 |
| 200K, `-c 212992`, N=2048 | 21.06 | 287.2 | 6.05 |

- At the matched 8K context N=1024/2048 are ~1% faster per step than the default (105.9-106.2 vs 106.8) - a
  wash. At the one-launch 8K setting N=2048 is 3.5% faster per step (117.9 vs 122.2) but its t/s advantage
  (48.9 vs 37.1) is mostly tok/step (5.77 vs 4.54), i.e. the lottery.
- At 32K the default is the fastest per step (112.6 vs 116.3) and N=2048's -37% t/s is almost entirely
  acceptance (2.57 vs 3.97 tok/step).
- At 200K the default is slightly faster per step (279.2 vs 287.2); N=2048's +36% t/s is acceptance
  (6.05 vs 4.91) - and 21.06 t/s is exactly the old "MMVQ wins" lottery number from section 42.

Controlled cross-check (graphs off so `GGML_FA_TIMING` is legal, 48 tokens, identical tok/step 5.88 in all
four cells): 8K matched default 119.9 ms/step, N=2048 118.7; 8K one-launch default 136.2, N=2048 126.8
(-7%). FA decode kernel time per step: 7.06 (matched default), 6.68 (matched N=2048), 4.37 / 3.07 at the
one-launch setting - but the staged layers' attention bypasses the FATIME hook (section 27), so those two
cells undercount FA and the step gap there includes the untimed staged attention plus copies.

**Conclusion: by kernel time there is no N worth adopting as a speed setting (deltas -3% to +1% except a 7%
per-step win at the one-launch 8K setting). `LLAMA_FA_Q4_SPLIT_TOKENS` stays opt-in as a *reproducibility*
control - it is what makes the output independent of `-c` - not as a tuning knob.** Section 49's t/s table
should be read with this correction; its recommendation ("no single constant") stands, for the kernel-time
reason rather than the t/s one.

## 52. M2 precursor: the GGUF mapping can be pinned in place on Windows (no mirror needed)

Review point (seq 68, item 1): try `cudaHostRegister` on the mmap'd weight ranges before paying for a pinned
mirror. Standalone test (`tests/wstage_register_test.cu`, 70 MiB and 417 MiB regions of the real IQ3_S GGUF,
6 chunks of `cudaMemcpyAsync` into a device slot, median of 9 reps):

| source of the copies | 70 MiB | 417 MiB |
|---|---:|---:|
| pageable heap buffer | 6.41 GB/s | 6.60 GB/s |
| read-only file mapping, unregistered | 6.57 GB/s | 6.68 GB/s |
| **read-only file mapping + `cudaHostRegister(..., cudaHostRegisterReadOnly)`** | **13.27 GB/s** | **13.38 GB/s** |
| explicit pinned mirror (`cudaHostAlloc` + memcpy at load) | 13.27 GB/s | 13.37 GB/s |

So on Windows `cudaHostRegister` with `cudaHostRegisterReadOnly` on the mapped GGUF works and reaches the
full pinned rate (13.3 GB/s), i.e. **no pinned mirror and no load-time copy are needed**: the streamed set
can be registered in place, saving the 417 MiB of host RAM the mirror would cost. Without registration the
mmap'd pages copy at 6.6 GB/s (64 ms for 417 MiB instead of 32 ms), which confirms that registration is
mandatory for the M2 design, not an optimisation.

## 53. M2 status: the staging object is in, and the integration point is now known exactly

The mechanism (opt-in, double-gated `LLAMA_W_STAGE_ENABLE=1 LLAMA_W_STAGE_BLOCKS="0,1"`): find the streamed
blocks' leaf tensors in the graph, allocate one CUDA slot per tensor, re-point the tensor into the slot with
`ggml_backend_tensor_alloc`, register the mapped GGUF source `cudaHostRegisterReadOnly`, refill every slot at
the start of each pass on a private copy stream, and make the consuming node wait on the fill event
(`cudaStreamWaitEvent` before `ggml_cuda_compute_forward`). Default is untouched (verified: 8K IQ3_XXS DFlash
d7, 52.78 t/s, sha `d0fb7092955c105a`; the M2 code only runs with both env vars set).

**The integration finding: a graph-time hook cannot work.** With `-ot "^blk\.0\.=CPU"` - the only way the two
layers fit today - the offloaded layer never appears in a *CUDA* graph at all: the scheduler assigns those
matmuls to the CPU backend, so the CUDA graph's node list contains no non-CUDA leaf tensors of blk.0/blk.1
(measured: "no streamed tensors found in the graph", and the identical baseline run with the M2 code loaded:
18.49 t/s, the same as before). So the streamed tensors must be re-pointed **at load time, before the
scheduler assigns ops**, i.e. the design needs a third placement, next to "CUDA buffer" and "CPU buffer":

| placement | VRAM | compute | today |
|---|---:|---|---|
| CUDA model buffer | full layer bytes | GPU, fast | resident |
| CPU buffer (`-ot ...=CPU`) | 0 | CPU, 68 ms/step per layer | the current IQ3_S config |
| **streamed (new)** | **2 x 69.7 MiB of slots** | GPU, fast, ~25-32 ms DMA per pass | this section |

With two slots serving all 28 tensors of the two layers (the reviewer's fixed-A/B assignment, tensors
aliasing a slot in consumption order, which is safe because no two sharers are live at once), the VRAM cost
is 139 MiB against the 267 MiB the two layers occupy - so it fits where residency does not, and it is the
only placement that is both fast and affordable.

What remains for M2, in order:
1. A `streamed` ggml buffer type whose tensors live in the slot pair (allocation = 2 slots, per-tensor
   address = slot A or B by consumption order), so the loader places those tensors there and the scheduler
   assigns their ops to CUDA0.
2. Loader wiring: when `-ot` maps a block to `STREAM`, place its tensors in that buffer instead of the CPU
   buffer (the mmap stays the source).
3. The staging object from this section handles the per-pass refill and the event ordering; with the tensors
   as CUDA buffers the graph is captured once and replays, so `LLAMA_CUDA_DISABLE_GRAPHS` can come off.
4. Exactness gate: streamed vs resident, 512-token sha, plus spec == no-spec, then 8K/64K/128K against the
   33-36 t/s ceiling.

Everything that can be measured without step 1 has been measured (sections 47, 48, 52): registration works
in place at 13.3 GB/s, ~70% of a copy hides, the CPU placement it replaces costs 68 ms/step per layer, and
the slot size is fixed by the largest tensor at 69.7 MiB.

## 54. M2 integration: mechanism verified byte-exact, end-to-end output still wrong (parked)

Progress on the last piece, and an honest stopping point.

Done and verified:
- `streamed` placement implemented: the loader (llama-model.cpp, at the end of `load_tensors`) walks every
  model tensor and calls a `ggml_backend_cuda_w_stage_adopt` proc exposed through the CUDA registry's
  `get_proc_address`; the adopt re-points a matching tensor into one of two device slots
  (`LLAMA_W_STAGE_SLOT_MIB`, default 70), registers the mapped GGUF source with
  `cudaHostRegisterReadOnly`, and records a per-tensor event.
- The tensors then present as CUDA buffers, so the scheduler keeps their ops on the GPU: the prefill went
  from 150-172 t/s (CPU layers) to 181-189 t/s and the step no longer blocks in `eval` for CPU compute.
- **The copy mechanism is byte-exact**: the self-check (`LLAMA_W_STAGE_CHECK=1`) reads 4 KiB back from the
  slot and compares it with the mapped source - "src=... slot=... bytes=20480 first 4096 bytes differ in 0".
- Slot VRAM: 2 x 70 MiB (140 MiB) against 267 MiB for blk.0+blk.1, and the three-block variant
  (blk.0/1/2 offloaded, three adopted) fits at 11.6 GiB peak.

Not working: the end-to-end output. Every streamed run produces 0 % draft acceptance (242 drafted, 0
accepted) with a prefill-throughput that looks normal, so the streamed layers are being *computed* with
wrong data even though one tensor was verified byte-identical. Two candidate causes, both in the hook
coverage rather than the copies:
1. the per-node hook may not cover every consumer: it sits at the `ggml_cuda_compute_forward` call in
   `ggml_cuda_graph_evaluate_and_capture`, and the fork's fused paths (and the multi-stream branch) can
   bypass that site;
2. adoption order != consumption order, so slot A/B sharing needs the copy tied to the *consumer* (my
   unconditional copy-on-consume version still failed, which points at (1) - if the hook does not fire, the
   slots keep the first tensor's data for every layer after it).
Next step is a one-line counter on the hook (fires per consumer?) which I could not get compiled cleanly in
this pass; everything else is in place.

State: fully opt-in and default-off. `LLAMA_W_STAGE_ENABLE` + `LLAMA_W_STAGE_BLOCKS` must both be set; with
them unset the path is dead code, and the default is verified unchanged twice after these edits
(8K: 52.60 t/s, sha `d0fb7092955c105a`; 200K with `LLAMA_FA_Q4_SPLIT_TOKENS=2048`: 15.62 t/s, sha
`f2a78a3f928f21d7`). No commits, no deletes.

## 55. Q4 weight streaming works: IQ3_S with no CPU layers (8K 19.76 -> 34.4-49.7 t/s), 64K 13.15 -> 23.45

Worker2, 2026-10-21. `D:\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, MTP d4 (and DFlash d7 at 8K), 128/512
greedy tokens, `-b 512 -ub 32`, `-fa on` q4_0 KV. Opt-in: `LLAMA_W_STAGE_ENABLE=1`,
`LLAMA_W_STAGE_BLOCKS=<layers>`, `LLAMA_W_STAGE_SLOT_MIB=48`, `LLAMA_W_STAGE_PREFETCH=1`; with them unset
the path is dead and the default is unchanged (re-verified twice with the same binary: 8K 19.61-19.76 t/s,
same sha; 64K baseline sha unchanged).

### 55.1 Why the previous attempt produced 0/242 accepted drafts

The copies were always byte-exact; the bug was hook coverage, in three parts:
1. Small tensors (norms, conv1d, `ssm_alpha/beta`, `ssm_a`, `ssm_dt.bias`, ~1.1 MiB per layer) were adopted
   into shared slots but consumed inside fused kernels (rms_norm+mul, gated_delta_net/ssm_scan cache
   fusion), which never reach the per-node hook: they read the previous tenant's bytes. Proof: the old code
   with `GGML_CUDA_DISABLE_FUSION=1` reproduced resident output exactly (sha `5cddc0edb7db543b`), with
   fusion on it did not. Fix: adopt only >=1 MiB 2-D tensors; the small ones stay where the loader put them.
2. The MMVQ+GLU fusion reads TWO weights in one kernel (`ffn_up->src[0]` and `gate_n->src[0]`,
   ggml-cuda.cu ~4390-4470) and bypasses the hook; in adoption order those two tensors also landed in the
   SAME slot. This was not in the review and survived the first fix (token-18 loop). Fix: `try_fuse()`
   returns 0 when any node in its 8-node window reads a streamed tensor (streaming mode only).
3. Pass boundary: `begin_pass()` refilled slot 0 after waiting on an event recorded before the previous
   pass's last consumer. Now `begin_pass` records `consumed` on the compute stream first, and is skipped for
   graphs that read no streamed tensor (MTP/draft/head graphs), which also removes a pointless prefetch per
   draft pass.
`GGML_CUDA_DISABLE_GRAPHS=1` (not `LLAMA_CUDA_DISABLE_GRAPHS`, which silently leaves capture on) was
mandatory until P3; under capture the hook's stream wait failed with "CUDA error: invalid argument".

### 55.2 Slot pairing and the spread: the whole 8K result

The 12 big matmuls of blk.0+blk.1 (265 MiB) in 2 x 70 MiB slots do NOT fit at 8K IQ3_S (OOM, 11555 MiB
after load). Keeping those layers resident needs their 655 MiB back, so the streamed set has to include
currently-resident tensors and they must be SPREAD:

| 8K config (ctx 16384, KV resident, 128 tok) | t/s | step ms | eval/sync/draft | peak |
|---|---:|---:|---|---:|
| baseline, blk.0/1 on CPU, KV staged | 19.76 | 210/178 | 152/33, 128/28 | 11657 |
| streamed 8 layers x 3 ffn (904 MiB), copy-on-demand | 22.19 | 185/183 | 103/56, 107/52 | 11081 |
| same, KV resident | 22.54 | 181/182 | 94/64, 99/58 | 11329 |
| spread every 3 layers (21 x ffn_down, 804.8 MiB), copy-on-demand | 23.73 | 172/174 | 99/52, 98/51 | 11429 |
| **+ prefetch (`LLAMA_W_STAGE_PREFETCH=1`)** | **34.43** | 118 | 64-69/26-30 | 11429 |
| same, repeat | 34.20 | | | |
| **+ CUDA graphs (P3)** | **36.34** | 110 | 1.7/85 | 11467 |
| **+ DFlash d7** | **46.24** | 119-128 | 70/33-36 | 11637 |
| **+ DFlash d7 + graphs** | **49.70** | 114-124 | | 11681 |
| DFlash d6 (graphs off) | 42.72 | 124-129 | | 11631 |

Three ffns per layer are consumed back-to-back, so with 2 slots the 2nd/3rd copy cannot start before the
previous consumer finishes: prefetch there was worth +2% only. Spread one tensor every ~3 layers and each
48 MiB copy (~3.5 ms) hides under ~4.5 ms of layer compute; prefetch then removes ~25 of the ~51 ms of DMA
exposure. That is the reviewer's "spread the set" point, and it is what makes P2 pay.

### 55.3 Exactness

All streamed rows are token-identical to their resident/CPU references:
- 8K, 128 tok: `7179cfb5bc6675e6` for the CPU-layer baseline, spread, prefetch, and graphs-on rows.
- 8K, 512 tok with `LLAMA_FA_Q4_SPLIT_TOKENS=16384`: `f3c606d432bfd2c2` for no-spec, MTP d4 (500/385),
  DFlash d7 (605/424) and DFlash d6 - so spec == no-spec holds on the streamed config.
- 64K, 128 tok: `3ebc58afe4d85a68` for the CPU baseline, S=1212 and S=612.
- A same-config resident-vs-streamed A/B at IQ3_S (layers 8+24 ffn resident vs streamed, blk.0/1 on CPU):
  identical sha, 19.89 vs 18.59 t/s - also prices a fully exposed copy at +18 ms/step for 236 MiB.

Snapshot caveat: a restored slot is NOT bit-exact (the 64K save restored into the CPU-layer config gave
`0f124dfd474647f7` with the same 131/94 acceptance), so snapshots are used for VRAM/step probes only.

### 55.4 64K and 128K

| 64K (t64k.txt, 65532 tok, ctx 65792, KV 16/16 staged, 128 tok) | t/s | step ms | eval/sync | peak | prefill |
|---|---:|---:|---|---:|---:|
| baseline, blk.0/1/2 on CPU | 13.15 | 233/300 | 167/45, 233/46 | 11649 | 495 s |
| streamed S=1212 MiB (31 x ffn_down) | 18.06 | 212/222 | 113/78, 122/78 | 10891 | 386 s |
| streamed S=612 MiB (16 x ffn_down) | **23.45** | 164/172 | 79/71, 84/59 | 11493 | 368 s |

The PCIe bill per step is (streamed weights + staged KV), and both are unavoidable. At 64K all-staged KV is
16 x 72 = 1152 MiB/step, so streaming 1212 MiB of weights just adds 91 ms of copies; the minimum set that
still fits the GDN layers (612 MiB) is worth +30% over the bigger set. Making a KV layer resident costs
72 MiB of VRAM to save 72 MiB of DMA - the same exchange rate as one more streamed weight tensor - so the
rule is: smallest streamed set that fits, KV stays staged.

128K (t128k.txt, 131,065 tok, ctx 131584, KV 16/16 staged, 128 tok, MTP d4, S=612, user-approved native
prefill, 897 s = 146 t/s prefill):

| 128K config | t/s | ms/step | tok/step | peak | sha |
|---|---:|---:|---:|---:|---|
| baseline, blk.0/1/2 on CPU (restore) | 10.07 | 371.1 | 3.76 | 11585 | b1423a923497e2e9 |
| streamed, graphs off (native prefill) | **15.02** | 241.6 | 3.66 | 11641 | b1423a923497e2e9 |
| streamed, graphs on (restore) | 14.87 | 251.2 | 3.76 | 11439 | b1423a923497e2e9 |

+49% over the CPU-layer baseline. At 128K the step is PCIe-bound: all-staged KV is 16 x 144 = 2304 MiB
(~173 ms) plus 612 MiB of weights (~46 ms), so the graph launch saving is invisible (14.87 vs 15.02, i.e.
graphs on == off) - unlike 8K, where the copies have compute to hide under. Note the streamed native run
and both restored runs share one sha: restoring a snapshot saved by the same model/placement reproduces the
token stream exactly; the earlier mismatches came from a foreign (IQ3_XXS-era) snapshot and from restoring
a streamed save into the CPU-layer placement.

### 55.5 P3 recipe (CUDA graphs) and the v8 gate

Capturing the refills but not the prefetch is correct but slower (23.75 t/s, sync 144 ms: every memcpy node
serializes with its consumer). Two changes make graphs a win (+5.5% MTP d4, +7.5% DFlash d7): `begin_pass`
does bookkeeping only when graphs are on (a pre-capture event cannot be depended on by a captured stream;
the first fill is emitted inside the capture by its consumer's hook), and the prefetch is emitted inside the
capture so the graph itself overlaps the copies with compute.

MMQ repack v8 microbench (`tests/bin/mmsb_bench_v8.exe --v8 --reps 20`, output in
`comm/files/seq74_v8_run.txt`): correctness clean (0 bit diffs, rel_err 1.4e-7, column invariance 0), but
ncols=8 x-of-floor is 2.29/3.23 (IQ3_S) and 2.71/3.81 (IQ3_XXS) - worse than in-tree MMQ@8 (2.07/2.04) and
far past the pre-set 1.60 abandon line; ncols=16 is worse than ncols=8, i.e. issue-bound. Repack route
abandoned.

Files: results/w2/*.json + .server.log, tests/w2_dflash.py, comm/files/seq72_gate.txt, seq73_p1p2.txt,
seq74_contexts.txt, seq74_v8_run.txt.

Numerics, measured with `logprobs` (8 tokens, top-8, same prompt/config): streamed vs resident of the SAME
tensors is **bit-identical** - 80/80 fields equal (token, prob, every top_logprob, 8 steps; P1a pair).
CPU-layer placement (blk.0/1 on CPU) vs GPU is **not**: step-0 top-1 logprob -0.6643185019493103 (CPU) vs
-0.6130914688110352 (GPU), while the 512 greedy tokens coincide - so the equal sha across those two
placements is token-level, not bitwise, and should not be quoted as "bit-exact". KV staged vs resident is
numerically neutral (80/80 identical), so the CPU/GPU difference is the GDN compute path, not staging.

8K streamed-set sizing (graphs on): S=390 MiB OOMs; S=612 gives 36.98 t/s MTP d4 (peak 11661) vs 36.34 at
S=805; DFlash d7 needs S=805 (S=612 OOMs with the 561 MB drafter) and then reaches 49.70.

### 55.6 Multi-prompt check, 8K IQ3_S (report 41.1 prompts, 512 greedy tokens) - CORRECTED

Prompts `results/prof/t6_8k_q1..q5.txt` (12.3K tokens), ctx 16384, MMQ, `LLAMA_FA_Q4_SPLIT_TOKENS=16384`.
CORRECTION (reviewer seq 80): the first version of this table reported `tok/step` from the STEPTIME `ops`
column, which is the *verify batch size* (proposed tokens per step), not the accepted rate. Corrected here:
`tok/step = predicted_n / decode steps`, `ms/step = predicted_ms / decode steps`, and
`t/s == tok/step / ms/step * 1000` is checked for every cell (all within 0.2%). The old column
(4.96, 4.97, 7.84) was proposed tokens/step; the accepted values are 3.17-3.72 and are below.

"base" = blk.0/1 on CPU, KV staged 16, MTP d4, -ub 32. This placement cannot fit the 561 MB DFlash drafter
at all (the draft model fails to load: "unable to allocate CUDA0 buffer"), so DFlash at 8K exists only
because weight streaming frees the VRAM. "stream" = spread S=805 MiB, blk.0/1 resident, KV resident,
prefetch, CUDA graphs on.

| config | q1 | q2 | q3 | q4 | q5 | mean t/s | mean ms/step | mean tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base MTP d4, ub32 | 14.05 | 17.11 | 13.51 | 15.17 | 13.07 | **14.58** | 216.9 | 3.17 |
| stream MTP d4, ub32 | 28.72 | 32.40 | 25.05 | 29.17 | 24.43 | **27.95** | 117.7 | 3.29 |
| stream MTP d4, ub128 | 20.45 | 30.90 | 25.76 | 27.03 | 22.47 | 25.32 | 125.1 | 3.14 |
| stream DFlash d7, ub32 | 26.89 | 29.69 | 26.66 | 33.77 | 25.62 | 28.53 | 130.1 | 3.72 |
| stream DFlash d7, ub128 | 28.72 | 35.82 | 28.73 | 30.70 | 25.26 | **29.84** | 119.0 | 3.56 |

Readings (same-ub rows share one sha per prompt, so those comparisons are at identical text):
- base -> stream (MTP d4, ub32): t/s 14.58 -> 27.95 = 1.92x = step time 216.9 -> 117.7 (1.84x) x accepted
  tok/step 3.17 -> 3.29 (1.04x). The win is step time; acceptance is essentially flat (+3.8%, and the text
  differs because the GDN layers moved CPU->GPU, so this is the lottery, not a streaming effect).
- MTP vs DFlash at ub32: 27.95 vs 28.53 (+2.1%) - DFlash buys acceptance (3.29 -> 3.72) but pays step time
  (117.7 -> 130.1).
- MTP vs DFlash at ub128: 25.32 vs 29.84 (+17.9%) - DFlash is both faster per step (119.0 vs 125.1) and
  higher acceptance (3.56 vs 3.14).
- Best -ub per mode: MTP prefers ub32 (27.95 vs 25.32, +10%); DFlash prefers ub128 (29.84 vs 28.53, +4.6%).
  Per-prompt t/s across ub values is not comparable (different sha), so only same-ub cells are quoted as
  like-for-like.
Worker1's tables were spot-checked for the same formula error and are consistent (27: 4.27/0.1444 = 29.6 vs
29.32; 28: 5.45/0.2024 = 26.9 vs 26.86; 41.1: 3.93/0.1624 = 24.2 vs 24.22 and 3.86/0.1047 = 36.9 vs 36.89;
42: 6.08/0.289 = 21.0 vs 21.02).

### 55.7 DFlash d7 at 64K/128K via `--budget`, re-baselined under the GPU lock, and the fusion caveat

Worker2, 2026-10-21 (seq 106). Every run below was taken with `comm\GPU_LOCK` held for the whole run and
`nvidia-smi --query-compute-apps` empty at launch (recorded per run in `results\w2\s106_*.lock.txt`), so the
rows in 55.4 and in the earlier DFlash attempt are now re-measured uncontended. Same model, prompts, env and
`-b 512 -ub 32 -fa on` q4_0 KV as 55.4; decode rows restore the snapshots saved there (same model + placement),
which is valid here: the restored 64K text is `4f3bce79f1efe976` and the 128K text `b1423a923497e2e9` for every
placement tried (S=612/927/1598, KV 16/16 and 14/16 staged), so all rows of one context share one text.

**(a) `LLAMA_KV_GPU_BUDGET_MIB` is a CEILING, not a target.** The loop (`llama-kv-cache.cpp` ~205-220) picks the
SMALLEST staged count c in 1..cache_layers-2 whose resident bytes + 2 staging slots fit in the budget. So a small
budget stages MORE: at 64K (72.28 MiB/layer, 1156.5 MiB total) budget 1000 MiB gives **5/16** staged and leaves
939.6 MiB of KV resident (log: "staged KV: 5/16 attention layers ... 195.00 MiB dedicated VRAM free"), while
budget 300 MiB gives **14/16** and 289.1 MiB of KV. Consequence: the DFlash drafter only fits with a small budget,
and budget mode can never stage all 16 layers (`cache_layers - 2` cap) - at 64K its KV floor is 2 x 72.28 +
2 slots = 289.1 MiB vs 144.6 MiB for `--staged 16`.

**(b) The `--staged` crash is the flag, deterministic, and not contention.** `--staged 16 --mode dflash` at 64K,
alone on the card, fails at context creation:
`E llama_init_from_model: failed to initialize the context: staged KV requires GPU offload, flash attention,
-np 1, and a non-SWA/non-MLA cache without cross-layer sharing`. Cause confirmed from the drafter GGUF:
`dflash.attention.sliding_window = 2048`, `sliding_window_pattern = [T,T,T,T,T]`, 5 KV layers, so
`min(staged_requested, 5) = 5 > 0` and the `n_swa` term of the same check trips. Budget mode never reaches that
branch because the drafter's whole SWA cache (12.66 MiB) is below any budget we use, so it stays resident.

**(c) 64K, DFlash d7 via budget.** Budget 300 MiB (14/16 staged), S=612 MiB = the 16 attention-layer ffn_down
(`2,6,...,62`), 128 tok, ctx 65792. NOTE: every row in this subsection is restore-based; 58.1 later measured that the
S=612 set cannot serve a REAL request (the prefill compute buffer does not fit) and re-sized the production 64K profile
to S=797 with -ub 32, which decodes 23.65 t/s at 64K context.

| 64K, 128 tok, restore text `4f3bce79f1efe976` | t/s | ms/step | tok/step | peak |
|---|---:|---:|---:|---:|
| MTP d4, `--staged 16`, S=612 (55.4 row, graphs off) x2 | 21.73 / 21.68 | 167.0 / 167.3 | 3.657 | 11293 |
| MTP d4, `--staged 16`, S=612, graphs on x2 | 21.41 / 21.40 | 169.4 / 169.6 | 3.657 | 11293 |
| MTP d4, budget 300 (14/16), S=612 | 22.92 | 158.3 | 3.657 | 11439 |
| **DFlash d7, budget 300 (14/16), S=612, graphs off x2** | **25.33 / 25.36** | 156.7 / 156.5 | 4.000 | 11643 |
| DFlash d7, same, graphs on | 25.18 | 157.6 | 4.000 | 11651 |
| no-spec, budget 300, S=612 (same placement) | 7.55 | - | - | 10885 |

The 23.45 row of 55.4 reproduces on step time (164.1 -> 167.0 ms/step, +1.8%) and was therefore NOT inflated by
contention; its higher t/s is the native-prefill text (3.879 accepted tok/step vs 3.657 for the restored text).
DFlash d7 beats the step-1 MTP row by +16.8% (25.35 vs 21.70), but 144 MiB/step of that is placement (budget mode
has 144 MiB less KV DMA): at EQUAL placement the win is +10.6% (25.35 vs 22.92) and comes from acceptance
(4.000 vs 3.657, +9.4%) with step time flat (156.6 vs 158.3). DFlash peak 11643 = 645 MiB free; S=612 is the
smallest workable set with the drafter: S=539 (2 fewer ffn_down) loads and then OOMs on the first eval
(`CUDA error: out of memory`), S=466 fails while loading the drafter (524.8 MiB request).

**(d) 128K, DFlash d7 via budget.** Budget 600 MiB (14/16 staged, KV 578.2 MiB), S=927 MiB = the same 16 attention
ffn_down plus the GDN-layer ffn_down of `0,4,8,12,16,20,24,28,32`, ctx 131584, 128 tok.

| 128K, 128 tok, restore text `b1423a923497e2e9` | t/s | ms/step | tok/step | peak |
|---|---:|---:|---:|---:|
| MTP d4, `--staged 16`, S=612 (55.4 row) graphs off x2 | 14.95 / 14.96 | 257.4 / 257.2 | 3.879 | 11441 |
| MTP d4, `--staged 16`, S=612, graphs on x2 | 14.83 / 14.79 | 259.5 / 260.1 | 3.879 | 11441 |
| MTP d4, budget 600 (14/16), S=927 | 14.70 | 261.8 | 3.879 | 11415 |
| **DFlash d7, budget 600 (14/16), S=927, graphs off x2** | **15.12 / 15.18** | 262.5 / 261.4 | 4.000 | 11621 |
| DFlash d7, same, graphs on | 15.11 | 262.6 | 4.000 | 11627 |
| no-spec, budget 600, S=927 (same placement) | 4.18 | - | - | 10863 |

The 15.02 row of 55.4 reproduces (15.02 -> 14.955, step 256.3 -> 257.4 ms). DFlash d7 is +1.3% over it and +3.1%
over MTP at the same placement; step time is identical (262 vs 261.8 ms/step) and the whole difference is
acceptance (4.000 vs 3.879 tok/step). DFlash at 128K needs the larger set: S=612 leaves 303 MiB free and fails on
the 524.8 MiB drafter; S=797 loads the drafter but misses its 12.66 MiB KV cache by that much; S=927 fits with
~660 MiB free. At 128K the budget route costs MTP 1.7% (S 612 -> 927 = +24 ms/step) and buys DFlash the ability to
run at all, which `--staged` cannot - the -27% KV DMA (14 layers staged instead of 16) nearly cancels the extra
weight DMA (+315 MiB/step).

**(d2) Streamed-set floor, measured (same lock discipline).** The floor is set by the drafter, and both contexts sit
within one tensor of it:

| ctx | streamed set | S MiB | DFlash d7 t/s (x2) | peak | smallest that ran | next smaller fails |
|---|---|---:|---|---:|---|---|
| 64K | 16 attention ffn_down (2,6,...,62) | 612 | 25.33 / 25.36 | 11643 | S=575 (drops layer 6): 25.52 t/s, peak 11681 | S=539: loads, `CUDA error: out of memory` on the first eval |
| 128K | every other layer to 32 (0,2,...,32) then every 4th | 927 | 15.12 / 15.18 | 11621 | S=863 (GDN 0,8,16,24,32,40,48): 15.22 / 15.28, peak 11685 | S=826: loads, `CUDA error: the resource allocation failed` on the first eval; S=817: `graph_reserve: failed to allocate compute buffers` at load |

So the smallest sets that run are S=575 at 64K and S=863 at 128K, worth +0.7% each over S=612/927; both leave <100 MiB
free at eval peak, which is why they are quoted as measured points and not as targets. 128K MTP at the same placement:
S=927 14.70, S=863 14.79 (vs DFlash 15.15 / 15.25), i.e. DFlash is +3.1% over MTP at BOTH 128K placements, and the
budget run's t/s does not depend on which of the two sets is used.

**(e) Exactness.** Every 64K row above is token-identical (`4f3bce79f1efe976`) across 
`--staged 16`/budget 300, S=612/1146/1005, graphs on/off and MTP/DFlash/no-spec; every 128K row is
`b1423a923497e2e9` likewise. So spec == no-spec holds at both contexts in the budget placement, and the DFlash
sha equals the no-spec sha at 64K as required.

**(f) Fusion caveat (limits every "streamed == resident" claim to equal fusion state).** `ggml_cuda_try_fuse()`
returns 0 whenever `ggml_cuda_w_stage_window_has(cgraph, i, 8)` - any of the 8 nodes in the fusion window reads a
streamed tensor - so growing S changes the KERNEL MIX, not only the DMA bill: more streamed tensors = the MMVQ+GLU
fusion is skipped on more layers, so two runs at different S are not the same computation even when both are
token-identical to the resident reference. The 55.3 A/B (resident vs streamed of the SAME tensors, bit-identical
logprobs) is only valid at that equal fusion state. Making the fused launch keep working under streaming needs the
slot-ready refill to happen before the fused kernel (the slot address is already valid at that point; the fuse
currently returns before reaching the per-node hook) plus the existing pair-aliasing guard when the GLU reads two
streamed tensors - not implemented, not measured.

**(g) Sync (DMA wait) share of the step**, mean over decode steps from `STEPTIME` (same runs, graphs off):
64K MTP staged-16 59.1 of 167.0 ms (35.4%), 64K DFlash budget 53.8 of 156.8 (34.3%), 128K MTP staged-16
103.5 of 257.5 (40.2%), 128K MTP budget-600 90.6 of 261.9 (34.6%), 128K DFlash 90.7 of 262.6 (34.5%). At 128K the two
KV layers staged instead of resident (289 MiB) buy back the +315 MiB of extra weight DMA almost exactly, while `eval`
grows 130.1 -> 147.9 ms when S goes 612 -> 927 (fusion state, not copies). Both 128K DFlash sets sit at ~34.5% sync, so
neither is close to making the step compute-bound.

**(h) What lifting the fusion guard can buy: bounded at +3.4% (8K) / <1% (64K, 128K), so not done.** The guard
(55.7 f) is the only reason the streamed layers lose the GLU fusion, so the win it could return is bounded above by the
value of ALL fusions on that config. Measured with `GGML_CUDA_DISABLE_FUSION=1`, same placement, same sha, 128 tok:

| config | fusion on | fusion off | bound |
|---|---:|---:|---:|
| 8K, DFlash d7, S=805 (21 x ffn_down), KV resident, graphs on | 51.76 t/s | 50.08 t/s | +3.4% |
| 64K, DFlash d7, budget 300, S=612 | 25.33 / 25.36 | 25.18 | +0.6% |
| 128K, DFlash d7, budget 600, S=863 | 15.22 / 15.28 | 15.21 | <=0 (noise) |

Both 64K and 128K are PCIe-bound (55.7 g: sync 34-35% of the step, and the compute part is dominated by the 512-token
verify matmuls that are not fused in either case), so recovering the skipped GLU fusion would be worth under 1% there
- less than the +0.7% that simply picking a one-tensor-smaller streamed set already gives. At 8K it is worth at most
3.4% and would need the slot refill emitted before every fused launch plus the slot-pair rule (two streamed weights in
the same fused kernel must sit in different slots, which adoption order does not guarantee). Cost/risk: a change in the
fused MMVQ path, whose failure mode (55.1) is silent wrong tokens, for a bound of <1% at the two contexts this section
is about. Decision: not implemented; the entry above is the measured ceiling if anyone wants it later.

Files: `results/w2/s106_*.json` + `.server.log` + `.lock.txt` (lock line and `nvidia-smi` per run),
`tests/w2_run106.sh` (runner), `tests/w2_tab106.py` (table above from the JSONs).

## 56. Q3 int8 prefill, first gate: 2.0-2.1x over MMQ, but 1.19% added error with per-row scales

Worker2, 2026-10-21. Standalone microbench, no production path: `tests/int8_fwht_bench.cu`
(`tests/build_int8_fwht_bench.bat`, exe in build-cuda\bin). One real matrix: IQ3_S `blk.9.ffn_gate`
dequantized to f32 (17408 x 5120, 89.1 M params - the same shape as report 20), `results/w2/w_ffn_gate_blk9.f32`.
Activations are synthetic but outlier-shaped (0.35% of channels at 30x, the structure the rotation is for);
that is a limitation, not a claim. Reference = cuBLAS f32 GEMM of the SAME dequantized weights, so the
number below is the ADDITIONAL error the int8 path introduces (MMQ's own error in this setup is ~0: it
computes with those same weights; its error against the original fp16 weights needs the fp16 model, which
we do not have).

| path | 512 tok | TOPS | 2048 tok | TOPS | rel_err vs f32 GEMM |
|---|---:|---:|---:|---:|---:|
| int8, per-row W + per-token X, no rotation | 1.28 ms | 71.4 | 4.85 ms | 75.3 | **7.51%** |
| int8 + FWHT block-1024 (W rows and X tokens rotated identically) | 1.28 ms | 71.3 | 4.80 ms | 76.1 | **1.19%** |
| fp16 tensor cores (reference) | 2.15 ms | 42.4 TFLOPS | 8.51 ms | 42.9 TFLOPS | - |
| MMQ iq3_xxs (report 20.2, same shape) | - | - | 9.75 ms | 37.4 TFLOPS | - |

Readings:
- Speed: int8 is **2.03x** MMQ's time at 2048 tokens (4.80 vs 9.75 ms) and 2.0x fp16; 71-76 TOPS. The
  epilogue and scales are included. So the 2.4x vendor-int8 headroom from report 21 is real at these shapes.
- Accuracy: the rotation is not optional - 7.51% -> 1.19% (6.3x better) at identical time; without it a
  single 30x channel destroys the per-token scale. But 1.19% added relative error on every ffn matmul is
  large: it is on top of IQ3_S's own quantization noise, so it would very likely cost measurable KLD.
- The cause is the scale granularity that a single cuBLAS int8 GEMM allows: one scale per weight row and
  one per token. Per-128-element weight groups are the standard fix, but they need ~40 separate
  accumulation groups (K/128), i.e. either 40 small GEMMs or an epilogue that can read per-group sums -
  neither is a single cuBLAS call.

Gate verdict: speed gate PASSES (2x), accuracy gate is MARGINAL/NOT PASSED at 1.19%. Next step, if the
user wants to spend GPU time, is a KLD run of a per-row int8 ffn path against the default (opt-in, default
off) to see whether 1.19% is tolerable in practice; otherwise the recipe needs per-group scales, which is a
different (and much larger) kernel.
Files: tests/int8_fwht_bench.cu, tests/build_int8_fwht_bench.bat, results/w2/w_ffn_gate_blk9.f32.

## 57. v9 paired-layout IQ3 route (worker3 seq 98-105): exact, wins in isolation, loses in-model -> shelved

Opt-in only (`GGML_CUDA_MMSB=1`, default off, default path byte-identical): IQ3_S/XXS weights are repacked in place into a
paired 512-k tile-block layout (0 bytes added) and a 3-stage v9 kernel (quant, mma, combine) serves ncols 1..16; above 16
MMQ reads the same packed bytes. Tree `llama.cpp-w3`, build `build-cuda-w3`.
Loads and the default path
- Loader d-bug: both mmsb loaders read `d` from the *step* chunk; `d` lives in the chunk of row `(half<<3)|quad` at 192
  (IQ3_XXS) / 216 (IQ3_S). Fixed with a `pcd` pointer. Bitwise proof: perplexity natural vs packed(V9_MAX_NCOLS=0) PPL
  8.5151 both, saved logits byte-identical (md5 4e888290d8fe6272b7f3f7340c30317f), KLD at the f16 storage floor.
- Default-path inlining regression found and fixed: `mul_mat_q_process_tile` chose the load-tiles function through a
  runtime pointer, so nvcc stopped inlining the natural loader; both are `constexpr` now. w3 OFF 24.4 -> 20.7 ms/token,
  PP 420 -> 442 t/s (residual w3-vs-main ~2%; `mmsb_pair` deliberately not templated).
Isolated bench (`tests/mmsb_bench_v9.cu --pair`)
- vs production MMQ@8: -16.6% (IQ3_S) / -19.6% (IQ3_XXS) byte-weighted; vs MMQ@1: -22.0% / -24.5%. Caveat: the bench's
  default-mode `k_mma` baseline is ~1.75x slower than production MMQ (ffn_down 17408x5120 IQ3_S 558 vs 319 us), so only
  mmvq_cols_bench4/production MMQ may serve as the reference.
- Never-benched small shapes, n=1 us at best split: attn_k 5120x1024 IQ3_S 35.6 (split 4) vs MMQ 20.2; attn_v IQ3_XXS
  57.3 vs 78.8; attn_q 5120x12288 IQ3_XXS 132.1 (split 2); ssm_out/attn_output 6144x5120 IQ3_S 90.1 (split 2) vs MMQ
  ~109; attn_gate 5120x6144 85.0.
- In-tree config vs bench: grid identical (sms*8 = 224 blocks, 128 thr); split is 4 at out <= 6144 and 2 above, while the
  bench prefers 2 at out 5120/6144 (4-8% off its own best) - near-, not mis-configured.
In-model, CUDA graphs ON, n=1 no-spec, w3 build
- d7 gate w3 OFF 106.1 vs ON 107.9 ms/step; n=1 79.2 vs 81.7 ms/token (+3.0%).
- Per class (`GGML_CUDA_MMSB_ONLY` one class at a time, TG delta = that class's cost; 128 tokens, two OFF anchors):
  ffn_up +0.53, attn_v +0.43, ffn_down +0.42, ssm_out +0.39, attn_k +0.37, attn_qkv +0.32, ffn_gate +0.28, attn_gate
  +0.20, attn_output +0.15, attn_q -0.14 (+-0.15 each) ms/token; all-ON +2.4. No class is measurably faster, so
  per-shape routing recovers <0.4% - below the 2% gate. Exactness (all-ON): d7 sha = d3 sha = no-spec sha =
  7d46062ffda13194.
Why the seq-103 table was invalid, and the open lead
- It ran with GGML_CUDA_DISABLE_GRAPHS=1 (cudaEvent syncs are illegal in capture) plus a sync between every launch. On
  WDDM that puts a host round-trip before every kernel, and v9 is 3 launches vs MMQ's 1-2: hence +13.8 ms/token claimed
  against +2.5 end-to-end, and 359 us/call for ffn_down against 191-249 us in the bench. Relative pattern only.
- nsys ground truth was unobtainable here: `nsys profile` as non-admin fails (`Connection to Agent lost`, target never
  launches, `cuda_gpu_kern_sum` empty); graphs-on per-kernel sums need an elevated session.
- Open lead: the bench promises 22-25% at n=1, in-model is +3%. Every class pays ~+10-18 us/call and no single class
  accounts for it, so it is a per-call in-model penalty, not shape under-fill - candidates are the extra quantize stage
  (the pair bench times mma+combine only), 3 graph nodes vs 1 (WDDM node launch), and a clock/occupancy difference vs the
  tight bench loop. attn_k/attn_v were the never-benched shapes and are not the loss (attn_k +15 us isolated, ~0.03
  ms/token in-model).

## 58. Real-world checks through llama-server: the configs work, -c 204800 costs 16% PP / 10% decode at 4K, and the 64K profile had to be re-sized (S=797, -ub 32)

Worker2, 2026-10-21 (seq 110). Reviewer's phase B (seq 107/108): not microbenchmarks but a server, started the way a
user would start it, driven over the OpenAI API (`/v1/chat/completions`, streaming), GPU lock held for every launch.
Two new files: `tests\serve_prod.ps1` (three profiles, every flag cited to the report row that measured it) and
`tests\realworld_chat.py` (scenarios 1-5; per request it records prompt tokens, cached tokens, prompt ms, TTFT, the
server's own decode t/s, draft/accepted, finish reason, and VRAM sampled in the background). Main build
(`llama_cpp_src\build-cuda`), 12 GiB RTX 3060. Raw JSON + server logs: `results\rw\*`.

### 58.1 The IQ3_S 64K profile did NOT work as configured: S=612 fits only a snapshot restore

Every 55.7 row was measured after `POST /slots/0?action=restore`, which skips the prefill graph. A real request has to
allocate it, and with 645-667 MiB free it cannot:

| 64K streamed set, budget 300, DFlash d7 | first real 12K prefill |
|---|---|
| S=612 (55.7's row, 25.35 t/s decode) | OOM at once: `CUDA error: out of memory` (harness and launcher alike) |
| S=736 | OOM during the first prefill |
| **S=797** (16 attention ffn_down + GDN ffn_down of 0,4,8,12,16,20) | works, 629 MiB free at peak, prefill 12,265 tok in 57 s (215 t/s), decode 27.4 t/s |
| S=1146 | works, 25.35 t/s (only 979 MiB free), 20.29 t/s decode at 64K context |

Production set is therefore S=797, and `-ub 32` is required, not cosmetic: at `-ub 128` the same set dies in the first
prefill (`CUDA error: the resource allocation failed`, peak 11711 MiB), because the prefill compute buffer is sized by
the ubatch. Decode at 64K context: **23.65 t/s** (restore, text `4f3bce79f1efe976`); at 12K context 27.4 t/s. The
launcher carries the profile's own -b/-ub for exactly this reason. This is a correction to the "S=612 is the production
config" reading of 55.7 - the 25.33/25.36 rows there are decode-only and still correct.

### 58.2 Scenario 1: the same 4K-context prompts at -c 32768 vs -c 204800 (plong, cold cache)

Eight varied prompts (code review, arithmetic check, summarise, translate, JSON extraction, long-form, code fix,
number check), about 4.2-5.6K prompt tokens each, max_tokens 512, temperature 0, one server per context, everything
else identical:

| prompt | PP -c 32768 | PP -c 204800 | PP delta | decode 32K | decode 200K | decode delta |
|---|---:|---:|---:|---:|---:|---:|
| code-review | 432.4 | 368.3 | -14.8% | 25.74 | 23.81 | -7.5% |
| math-reasoning | 449.9 | 379.4 | -15.7% | 34.49 | 32.96 | -4.4% |
| summarise | 448.0 | 376.9 | -15.9% | 37.16 | 32.23 | -13.3% |
| translate | 450.6 | 377.8 | -16.1% | 31.76 | 26.40 | -16.9% |
| json-extract | 446.3 | 375.6 | -15.8% | 60.24 | 56.09 | -6.9% |
| long-form | 450.4 | 376.0 | -16.5% | 26.61 | 24.41 | -8.3% |
| code-fix | 441.3 | 373.1 | -15.5% | 33.17 | 30.59 | -7.8% |
| number-check | 446.8 | 376.3 | -15.8% | 41.03 | 35.97 | -12.3% |
| **mean** | **445.7** | **375.4** | **-15.8%** | **36.28** | **32.81** | **-9.6%** |

Peak VRAM 11145 MiB at 32K vs 11475 MiB at 200K. So the user's goal - the same speed at 4K - is NOT met: a 200K-context
server is 15.8% slower at prefill and 9.6% slower at decode on the same 4K prompts. The cause is not `-c` itself but the
VRAM budget: at 32768 the whole KV (16 x 36.1 = 578 MiB) is below the 1000 MiB budget, so nothing stages and both
prefill and decode run resident; at 204800 the budget forces 12/16 attention layers onto pinned host (KV resident part
578 MiB), which reproduces section 50's 17-19% staged-KV prefill penalty and adds KV DMA per decode step. Practical
answer for the user: keep a short-context server for short chats; the 200K server is for the long requests.
`LLAMA_FA_VEC_Q4_COLS`-class env is identical in both runs, so this is not a flag accident.

### 58.2b Third arm: the -9.6% decode is the split constant, the -15.8% prefill is the staging

Same eight prompts, same profile, `-c 204800` plus `LLAMA_FA_Q4_SPLIT_TOKENS=2048` (the constant that fixed the
`-c` dependence in 46.4; the forks default leaves it unset):

| arm | prefill t/s (mean) | decode t/s (mean) | vs -c 32768 |
|---|---:|---:|---|
| plong -c 32768 | 445.7 | 36.28 | - |
| plong -c 204800, default split | 375.4 | 32.81 | -15.8% PP, -9.6% decode |
| plong -c 204800, `LLAMA_FA_Q4_SPLIT_TOKENS=2048` | 376.2 | **36.54** | -15.6% PP, **+0.7% decode** |

So the decode deficit was the split partition (set `LLAMA_FA_Q4_SPLIT_TOKENS=2048` on any long-context server and it
disappears: 32.81 -> 36.54, +11.4%), while prefill stays 15.6% behind -c 32768 at any split constant, because that part
is the staged-KV DMA per ubatch and not a partition effect. Production plong should therefore carry the split constant;
the prefill penalty is the price of the 200K ceiling and is only avoidable with a second short-context server.

### 58.3 Scenario 2: growing multi-turn conversation to ~200K (plong, -c 204800)

Real turns: each appends ~10K tokens of actual repo text (report sections, ggml-cuda.cu, llama-kv-cache.cpp, ...) plus
a question, and the assistant's reply is kept in the history, so the conversation genuinely grows. max_tokens 256.

Prefix reuse was exact on every turn: `cached_tokens` equals the previous turn's prompt size, so only the new text
prefilled (`new tok` below); the first turn is cache-cold.

| turn | ctx ~ | prompt tok | cached | new tok | new-turn latency | decode t/s | acc/step | VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 20K | 29,196 | 15,273 | 13,923 | 43.0 s | 24.80 | 3.88 | 11479 |
| 4 | 40K | 61,958 | 46,373 | 15,585 | 60.7 s | 17.96 | 2.88 | 11479 |
| 6 | 60K | 88,356 | 77,374 | 10,982 | 50.0 s | 17.61 | 2.91 | 11479 |
| 8 | 80K | 113,681 | 100,490 | 13,191 | 68.0 s | 18.43 | 3.28 | 11479 |
| 10 | 100K | 137,189 | 125,348 | 11,841 | 67.7 s | 15.72 | 3.24 | 11479 |
| 12 | 120K | 158,761 | 148,140 | 10,621 | 66.0 s | 12.32 | 2.88 | 11479 |
| 14 | 140K | 179,881 | 168,147 | 11,734 | 78.9 s | 11.88 | 3.12 | 11479 |
| 16 | 156K | 200,607 | 192,056 | 8,551 | 61.5 s | 9.79 | 2.78 | 11479 |

Per-turn step accounting (same run, `LLAMA_STEP_TIMING` gives the verify-step counter): the step time grows with context
exactly as section 42 measured, 131.1 ms at 15K to **283.3 ms at 200K** (section 42: 281.9 ms), while accepted tok/step
swings with the text, 2.76-3.94 (4.01 was the section-42 task mean). At turn 16 that is 283.3 ms/step x 2.78 tok/step =
9.79 t/s: the 200K figure of 58.3 is therefore ACCEPTANCE on this reasoning-heavy text, not a step-time regression. The
per-turn columns are in `results/rw/s2_grow199k.json` (`timings_raw.draft_verification_steps`).

No OOM, no context shift, VRAM flat at 11479 MiB for the whole 1,219 s run, and the 17th turn is refused cleanly rather
than truncated: `HTTP 400 ... request (211792 tokens) exceeds the available context size (204800 tokens),
type exceed_context_size_error`. New-turn latency is the price of the new text: 8.5-17K new tokens prefill at
117-215 t/s against 446 t/s at 4K (section 58.2), which is the staged-KV prefill penalty at depth. Decode falls
24.8 -> 9.8 t/s as context grows, the same shape as section 42's 200K rows. Three answers were read (turns 8, 12, 16);
they describe the actual excerpts they were given (CUDA backend layout, weight-staging hooks, KV save/restore), so
the answers are sane.

### 58.4 Scenario 3: exactness through the server (p8k, temperature 0)

Two prompts, 768 tokens each, one spec server and one `-NoSpec` server at the same profile/port, run one after the
other (they do not fit on the card together):

| prompt | identical text | identical reasoning | spec | no-spec |
|---|---|---|---|---|
| 1 | yes | yes (2284/2284 chars) | 48.91 t/s | 13.21 t/s |
| 2 | yes | yes (2508/2508 chars) | 37.56 t/s | 12.85 t/s |

Spec == no-spec holds through the HTTP layer too, including the thinking stream, and speculation is 2.9-3.7x faster on
this prompt pair. (Both hit max_tokens, so the comparison is 768-token sequences, not truncated answers.)

### 58.5 Scenario 4: sampling (temperature 0.7, top_p 0.9)

| profile | requests | decode mean (min) | acc/step | answers produced | errors | peak VRAM |
|---|---:|---|---:|---:|---:|---:|
| p8k (-c 16384) | 5 | 25.21 (20.35) | 2.44 | 3/5 | 0 | 10819 |
| plong (-c 204800) | 5 | 23.12 (17.44) | 2.46 | 3/5 | 0 | 11479 |
| s64k (IQ3_S, -c 65792) | 3 | 26.13 (20.10) | 3.17 | 2/3 | 0 | 11663 |

No crash, acceptance drops from the greedy 3.5-6.2 tok/step to ~2.4-3.2 as expected for t=0.7, and the outputs are
coherent. Two real-world notes: (a) the build streams the thinking part as `reasoning_content` and the answer as
`content`, so a client that only reads `content` sees empty output; (b) with reasoning-preserve on (the default) and
max_tokens 512, all 8 scenario-1 prompts and 2 of 5 sampling requests had their whole budget consumed by the thinking
part and produced no answer - a usability point, not a server bug.
**Measured fix (one run, p8k, same 8 prompts):** `--reasoning-budget 0` (env `LLAMA_ARG_THINK_BUDGET=0`) gives 8/8 answers
with zero reasoning characters (was 0/8 answers at max_tokens 512), mean decode 40.98 t/s, mean prefill 454.8 t/s,
acceptance 2.80-7.42 tok/step. For chat, set `--reasoning-budget 0` (short answers) or a positive budget with a larger
max_tokens; a content-only client must not rely on the default.

### 58.6 Scenario 5: stability over the whole session

58 requests across 9 server sessions (profile switches by design): 1 request error, and it is the deliberate
one-past-the-limit probe of 58.3 (HTTP 400). Peak VRAM 11674 MiB (s64k with the new S=797 set). Server logs: 0 errors in
the plong-32K, p8k, p8k-NoSpec and s64k logs; 3 in the plong-200K log, all three the `send_error` lines for the
context-limit refusals. Every remaining warning is the standard no-API-key/CORS notice and the
"tensor overrides to CPU are used with mmap enabled" notice. No restart was needed during any scenario. The only
crash-like events in the whole session were the s64k profile's prefill OOMs in 58.1, which is what forced the re-size.

Files: `tests\serve_prod.ps1`, `tests\realworld_chat.py`, `tests\rw_table.py` (prints these tables from the JSONs),
`results\rw\s1_c32768.json`, `s1_c204800.json`, `s1_s64k.json`, `s2_grow199k.json`, `s3_p8k_spec.json`,
`s3_p8k_nospec.json`, `s3_compare.json`, `s4_p8k.json`, `s4_plong.json`, `s4_s64k.json`,
`s1_c204800_split2048.json` (58.2b), `s1_p8k_reabud0.json` (the reasoning-budget run), `srv_*.log`.

## 59. Stock llama.cpp at its best settings: 1.26x behind at 8K, and 3.7x / 7.6x / 7.7x behind once FFN leaves the GPU

Worker2, 2026-10-21 (seq 110 Task A, amended by seq 111). Baseline for the user: the UNMODIFIED official binary
`llama-b11717-bin-win-cuda-13.4-x64\llama-server.exe` (build 11717, commit 848bbda90), same IQ3_XXS-mtp model, same
`Qwen3.8-27B-DFlash2-Q2_K_S-MIX` drafter, `-fa on -ctk q4_0 -ctv q4_0 -ngl 99 --jinja`, prompts
`results\dq4\code_short.prompt.txt` (8,829 tok), `results/charts/prompts/t64k.txt` (65,532), `t128k.txt` (131,065),
`t200k.txt` (200,369), 128 greedy tokens. The stock build has no step counter, so ms/step = predicted_ms / (draft_n /
n_max) and tok/step = 1 + accepted/steps; every row is token-identical to the fork's hashes on the same prompt.
Runner: `tests\stock_bench.py`; JSONs in `results\stock\*`.

| ctx | config | t/s | ms/step | tok/step | PP t/s | peak MiB | fork best |
|---|---|---:|---:|---:|---:|---:|---:|
| 8K | **a. all on GPU** | **41.93** | 144.2 | 6.10 | 460.1 | 11735 | **54.19** |
| 8K | all on GPU, `-t 12` | 43.16 | 140.1 | 6.10 | 460.9 | 11735 | - |
| 8K | all on GPU + `GGML_CUDA_FORCE_MMQ=1` | 43.16 | 140.1 | 6.10 | 460.1 | 11735 | - |
| 8K | e. MTP d4, all on GPU | 43.20 | 105.0 | 4.57 | 472.3 | 11133 | (39.82 = its own default hash) |
| 8K | b. 8 spread FFN layers on CPU | 14.97 | 404.1 | 6.10 | 320.2 | 11079 | - |
| 64K | a. all on GPU | OOM (298 MiB request) | - | - | - | - | 38.51 |
| 64K | b. 5 / 7 / 10 spread layers | OOM at drafter load (524.8 MiB) | - | - | - | - | - |
| 64K | **b. 12 spread = smallest that fits** | **10.35** | 584.6 | 6.10 | 238.5 | 11745 | **38.51** |
| 64K | b. 16 (N+4) | 8.03 | 753.0 | 6.10 | 215.6 | 11415 | - |
| 64K | b. 20 (N+8) | 7.13 | 848.4 | 6.10 | 195.4 | 11041 | - |
| 64K | b. 12 + `GGML_CUDA_FORCE_MMQ=1` | 11.12 | 544.0 | 6.10 | 250.7 | 11745 | - |
| 64K | stock form `--n-cpu-ffn 12` | OOM on the first prefill | - | - | - | - | - |
| 64K | stock form `--n-cpu-ffn 16` | 10.15 | 595.6 | 6.10 | 240.7 | 11611 | - |
| 64K | e. MTP d4 on the 12-layer config | 4.96 | 914.3 | 4.57 | 250.6 | 11351 | ~37.6 |
| 128K | b. 18 / 24 spread | OOM at drafter load | - | - | - | - | 35.56 |
| 128K | **b. 32 spread = smallest that fits** | **4.70** | 1286.7 | 6.10 | 148.3 | 11363 | **35.56** |
| 200K | b. 40 spread | OOM at drafter load | - | - | - | - | 21.06 |
| 200K | **b. 44 spread = smallest that fits** | **2.74** | 1852.4 | 5.12 | 111.2 | 12106 | **21.06** |

### 59.1 Fork column on the SAME prompts (reviewer seq 114)

The "fork best" column above is section 33's (512 greedy tokens, its own prompts). For a like-for-like column the fork
was re-run on exactly these prompts (code_short / t64k / t128k / t200k), 128 greedy tokens, production config per
context: all-resident MMQ + DFlash d7 at 8K, `LLAMA_KV_GPU_BUDGET_MIB` 1678 / 1300 / 1000 at 64K / 128K / 200K,
`GGML_CUDA_PREFER_MMQ=1` and the section-33 common env, snapshot restore where the placement matches
(`t1-xx-64k-mtp.bin`, `t1-xx-128k-mtp.bin`, `t2-200k-base.bin`; the 8K row is a native prefill).

| ctx | stock b11717 | fork, same prompt | fork/stock | fork ms/step | fork tok/step | fork peak MiB | sha (both) |
|---|---:|---:|---:|---:|---:|---:|---|
| 8K | 43.16 | **54.92** | 1.27x | 110.1 | 6.10 | 11238 | d0fb7092955c105a |
| 64K | 10.35 | **41.30** | 3.99x | 113.9 | 4.74 | 11487 | d0fb7092955c105a |
| 128K | 4.70 | **30.18** | 6.42x | 161.8 | 4.92 | 11489 | d0fb7092955c105a |
| 200K | 2.74 | **18.82** | 6.87x | 281.2 | 5.33 | 11457 | 313112b176fc3339 |

Both columns produce the identical token stream per context, so this is the same text on both sides. Two caveats stated
plainly: the fork rows are restore-based (decoding only) while the stock rows are native prefills, so no PP column is
comparable here - the PP comparison stays in the main 59 table (stock 460.9/238.5/148.3/111.2 t/s); and the 200K fork row
is 18.82 t/s against section 33's 21.06 because that row used 512 tokens and a different prompt (section 42's task mean
14.21 shows how much the text matters). The fork's step time at 200K, 281.2 ms, is the same 282 ms found in 58.3 - the
fork's advantage at depth is entirely that its step stays at ~113-281 ms while the stock step grows to 1852 ms.

Readings:
1. **Where it fits, stock is only 1.26x behind** (8K 43.16 vs 54.19). The whole gap is the MMQ route: stock has no
   `GGML_CUDA_PREFER_MMQ` (fork flag) and `GGML_CUDA_FORCE_MMQ` is worth +0 here (43.16 with and without), so it runs
   MMVQ fused matvecs at 1-8 columns. Everything else (DFlash d7, q4_0 KV, `-fa on`) is already stock.
2. **Once FFN has to leave the GPU the stock binary collapses**: 10.35 t/s at 64K (-73% vs the fork's 38.51), 4.70 at
   128K (-87%), 2.74 at 200K (-87%). The reviewer's back-of-envelope (DDR4 ~19 GB/s ~ our 13.3 GB/s PCIe stream, and
   DFlash's 5-6 tok/step amortises both) does NOT hold, because the two paths are not the same shape: the fork streams
   one 38 MiB ffn_down tensor per layer over PCIe and runs the matmul on the GPU (the DMA hides under ~4.5 ms of layer
   compute, 55.2), while a CPU FFN layer runs the matmul itself on 6 cores, so every verify step pays ~65-90 ms per CPU
   layer instead of ~3 ms of copy. That is why the slope is brutal: -22% for +4 CPU layers at 64K (584.6 -> 753.0
   ms/step), -13% for the next 4 (-> 848.4).
3. **`--n-cpu-ffn N` is not equivalent to spreading N layers**: at 64K, 12 layers of `--n-cpu-ffn` do not even survive
   the first prefill (it offloads the first N layers, which are the smallest FFN layers, ~950 MiB) while 16 layers of it
   (10.15 t/s, 240.7 PP) lands close to 12 spread layers (10.35, 238.5). Spread is the better control variable, as
   section 55 found for streaming, but for the stock binary the stock knob is competitive at equal offloaded bytes.
4. **MTP d4 is not a stock escape hatch**: 43.20 at 8K (a wash with DFlash d7) but 4.96 vs 10.35 at 64K - the extra
   accepted length of DFlash (6.10 vs 4.57 tok/step) is what pays for the CPU layers.
5. Peak VRAM at 200K is 12106 MiB (182 MiB free) - the stock path has no headroom left at all, and the smallest working
   config is forced by the drafter's 524.8 MiB, not by the model.

So: the fair stock baseline at its best settings is 8K 43.16 t/s, 64K 10.35, 128K 4.70, 200K 2.74, against the fork's
54.19 / 38.51 / 35.56 / 21.06 - i.e. the fork's advantage is +26% where the model fits and 3.7x-7.7x where it does not,
and no stock flag in this build (FORCE_MMQ, -t 12, --n-cpu-ffn, MTP d4) changes that ordering.

## 60. Higher-KV-quant tuning: the q8 regression is the vec_quant_kv path, not KV size - and the MMA route makes q8 KV free at 8K

Worker2, 2026-10-21 (seq 110 Task B.1/B.2, fork build, IQ3_XXS-mtp + DFlash d7, 128 greedy tokens, all weights
resident, prompts `results\dq4\code_short.prompt.txt` (8,829 tok) and `results\kv\p32k_from_t64k.txt` (32,639 tok,
a real-text truncation so the model does not stop early). Runner `tests\kvrun.py` -> `tests\w2_dflash.py` (now takes
`--ctk/--ctv`), JSONs in `results\kv\*`. `LLAMA_FA_VEC_QUANT_KV` is a PRESENCE test in `fattn.cu:657`, so the "novq"
variant unsets it rather than setting it to 0.

### 60.1 8K: the regression reproduces, and unsetting one env var removes it

| KV | variant | decode t/s | ms/step | tok/step | PP t/s | peak MiB |
|---|---|---:|---:|---:|---:|---:|
| q4_0/q4_0 | current env | 54.92 | 110.1 | 6.10 | 457.1 | 11238 |
| q8_0/q4_0 | current env | 48.12 | 125.7 | 6.10 | 325.0 | 11564 |
| q8_0/q4_0 | **no VEC_QUANT_KV** | **56.83** | 111.7 | 6.40 | 461.1 | 11592 |
| q8_0/q4_0 | VEC_MAX_NQ=4 | 53.25 | 113.6 | 6.10 | 456.0 | 11592 |
| q8_0/q8_0 | current env | 42.47 | 142.4 | 6.10 | 237.6 | 11492 |
| q8_0/q8_0 | **no VEC_QUANT_KV** | **54.35** | 111.3 | 6.10 | 459.7 | 11720 |
| q8_0/q8_0 | VEC_MAX_NQ=4 | 52.98 | 114.1 | 6.10 | 456.7 | 11718 |
| f16/f16 | current env | 54.77 | 110.4 | 6.10 | 458.5 | 11972 |
| q8_0/* | stock binary b11717 | **OOM** (no streaming, no headroom) | - | - | - | - |

The user's -14% decode / -28% prefill on q8 KV reproduces exactly (48.12 and 325.0 vs 54.92 and 457.1 = -12.4% /
-29.0% for q8_0/q4_0; -22.7% / -48.0% for q8_0/q8_0) and the hypothesis in the task is right: it is the
`LLAMA_FA_VEC_QUANT_KV` route, which only knows q4_0 and therefore drags q8 KV through the generic vector kernel, for
verify AND for 128-token prefill ubatches. Unsetting it (stock FA routing, MMA with an f16 conversion) restores both:
**56.83 t/s and 461.1 PP for q8_0/q4_0, i.e. +18.1% / +41.9% over the current env and +3.5% / +0.9% over q4_0/q4_0**;
q8_0/q8_0 lands at 54.35 and 459.7 (-1.0% vs q4_0/q4_0). `VEC_MAX_NQ=4` only recovers half of it, because the decode
batch (verify, 8 columns) still takes the vector path at that setting. So the rule is not a new kernel but a routing
decision: **q8 KV => do not set LLAMA_FA_VEC_QUANT_KV**; q4/q4 keeps the current env, which is 1% behind MMA but exact
and (55.7) the measured production route.

### 60.2 32K: the same ordering, and q8_0/q8_0 stops fitting

| KV | variant | decode t/s | ms/step | tok/step | PP t/s | peak MiB |
|---|---|---:|---:|---:|---:|---:|
| q4_0/q4_0 | current env | 21.80 | 116.5 | 2.56 | 410.6 | 11418 |
| q8_0/q4_0 | current env | 17.54 | 181.0 | 3.20 | 187.6 | 12024 |
| q8_0/q4_0 | no VEC_QUANT_KV | 23.13 | 124.8 | 2.91 | 408.7 | 12030 |
| q8_0/q8_0 | current env | 13.84 | 235.2 | 3.28 | 101.3 | 11928 |
| q8_0/q8_0 | no VEC_QUANT_KV | **OOM** at the first eval (`cublasCreate_v2`) | - | - | - | - |
| f16/f16 | - | context creation fails (OOM) | - | - | - | - |

Acceptance differs between rows here because q8 K changes the token stream at this length (8K rows all share
`d0fb7092955c105a`; at 32K the shas are a5b37b2b354031b1 / 5f7e2f0abc5f9e54 / 0d9ae96fb1850fd1 / 16f0d4a2bc2d0817), so the
honest comparison is ms/step at equal configuration rather than t/s across KV types: q8_0/q4_0 without VEC_QUANT_KV
is 124.8 ms/step against q4_0/q4_0's 116.5 (+7%), while with the current env it is 181.0 (+55%).

### 60.3 Quality: what the higher quant buys (teacher-forced, top-20 support, 8K, 32 steps)

`tests\dq4_kld.py --ctx 16384 --n 32 --configs f16,q4q4,q8q4,q8q8` (all on the MMA route, i.e. the fast one):

| KV | top-1 agreement vs f16 | KLD mean | KLD max | max abs dprob |
|---|---|---:|---:|---:|
| q8_0/q8_0 | 32/32 | **1.090e-04** | 1.348e-03 | 1.604e-02 |
| q8_0/q4_0 | 32/32 | 4.030e-04 | 5.505e-03 | 2.425e-02 |
| q4_0/q4_0 | 32/32 | 1.017e-03 | 9.984e-03 | 3.157e-02 |

So q8_0/q8_0 is 9.3x closer to f16 than q4_0/q4_0 and q8_0/q4_0 is 2.5x closer, at identical top-1 over 32 steps - the
quality ladder is real and monotone, and (60.1) it is free of speed cost at 8K once the routing is fixed.

### 60.4 Crossover per (quant, context), from the measurements above

| KV | resident ceiling (this model, 12 GiB) | speed consequence |
|---|---|---|
| q4_0/q4_0 | fits at 32K resident (peak 11418); at 64K+ needs staged KV (55.7) | the production route, exact vs itself |
| q8_0/q4_0 | fits at 32K resident (peak 12030, 258 MiB left) | -7% ms/step vs q4/q4 on the MMA route, +55% on the vec route |
| q8_0/q8_0 | fits at 8K (11720), OOM at 32K | needs streaming or KV staging above ~16K |

Rule for a dynamic config: keep `LLAMA_FA_VEC_QUANT_KV` only for q4_0/q4_0 (the exact, measured production path);
for any q8_0 KV unset it and accept the MMA f16 scratch, which at 8K costs nothing (54.35 vs 54.92) and at 32K costs +7%
in ms/step. CORRECTION (60.6, measured): the "8K-16K only" limit stated below was a property of the placements tried, not
of q8 KV - with STAGED KV (and at 128K about 513 MiB of streamed weights on top) q8 KV does run at depth. See 60.6.

### 60.4b SUPERSEDED BY 60.6: q8 KV did not fit at 64K in the two placements tried first

> Correction: everything below is a record of the IQ3_S-streamed and IQ3_XXS-all-resident placements only. 60.6
> measured the missing case (IQ3_XXS + STAGED KV) and q8_0/q4_0 does run at 64K (37.8-38.7 t/s) and at 128K
> (17.6 t/s, with 513 MiB of streamed weights). The "8K-16K only" conclusion below is withdrawn; the rule that
> applies is the corrected one at the end of 60.6.

The 8K story does not carry to depth. Measured attempts at 64K with the production streamed placement (IQ3_S, S=797,
DFlash d7, restore, text 4f3bce79f1efe976):

| attempt | result |
|---|---|
| IQ3_S S=797, budget 300 (55.7's q4_0 value), q8_0/q4_0 | context creation fails: `KV GPU budget is too small for two staging slots and the retained layers` |
| same, budget 400 | same failure - the per-layer KV with q8_0 K is 108.4 MiB (72.28 K + 36.14 V), so 14/16 staged needs 4 x 108.4 = 433.8 MiB of budget, not 300 |
| same, budget 450 (14/16 staged), q8_0/q4_0, no VEC_QUANT_KV | loads, then `CUDA error: out of memory` on the first eval |
| same, budget 450, q8_0/q4_0, VEC_QUANT_KV route | idem |
| IQ3_XXS all-resident config (budget 1678, 41.30 t/s on q4_0/q4_0), q8_0/q4_0 | OOM at drafter load |

Reason: q8_0 K is +36.14 MiB per layer at 64K (x16 layers = +578 MiB of KV bytes) and the MMA route additionally needs
an f16 K scratch that scales with context - at 64K that is roughly 65,536 x 8 heads x 128 dims x 2 B = 134 MiB, against
~17 MiB at 8K. On a 12 GiB card the streamed 64K profile has 614 MiB of headroom at peak (58.1), so the +578 MiB of KV
and the +134 MiB scratch together cannot fit without another ~350 MiB of weight streaming, which costs ~26 ms/step at
64K (55.7 d). That is this subsection's conclusion, and 60.6 shows it was too pessimistic: it described the placements tried
here (IQ3_S streamed, IQ3_XXS all-resident), not q8 KV. With staged KV on IQ3_XXS, q8_0/q4_0 runs at 64K at 37.8-38.7 t/s
and at 128K at 17.6 t/s, because staging takes the extra KV bytes off the card.

### 60.5 dq8 (q8_0 K/V in the split-K decode kernel): estimate, not built

What it would be: a `fattn-decode-q8` twin of `fattn-decode-q4.cu` - q8_0 K is 34 bytes per 32-element block (f16 scale
+ 32 int8), so the fragment builder dequantises one block instead of two (q4_0 is 18 bytes per 32), the same split-K
partition and the same `LLAMA_FA_Q4_SPLIT_TOKENS`-style constant apply, and exactness is checkable the same way (spec ==
no-spec sha plus a KLD against f16). Effort: a copy-and-edit of an existing 500-line kernel plus the launcher/route
plumbing, ~1-2 days including the A/B, with the same failure mode as 55.1 (silently wrong bytes if a builder is off), so
the gate is a KLD run before any timing is believed.
Expected gain: NOT fewer bytes - q8_0 K is 1.44x the bytes of q4_0 K (q8/q8 1.89x), so at 64K-200K, where 55.7 g shows
the step is PCIe-bound (sync 34-35% of step, 1012 MiB of staged KV per step at 64K, 2024 MiB at 128K) a q8 K would add
~44% to the KV DMA: at 128K that is +891 MiB/step = +67 ms on a 262 ms step (-26%) before the kernel gains anything.
So dq8 only pays if the vec-path penalty it removes is bigger than the byte penalty it adds, and the measurements above
say the vec penalty is 12-20% at 8K-32K - i.e. a wash to a loss at depth. Recommendation: do not build dq8; instead
(a) unset VEC_QUANT_KV for q8 KV, and (b) keep q4_0/q4_0 where the KV has to cross PCIe.

Files: `results/kv/b8k_*.json`, `results/kv/c32k_*.json`, `results/kv/kld_kv_8k.json`,
`results/kv/stock8k_*.json` (the OOM evidence), `tests/kvrun.py`, `tests/dq4_kld.py` (now takes `--ctk/--ctv` and
KV-quant configs), `tests\w2_dflash.py` (`--ctk/--ctv/--unset-env`), `tests\stock_bench.py` (`--ctk/--ctv`).

## 60.6 Task C: q8 KV pushed to depth - staged KV makes it work, with a budget that scales with the KV bytes

Worker2, 2026-10-21 (seq 118 Task C). Model IQ3_XXS-mtp + DFlash d7, `LLAMA_FA_VEC_QUANT_KV` unset for every q8 row
(MMA f16 route, 60.1), native prefill for every row (so PP is measurable and no snapshot-quant mismatch is possible),
128 greedy tokens, `-b 512 -ub 128`. JSONs `results/kv/c*_xx_*.json`.

### C.1 64K: q8 is ~11% slower than q4_0, at equal staged count

| ctx | KV | budget MiB | staged | route | t/s | ms/step | tok/step | PP t/s | peak MiB |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 64K | q4_0/q4_0 | 450 | 12/16 | vec | 43.23 | 133.5 | 5.82 | 307.7 | 10971 |
| 64K | q4_0/q4_0 | 450 | 12/16 | mma | 43.52 | 132.6 | 5.82 | 307.7 | 10969 |
| 64K | **q8_0/q4_0** | 700 | 12/16 | mma | **38.65** | 156.5 | 6.10 | 308.8 | 11285 |
| 64K | q4_0/q4_0 | 300 | 14/16 | vec | 42.64 | 135.4 | 5.82 | 301.1 | 10825 |
| 64K | **q8_0/q4_0** | 450 | 14/16 | mma | **37.77** | 160.1 | 6.10 | 302.2 | 11077 |
| 64K | q8_0/q4_0 | 1000 | 9/16 | mma | OOM at eval | - | - | - | - |
| 64K | q8_0/q4_0 | 1300 | 7/16 | mma | OOM at load | - | - | - | - |

The budget values are not the production 300 of 55.7: with q8_0 K the per-layer KV at 64K is 108.4 MiB (72.28 K + 36.14 V)
instead of 72.28, so the 14/16 staging point moves to budget 434 MiB, and below that the launcher fails with `KV GPU
budget is too small for two staging slots and the retained layers`. Above it the staged count falls (1000 -> 9/16,
1300 -> 7/16) and the resident remainder no longer fits next to the drafter, so the sweep has a working window rather
than one value.

**Pure KV byte cost at matched staging (both rows on the MMA route):** 12/16: 43.52 -> 38.65 t/s = -11.2%
(132.6 -> 156.5 ms/step, +23.9 ms). 14/16: 42.64 -> 37.77 = -11.4% (+24.7 ms). The predicted cost of +434 MiB of staged
K per step at 13.3 GB/s is +32 ms, so the measured +24 ms means part of the extra copy still hides under compute - the
overlap 55.2 measured. Prefill is unaffected (301-309 t/s everywhere) because the staged-KV DMA per ubatch keeps its
shape and q8 only doubles the K half.

### C.1 128K: q8 alone does not fit, the layered scheme does

| ctx | KV | budget MiB | extra | staged | route | result | ms/step | PP t/s | peak MiB |
|---|---|---:|---|---|---|---|---:|---:|---:|
| 128K | q8_0/q4_0 | 1000 | - | 14/16 | mma | fails at eval (`cublasCreate_v2`, 1213 MiB free at load) | - | 244 | - |
| 128K | q8_0/q4_0 | 900 | - | 14/16 | mma | fails at eval, also with `--ctx-checkpoints 0` | - | 244 | - |
| 128K | q4_0/q4_0 | 600 | S=513 | 14/16 | vec | 24.52 t/s | 246.7 | 240.8 | 10701 |
| 128K | q4_0/q4_0 | 600 | S=513 | 14/16 | mma | 25.73 t/s | 246.8 | 240.9 | 10713 |
| 128K | **q8_0/q4_0** | 900 | **S=513** | 14/16 | mma | **17.59 t/s** | 328.2 | 241.6 | 11313 |

The scheme that works at 128K is: KV staged (budget 900 -> 14/16, slots 2 x 208.81 MiB) **plus** 513 MiB of streamed
ffn_down (16 spread layers, `LLAMA_W_STAGE_*`, 55.2) to pay for the extra KV bytes and the f16 scratch. Matched
comparison (same 14/16 staging, same S=513, both MMA): q4_0/q4_0 25.73 -> q8_0/q4_0 17.59 = **-31.6%**
(246.8 -> 328.2 ms/step, +81.4 ms). The predicted +899 MiB of staged K per step is +68 ms, so ~13 ms of the delta is the
f16 conversion overhead. Prefill is again unaffected (241 t/s both).

### C.2 The f16 K/V scratch: full-context, one allocation per op from the CUDA pool

Code: `ggml/src/ggml-cuda/fattn-common.cuh:980-984`

```cpp
ggml_cuda_pool_alloc<half> K_f16_alloc(pool);
ggml_cuda_pool_alloc<half> V_f16_alloc(pool);
half * K_f16 = need_f16_K && K->type != GGML_TYPE_F16 ? K_f16_alloc.alloc(ggml_nelements(K)) : nullptr;
half * V_f16 = ... V_is_K_view ? K_f16 : V_f16_alloc.alloc(ggml_nelements(V));
```
and the conversion at ~999-1020 (`to_fp16(K_data, K_f16, ggml_nelements(K), main_stream)`, or the per-dimension variant
for non-contiguous K). So:
- it is sized for the FULL KV tensor - the whole context x the KV-head width (`ggml_nelements(K)`), not per staged layer
  and not per ubatch: 131,584 x 1,024 x 2 B = 269 MiB for K at 128K, plus 269 MiB for V whenever V is not f16 (true for
  every q4_0 V here); 134 + 134 MiB at 64K; 17 + 17 MiB at 8K;
- it is allocated ONCE per attention op call (i.e. per layer, per graph compute) from `ctx.pool()`, and the pool keeps it,
  so after a 128K prefill the memory is still held when the first decode runs. That is exactly why the 128K q8 rows died
  with 1213 MiB "free" at load: 538 MiB of scratch + the prefill's own pool high-water mark + a 160.9 MiB context
  checkpoint (`created context checkpoint 2 of 32 ... size = 160.923 MiB`) exceed it;
- the conversion reads and writes the whole tensor on every call, so it also costs time per layer: with 14 staged layers
  that is 14 x (269 + 269) MiB of conversion traffic per decode step at 128K - the ~13 ms visible in C.1 is that cost.
Candidate scheme (estimate only, not built): convert only the window the kernel reads. The MMA/vec kernels walk the KV in
tile bands, so a scratch of one band (e.g. 4,096 positions: 4,096 x 1,024 x 2 B = 8 MiB) plus a per-band conversion inside
the loop would cut both the scratch and the conversion traffic by the band factor, and would make q8 K affordable at
depth. Same risk class as 55.1 (silent wrong bytes if the band bookkeeping is off), so the gate is a KLD run against f16
before any timing is believed.

### C.3 32K q8_0/q8_0 with staging: works, and is free at equal staging

| ctx | KV | budget | staged | t/s | ms/step | tok/step | PP t/s | peak MiB |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 32K | q4_0/q4_0 | 300 | 10/16 | 19.66 | 166.5 | 3.36 | 356.9 | 10811 |
| 32K | q8_0/q8_0 | 600 | 10/16 | 19.80 | 165.3 | 3.36 | 342.9 | 11507 |

Equal staging (10/16) and equal tok/step: q8_0/q8_0 is +0.7% t/s and -0.7% ms/step, i.e. the +201 MiB of staged KV per
step at 32K is inside the noise of this decode - the 60.1 8K conclusion, now with staging. (37 tokens because this
prompt hits EOS early; ms/step and tok/step remain well defined.)

### C.4 serve_prod.ps1

`-Ctk`/`-Ctv` exist now (default q4_0/q4_0) and the launcher prints the chosen route. If either type is `q8*` it removes
`LLAMA_FA_VEC_QUANT_KV` and prints `route: mma-f16 (LLAMA_FA_VEC_QUANT_KV not set)`; for q4_0/q4_0 it prints
`route: vec-q4 (LLAMA_FA_VEC_QUANT_KV set)` and the command line is byte-identical to before (checked with -DryRun on all
three profiles).

**Corrected rule for the dynamic config** (replaces the "8K-16K only" line in 60.4/60.4b): q8 KV is a depth option, not
a short-context one, provided that (1) `LLAMA_FA_VEC_QUANT_KV` is unset, (2) the KV is STAGED with a budget scaled to the
KV bytes (>= 434 MiB at 64K, >= 835 MiB at 128K to reach 14/16), and (3) at 128K about 513 MiB of weight streaming pays
for the full-context f16 scratch. Cost at matched placement: -11% at 64K, -32% at 128K, +-0% at 32K for q8_0/q8_0.
Whether that is worth paying is a quality call, and 60.3 measured the ladder: q8_0/q8_0 9.3x and q8_0/q4_0 2.5x closer to
f16 than q4_0/q4_0.

Files: `results/kv/c64k_xx_*.json`, `results/kv/c128k_xx_*.json`, `results/kv/c32k_xx_*.json`, `tests/serve_prod.ps1`
(`-Ctk`/`-Ctv` and the automatic route), plus 60.1's runner.
