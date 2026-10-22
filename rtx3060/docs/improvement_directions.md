# Improvement directions (parked ideas)

Ideas worth revisiting later. Each has the reasoning, the numbers behind it, and what it needs.
Model: Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf on an RTX 3060 12 GiB (~340 GB/s).

## 1. Swap IQ3_S tensors for IQ4_XS (faster speculative verify, better quality)

Status: parked, 2026-10-20. Not measured end to end.

**Why.** Speculative verify reads every weight once per step, so the floor is weight bytes / bandwidth
(~30.6 ms for the whole model). Measured MMQ at 8 columns, as a multiple of each tensor's own read floor
(comm\files\seq50_step1_feasibility.txt, section 4):

| type | bits/weight | share of weight bytes | MMQ@8 / floor | relative verify cost per weight (bits x factor) |
|---|---:|---:|---:|---:|
| IQ4_XS | 4.25 | 13.9% (incl. output head) | 1.08-1.37 | ~5.1 |
| IQ2_S | 2.5 | 19.2% | 1.71-2.86 | ~5.8 |
| IQ3_XXS | 3.06 | 19.6% | ~2.04-2.16 | ~6.2 |
| IQ3_S | 3.44 | 21.7% | 2.07-2.61 | ~7.1 |

IQ4_XS is cheap to unpack, so an IQ4_XS weight costs less to verify than an IQ3_S weight despite having more
bytes. At n=1 (MMVQ) IQ4_XS is also nearer its floor (0.85-1.0x vs 1.2-1.4x for IQ3).

**Why not the whole model.** A 27B model at IQ4_XS is ~13.4 GiB of weights, which doesn't fit 12 GiB.

**Partial version.** Requantize only the IQ3_S tensors (~2.1 GiB now) to IQ4_XS:
- est. +0.5 GiB VRAM;
- est. ~4 ms faster per 8K verify step (~54 -> ~56 t/s with DFlash d7);
- slightly better quality.
- At 200K the extra 0.5 GiB costs ~2-3 more staged KV layers (~17.7 ms each per pass). So this is a short/mid-context
  model file, not the 200K one.

**Needs.** The original BF16/FP16 weights: requantizing from the current IQ3_S weights only re-encodes their
existing error, so the VRAM goes up with no quality gain. Also an imatrix if the GSQ-RCO recipe uses one. Then measure
MMQ@8 per tensor, e2e 8K DFlash d7 t/s, and a KLD check against the current model.

## 2. Small-batch matmul kernel (8-16 columns) - investigated, not worth it

Status: closed 2026-10-20 (comm seq 50/51, report.md section 34). Seven kernel designs land at 3.3-6.6x the read floor,
against MMQ@8 at 1.7-2.9x. The practical ceiling for the IQ grid types is ~1.6-1.8x, so a perfect kernel would win
~8-9 ms/step at 8K (+9%). Revisit only if a different latency-hiding schedule is found.

## 3. Streamed higher-precision "prefiller" weights (for low-bit models / 8 GB cards)

Status: parked, 2026-10-20. Not worth it for the IQ3_XXS model on 12 GB (the accuracy left to recover above IQ3_XXS is small).
Revisit when targeting 8 GB cards with IQ1/IQ2-class weights, where it should buy the most accuracy per GB.

**Idea** (ISTA-DASLab/Qwen3.8-27B-NVFP4-prefiller): prefill with a separate higher-precision copy of the transformer
linear weights, streamed one layer at a time from pinned host RAM or SSD through a 2-slot ring placed in the output
head's VRAM (the head is unused until the last prompt token), so it needs no extra VRAM. Decode keeps the low-bit GGUF.
Claimed: ~50% of the accuracy lost to aggressive quantization recovered. Their prefillers are trained to pair with specific
Unsloth IQ1_S..IQ2_S decoders; they use -ub 8192 so the stream hides under compute.

**On Ampere:** NVFP4 needs Blackwell FP4 tensor cores (sm_120/121), but the idea transfers with IQ4_XS/Q4_K (MMQ) or
Q8_0 (int8 mma). Sizes for 27B: IQ4_XS/Q4_K ~14 GB (fits pinned in 31 GiB host RAM, ~1.0 s per pass at 13.2 GB/s),
Q8_0 ~29 GB (SSD only: ~2.2 s per pass at PCIe speed, ~8 s at NVMe speed). Hidden under a 2048+-token ubatch
(8K: ~3.6 s/ubatch; fresh 200K: ~7.5 s/ubatch).
**Not a PP speed lever:** prefill MMQ is not dequant-bound (report.md 20: 30-41 TFLOPS from IQ2_S to F16, 1.2x for 6x bytes),
so ~+5-10% at most. The PP speed lever is int8 tensor cores (report.md 21: cuBLAS int8 ~2.4x MMQ), independent of streaming.
**Constraints:** only for large prompts (small new-turn ubatches at ~199K would double in time); PCIe is shared with staged
KV at long context (~16 GB/pass total at 200K, still hidden under 7.5 s); a 2x IQ4_XS layer ring (~420 MB) fits the 644 MiB
head; spec == no-spec exactness is unaffected.
**Cheap accuracy test first (no code):** prefill with a higher-precision GGUF of the same base (partial CPU offload is fine),
slot-save, restore into the low-bit model, and compare decode KLD vs a reference and vs the low-bit model's own prefill.

## 4. Small decode levers to combine later (~10% together)

Status: parked, 2026-10-20 (user: keep them in reserve; try them when bigger ideas run out).
- **DFlash tree / forest verification at LARGE context** (the intended target is 200K, where the verify hides under the PCIe
  stream: width 8 -> 16 costs only +7.4 ms/step, comm seq 53). Measured gate (comm seq 56, report.md 39): the target token is
  DFlash's runner-up in 54% of draft failures; a blind "branch at the least-confident position" policy rescues ~31-36%, i.e.
  +0.35-0.45 tok/step (+6-7% tokens); top-3 adds nothing over top-2. At 200K only one extra path fits (one GDN state copy =
  294 MiB, ~835 MiB free), est. +3-7%. Needs multi-seq batching, per-path sampler clones and n_seq_max > 1 (which loses
  LLAMA_RS_REPLAY), or a replay-style per-branch GDN state (~60 MB per 8 tokens) to avoid the 294 MiB copies.
- **Host-side overhead in the speculative loop** (comm seq 53): sampling+accept+bookkeeping 3.3 ms, verify graph submit 1.8 ms,
  inter-step 3.5 ms of a ~101 ms 8K step. Recoverable ~3-5 ms (+3-5%).

### 2a. Repacked "mma-native" weight layout (v8) - tested, abandon (2026-10-21)
Idea (worker3, comm\files\seq74_unpack_explainer.md): repack IQ3 weights at load into a per-quad layout so that each lane
does 2 aligned loads instead of 11 scattered byte loads. Bit-exact (0 diffs), 1.16-1.31x bytes. SASS: 124 instructions per
32-k step (the estimate was ~15), at the ~108 issue budget. Measured at ncols=8, x of read floor: IQ3_S 2.29 / 3.23,
IQ3_XXS 2.71 / 3.81 (for comparison: v1 3.94 / 4.64; MMQ@8 2.07). The pre-set rule (>1.60 = abandon) applies. The coalesced
loads help v1 but the kernel becomes issue-bound and still loses to MMQ. Data: comm\files\seq74_v8_run.txt.

## 5. Weight streaming beyond IQ3_S at 8K (from report.md 55, 2026-10-21)
Status: idea, not measured. Weight streaming (LLAMA_W_STAGE_*) pays off where VRAM forces something slower than a PCIe
copy (CPU layers), and only while PCIe has idle time inside the step. Rule measured at 64K: streamed weight MiB and staged
KV MiB cost about the same per step, so once PCIe is the bottleneck, swapping one for the other is a wash.
- IQ3_XXS 8K-96K: fits resident, so streaming has nothing to replace. Freed VRAM only helps if something useful needs it
  (DFlash forest paths: ~294 MiB per GDN state copy; a bigger drafter, which didn't raise acceptance in report 40).
- Faster PCIe (Gen4 ~26 GB/s, Gen5 ~50 GB/s on other machines): every staged byte costs half or a quarter as much. So
  long-context KV staging speeds up (the 200K step is ~76% PCIe wait), and so do weight streaming at 64K-128K for IQ3_S
  and bigger quants. Example (estimate): a whole IQ4_XS 27B (~13.4 GiB) on a 12 GB card needs ~2-2.5 GiB streamed per
  pass, ~180 ms on Gen3 (too slow), ~90 ms on Gen4 (borderline against a ~100 ms 8K verify), ~45 ms on Gen5 (hides).
- The streamed set should be chosen from the measured PCIe rate at startup, together with the KV budget (it belongs in
  the dynamic config work).

### 2b. v9 paired-layout IQ3 route - shelved 2026-10-21, one open lead
Bench (isolated): v9 is 16.6-19.6% (n=8) / 22-24.5% (n=1) faster than production MMQ on IQ3_S/XXS. In-model with graphs
on it is +3.0% (n=1 79.2 -> 81.7 ms/token; d7 106.1 -> 107.9 ms/step), and per-class graphs-on deltas show no faster class
(best single class attn_q -0.14 ms/token, within noise), so per-shape routing cannot reach the 2% gate. Open lead: the
~20-25 point isolated-vs-model gap, spread as a ~10-18 us/call penalty over every class, explained neither by shape
under-fill nor by the split rule (in-tree grid matches the bench; split 4 vs bench-preferred 2 at out 5120/6144 costs
only 4-8%). A revisit needs (a) nsys --cuda-graph-trace=node on an elevated/admin session (non-admin nsys cannot launch
targets here) for per-kernel graphs-on sums, and (b) a per-shape router that also picks the split (the shape-keyed
`GGML_CUDA_MMSB_ROUTE` skeleton is in the w3 source, unbuilt). Ship state: default off.

## 6. Keep the GLU fusion for streamed weight tensors (IQ3_S long context, from report 55.7, 2026-10-21)
Status: idea, not implemented. ggml_cuda_try_fuse() refuses any fusion window containing a w_stage streamed tensor, so a
larger streamed set S changes the kernel mix. Ceiling measured with GGML_CUDA_DISABLE_FUSION=1 (report 55.7 h): all fusions together are worth +3.4% at 8K (S=805),+0.6% at 64K and ~0 at 128K, so the gain is under 1% at long context (PCIe-bound). Only the 8K case might be worth it. Fix: emit the slot refill/ready hook before the fused launch (the slot address is valid there),
and keep the pair-aliasing guard for a GLU that reads two streamed tensors. Opt-in env var. Gate: spec == no-spec sha, plus
the resident A/B at 8K/64K at equal fusion state.

## 7. Band-wise f16 K/V conversion for higher-KV-quant decode (from report 60.2/60.6, 2026-10-22)
Status: parked, not built. This is the one change that would make q8_0 KV (or any non-f16 K/V) cheap at depth.

**Why.** With `LLAMA_FA_VEC_QUANT_KV` unset, a q8_0 K is served by the MMA f16 route, which needs an f16 copy of K (and
of V when V is not f16). `ggml/src/ggml-cuda/fattn-common.cuh:980-984` allocates that copy from the CUDA pool with
`alloc(ggml_nelements(K))` / `alloc(ggml_nelements(V))` - the WHOLE KV tensor - and ~999-1020 converts the whole tensor on
every call (`to_fp16(K_data, K_f16, ggml_nelements(K), main_stream)`). So the scratch is full-context, allocated once per
attention op (per layer, per graph compute), and retained by the pool: at 128K it is 269 MiB for K + 269 MiB for a q4_0 V
= **538 MiB**, plus 14 layers' worth of conversion traffic per decode step. Together with the pool high-water mark of the
prefill and a 160.9 MiB context checkpoint, that is exactly what made every un-layered q8 attempt at 128K die at
`cublasCreate_v2` even with 1213 MiB "free" at load (report 60.6 C.1/C.2). At 64K it is 134 + 134 MiB, at 8K 17 + 17 MiB.

**The change.** Convert only the KV window the kernel is reading, inside the tile loop, instead of the whole tensor once:
a scratch of one band (4,096 positions x 1,024 x 2 B = **8 MiB**) with a per-band conversion, which cuts the scratch by
~67x at 128K and the conversion traffic by the same factor. The kernels already walk the KV in `ncols`/`KV_max` bands, so
the band boundary exists; the work is the loop bookkeeping and the stream-ordering of the conversion.

**Expected effect.** q8_0/q4_0 at 128K today needs staged KV *plus* 513 MiB of streamed ffn_down and then runs at
17.59 t/s against 25.73 t/s for q4_0/q4_0 at the same placement (-31.6%, report 60.6 C.1). Removing the full-context
scratch would remove the need for that weight streaming (which is where most of the loss comes from: at 128K the step is
PCIe-bound, and the streamed set adds ~315 MiB/step), so the target is q8 KV at 128K within a few percent of q4_0/q4_0
instead of -32%, with the quality ladder of 60.3 (q8_0/q8_0 9.3x closer to f16 than q4_0/q4_0, q8_0/q4_0 2.5x).

**Needs / gate.** Same risk class as report 55.1: a band bookkeeping error computes on stale or wrong bytes silently, so
the gate is a KLD run against an f16-KV reference (tests/dq4_kld.py, which now takes --ctk/--ctv) plus spec == no-spec sha,
before any timing is believed; then 8K/32K/64K/128K rows at matched staging (the table shape in report 60.6 C.1). Effort
estimate: a focused edit in fattn-common.cuh plus whatever the MMA tile loop needs, ~1-2 days including the A/B.

**Scope note (2026-10-22, user):** q4_0/q4_0 is the production KV and q5/further q8 work is out of scope for this
project, so this entry is for a later revisit, not for now.
