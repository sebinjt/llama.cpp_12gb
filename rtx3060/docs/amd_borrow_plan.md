# Borrowing from the AMD adaptive-KV-streaming fork

Checklist of everything worth taking from `amd_ref/` (achyudh/llama.cpp, branch
`feature/adaptive-kv-stream-rocm`) into our CUDA fork. One line per item so progress is
trackable. Results go in the log at the bottom and in `report.md`.

Reference fork: `amd_ref/llama.cpp`, tip `9e24be0e5`. Our fork: `llama_cpp_src/llama.cpp`,
branch `staged-host-kv`.

## Calibration: what their numbers do and do not mean

| | theirs (RX 9070 XT) | ours (RTX 3060) |
|---|---|---|
| VRAM | 16 GB | 12 GB |
| VRAM bandwidth | ~640 GB/s | ~360 GB/s |
| Host link | PCIe 5.0 x16, ~50 GB/s | PCIe 3.0 x16, 13.3 GB/s |
| Weights | IQ4_XS 12.63 GiB | IQ3_XXS 9.23 GiB |
| KV types | Q8_0 K / Q4_0 V | q4_0 / q4_0 |
| Decode working set | ~2,100 MiB @ 262k | 2,025 MiB @ 200K |
| Attention head | D=256, GQA 6 | D=256, GQA 6 (24q / 4kv), verified from GGUF |

Their streaming design is the same shape as ours (bounded working set, async copy engine,
log-sum-exp merge for prefill). Their absolute t/s come from a 3.8x faster host link:
2,100 MiB at 50 GB/s = 44 ms, and they measure 27 t/s at 131k. **They are link-bound too.**
So chase their ratios, never their absolute numbers.

Their decode (reference, sqlite3.c workload, 128 tokens, greedy):

| Context | Stock | Stock+MTP | Fork | Fork+MTP |
|---|---|---|---|---|
| 1,024 | 33.87 | 65.83 | 36.53 | 71.83 |
| 4,096 | 33.46 | 66.00 | 36.22 | 86.51 |
| 16,384 | 30.27 | 59.58 | 35.54 | 81.11 |
| 65,536 | 21.33 | 37.87 | 32.13 | 72.89 |
| 131,072 | OOM | OOM | 26.96 | 78.60 |
| 262,144 | OOM | OOM | 9.50 | 31.08 |

**MTP ratio grows with context** (1.94x at 1k, 3.27x at 262k): a verify pass amortises one
KV sweep over many accepted tokens, which is exactly our 84.4 ms attention problem. Our MTP
loss at 200K was measured with mean accepted length 1.80, so nothing was amortised.

## Tier 0 - no code, config only

- [x] T0.1 Ran it. **Rejected - the 200K win was a measurement artifact.**
      Their config looked spectacular on our 200K snapshot (8.43 t/s against a 5.66
      baseline, +49%, 100% acceptance). But the continuation that snapshot produces is a
      run of newline characters, which is the best possible case for n-gram matching.
      Repeating the measurement on a real code prompt at 32K inverted it: ngram-mod alone
      is **0.87x** (a 13% loss) at 10.3% acceptance.
- [x] T0.2 Swept `--spec-ngram-mod-n-match` 4/6/8, `-n-min` 2/4/12, `-n-max` 32/64/128 at
      L=8/9/10. Everything plateaus at ~8.1-8.4 t/s on the degenerate prompt, i.e. the
      sweep was fitting noise. No config is worth keeping.
- [x] T0.3 Acceptance recorded - and it is the whole story (see 15.2 in `report.md`):
      | config | decode | vs baseline | acceptance |
      |---|---:|---:|---:|
      | real code 32K, no spec | 18.68 t/s | 1.00x | - |
      | real code 32K, MTP n_max=2 | 23.78 t/s | **1.27x** | 72.9% |
      | real code 32K, MTP n_max=3 | 24.28 t/s | **1.30x** | 62.0% |
      | real code 32K, MTP + ngram-mod | 19.38 t/s | 1.04x | 36.5% |
      | real code 32K, ngram-mod alone | 16.25 t/s | 0.87x | 10.3% |
      MTP is genuinely good on code (1.27-1.30x). Stacking ngram-mod **destroys** it,
      because wrong ngram drafts shorten the effective verify batch.
- [x] T0.4 New: long ngram drafts crash at 200K. `n_max=128` at L=9 and `n_max=64` at L=8
      both die with `CUDA error: out of memory` during decode: a verify batch of more than
      two tokens leaves our vec kernel, and the fallback allocates the 838 MiB f16 scratch.
      Direct evidence for Tier 2 and a hard cap on draft length today.

## Tier 1 - small ports

- [x] T1.1 MTP verify catch-up merge (`c288a0fac`). **Applied** - the patch applies clean to
      our tree (`git apply`, no fuzz), our fork has `llama_process` and the same
      `common_speculative_impl_draft_mtp` members. One manual edit: upstream adds a slot to
      `cparams.n_tokens_decode`, a field that comes from their KV-stream phase switch which
      we do not have, so that line is a no-op here (see `speculative.cpp:2756`).
      Measured: 200K MTP `n_max=2` 4.69 -> 4.80 t/s (+2.3%), 32K code unchanged within noise,
      output byte-identical. It is active and does what it says - the pass it removes is just
      the cheap one (one MTP pass over the verify rows), so the win is small. Kept anyway:
      it also stops rejected drafts from entering the draft KV, which is a quality fix.
      The expensive pass is the *draft* pass, and no merge can remove that.
- [ ] T1.2 Verify-batch graph arenas + phase switch (`e54b93bc0`, `ec12b5fc6`): **deferred.**
      Both build on their phase-switched arena infrastructure, which our fork does not have;
      porting them means porting the KV-stream scheduler first. Low value while decode is
      DMA-bound. Revisit only after T2.
- [x] T1.3 Probabilistic rejection sampling - already present in our fork, no port needed.
- [ ] T1.4 Residency budget for the draft model (`-ctkd/-ctvd` and draft page budget) so a
      verify batch does not force extra resident KV.

## Tier 2 - attention kernel (the structural fix)

- [ ] T2.1 Verify-batch decode attention without the full-context f16 scratch.
      Their `ggml/src/ggml-cuda/fattn-paged-decode*.{cu,cuh}`: 1-8 query tokens (later 96),
      split-K flash decoding, in-register dequant, rows = (query token, GQA head) packed
      into WMMA tiles. Current vector dispatch already supports more than two queries
      with `LLAMA_FA_VEC_QUANT_KV=1`; it handles them in two-query groups, repeating KV
      sweeps. The missing optimization is shared work, not enabling n_q=3/4.
      **Not portable as-is**: kernel is hardcoded `PD_D=256, PD_GQA=6`, K must be Q8_0 and
      V Q4_0, `PD_PAGE=256`. Our D=256/GQA=6 geometry matches. CUDA/Ampere support and
      Q4_0 key dequantization are still required; their implementation is HIP/RDNA4-only.
- [ ] T2.2 Same kernel at `n_q == 1`, to replace the vec kernel's 42 GB/s with something
      closer to bandwidth. Would cut the 84.4 ms attention chain directly.
- [ ] T2.3 Check their split-K/partials reduction against ours for numerical parity.

## Tier 3 - phase-specific residency

- [x] T3.0 Offline layer-group prefill producer: layers 0-31 then 32-63, resident active
      KV and MMA, standard target sequence snapshot for the staged server. Measured
      265.17 tok/s at 200K; 12m57s including save/shutdown. Restore is 2.4 s with only
      one prompt token processed. Optional target hidden-state companion retained for
      later one-layer MTP initialization. See report.md section 17 and dev/group-prefill.md.
- [ ] T3.1 Two budgets instead of one: `--kv-resident-mib 450` for prefill,
      `--kv-resident-mib-decode 2100` for decode. Ours is a fixed 2-slot scheme serving both
      phases. Cheap to emulate; prefill is at 219 t/s against their 1,400.

## Tier 4 - kernel fusions, to attack the 47.4 ms weight/GDN chain

- [ ] T4.1 Fused residual add + RMSNorm (`ggml/src/ggml-cuda/norm-rows.cu`).
- [ ] T4.2 Fused SwiGLU + q8_1 activation copy for mmvq (`73accdb69`).
- [ ] T4.3 Sibling activation reuse across gate/up and Q/K/V (`13a7201c0`, `f692e0422`).
- [ ] T4.4 GDN gate+beta fusion and GDN MMA path (`gdn-conv.cu`, `gated-delta-net-mma.cu`,
      `bc1727d40`) - our 48 GDN layers are most of the 47.4 ms.
- [ ] T4.5 16-byte vectorised Q4_K/Q5_K mmvq loads (`bd063a49a`) - check CUDA equivalent.

## Tier 5 - model level (needs a new GGUF, quality decision)

- [ ] T5.1 ASCII-condensed vocabulary. Their model is a prebuilt HF GGUF
      (`bsaleh03/Qwen3.8-27B-ASCII-Condensed`, IQ4_XS 12.63 GiB) which does not fit our
      12 GB card, and is not the GSQ-RCO fine-tune we run. Condensing our own model would
      free part of the 1,032 MiB of vocabulary tensors (`output.weight` 644.1 MiB,
      `token_embd.weight` 388.4 MiB, vocab 248,320) but changes what the model can emit.
      Blocked on an explicit quality decision.

## Deliberately skipped

- ROCm/HIP plumbing, `nix/`, `GPU_MAX_HW_QUEUES`, HIP graph modes - not our platform.
- Q8_0 K cache - they can afford +44% KV bytes at 50 GB/s; we cannot at 13.3 GB/s.
- `src/llama-moe-cache.*`, `src/llama-memory-hybrid-idx.*` - MoE machinery, our model is dense.
- mmvq 4-row multi-column RDNA4 wave tuning (`4c45cd35c`, `10719e69a`) - CUDA has equivalents.
- Their page size 256 and all `PD_*` constants as literal values.

## Log

| Date | Item | Result |
|---|---|---|
| - | baseline before borrowing | 200K, L=8, output on CPU: 5.77 t/s (170.6 ms/tok) |
| T0.1 | their spec config, 200K snapshot | 8.43 t/s (+49%), 100% acceptance - **artifact**: continuation is a newline run |
| T0.1 | ngram-mod, real code, 32K | 16.25 t/s vs 18.68 no-spec = **0.87x**, acceptance 10.3% |
| T0.3 | MTP n_max=2, real code, 32K | **23.78 t/s vs 18.68 = 1.27x**, acceptance 72.9% |
| T0.3 | MTP n_max=3, real code, 32K | **24.28 t/s = 1.30x**, acceptance 62.0% |
| T0.3 | MTP + ngram, real code, 32K | 19.38 t/s = 1.04x, acceptance 36.5% (stacking hurts) |
| T0.4 | long drafts at 200K | OOM: verify batch > 2 leaves the vec kernel and allocates the 838 MiB f16 scratch |
| T1.1 | catch-up merge applied | 200K MTP n_max=2: 4.69 -> 4.80 t/s (+2.3%); 32K unchanged; output identical |
| T1.3 | MTP `n_max=1` vs 2 vs 3 | **n_max=1 wins everywhere**: 32K code 24.44 / 23.83 / 23.74 t/s; 200K 5.74 / 4.80 / 4.69 t/s |
| - | `serve.ps1` default | `-Mtp` 2 -> **1** (was suboptimal at both ends of the context range) |
| - | all runs | output byte-identical with and without speculation (sha256 9924fc0bca5e5c21 / e48bf1f5b3f1d43e) |
