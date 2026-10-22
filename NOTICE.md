# NOTICE

This repository is a fork of **llama.cpp** by the ggml authors.

- Upstream project: https://github.com/ggml-org/llama.cpp
- Upstream base commit: `547208dca250ceb3e4c8615598d043f2b602a527`
  (upstream `master`, merge-base of the fork branch `staged-host-kv` and
  `origin/master`):
  `git -C llama_cpp_src/llama.cpp merge-base HEAD origin/master`
- Fork commit on top of that base: `094c9ecce`
  ("staged host KV: pinned host canonical KV with per-layer device staging slots")
- Published tree: the current **working tree**, including the uncommitted changes that
  were present when this repository was packaged. The upstream `.git` history is not
  part of this repository.

## License

llama.cpp and this fork are distributed under the MIT License. See `LICENSE`
(Copyright (c) 2023-2026 The ggml authors). No additional or different license terms
are applied to the fork's changes.

Third-party components keep their own licenses; see the `licenses/` directory and the
individual vendored projects under `vendor/`.

## Principal changes in this fork

The complete change set is the source diff against the base commit. The main areas are:

1. **Staged host KV cache** - pinned host canonical KV, per-layer device staging slots,
   overlapped PCIe copies, configurable staged-layer count and GPU KV budget, snapshot
   restore support. New `ggml/src/ggml-cuda/staged-kv.cuh`; changes in
   `ggml/src/ggml-cuda/ggml-cuda.cu`, `src/llama-kv-cache.{cpp,h}`,
   `src/llama-context.cpp`, `src/llama-cparams.h`,
   `src/llama-memory-recurrent.{cpp,h}`.
2. **Decode attention** - split-K q4_0 decode attention (`fattn-decode-q4.{cu,cuh}`,
   new), vector and MMA flash-attention routing changes, GQA-packed q4_0 path,
   avoidance of the full-context f16 KV scratch. Changes in
   `ggml/src/ggml-cuda/fattn*.{cu,cuh}` and `fattn-common.cuh`.
3. **Exact speculative decoding** - DFlash lattice/tree drafts, replay rollback,
   MTP window, batch-invariant MMVQ reduction, draft-verification server counters.
   Changes in `common/speculative.cpp`, `common/sampling.cpp`,
   `tools/server/server-common.cpp`, `tools/server/server-context.cpp`,
   `src/llama-*`.
4. **Weight streaming** - opt-in double-buffered quantized-weight staging with
   prefetch, including GDN participation. Changes in
   `ggml/src/ggml-cuda/ggml-cuda.cu`, `gated_delta_net.cu`, `models/qwen35.cpp`,
   `models/delta-net-base.cpp`, `mmvq.cu`.
5. **Prefill layer ranges** - `src/llama-prefill-range.h` (new) and corresponding
   context/cache plumbing for offline group prefill and snapshot production.
6. **Diagnostics** - opt-in timing, route and KV-dump hooks (environment variables
   listed in the README), plus an int8 tensor-core FFN prefill prototype.

All of these are gated by runtime options or keep the default path unchanged; the
default behavior is the upstream behavior except that the staged-KV machinery is
compiled in and inactive.

## Excluded upstream files

The following upstream automation is deliberately omitted because it belongs to
ggml-org and would attempt to build, test or publish on this fork's behalf (and some
jobs would fail on the intentionally divergent tree):

- `.github/workflows/` (entire directory), including `build-*.yml`,
  `ci-self-hosted-*.yml`, `server.yml`, `docker.yml`, `winget.yml`, `release.yml`,
  `bench.yml`, `labeler.yml`, `editorconfig.yml`, `stale.yml`, `ai-issues.yml`,
  `check-vendor.yml`, `close-issue.yml`, `pull-reviews.yml` and the remaining
  workflow files.

The rest of `.github/` (issue templates, pull request template, actions used by
documentation) is retained. No upstream branch, tag or remote is configured.

## Not included

Model weights and datasets are not distributed here. The GGUF files used by the
reports and the large raw measurement tree (`results/`, `comm/`, traces) are not part
of this repository. The Windows CUDA build in `build-cuda/bin` is a release artifact,
not committed source.
