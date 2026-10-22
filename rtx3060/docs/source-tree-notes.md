# Published source tree and the shelved `llama.cpp-w3` experiment

## What was published

- Source: the **current working tree** of `llama_cpp_src\llama.cpp`, including
  uncommitted changes. It was copied to the repository root without the upstream
  `.git` history.
- Upstream base: `547208dca250ceb3e4c8615598d043f2b602a527` (merge-base with
  `origin/master`).
- Fork commit inherited in the working tree: `094c9ecce` ("staged host KV: pinned host
  canonical KV with per-layer device staging slots") and the merge commit
  `acb612ff15772f019abd59c653c4ec37cb9025b7`.
- The working tree additionally contained **uncommitted modifications** in 21 tracked
  files and 3 untracked files:

  ```
  ?? ggml/src/ggml-cuda/fattn-decode-q4.cu
  ?? ggml/src/ggml-cuda/fattn-decode-q4.cuh
  ?? src/llama-prefill-range.h
  M  common/sampling.cpp
  M  common/speculative.cpp
  M  ggml/src/ggml-cuda/fattn-common.cuh
  M  ggml/src/ggml-cuda/fattn-vec.cuh
  M  ggml/src/ggml-cuda/fattn.cu
  M  ggml/src/ggml-cuda/fattn.cuh
  M  ggml/src/ggml-cuda/gated_delta_net.cu
  M  ggml/src/ggml-cuda/ggml-cuda.cu
  M  ggml/src/ggml-cuda/mmvq.cu
  M  ggml/src/ggml-cuda/staged-kv.cuh
  M  src/llama-context.cpp
  M  src/llama-cparams.h
  M  src/llama-kv-cache.cpp
  M  src/llama-kv-cache.h
  M  src/llama-memory-recurrent.cpp
  M  src/llama-memory-recurrent.h
  M  src/llama-model.cpp
  M  src/models/delta-net-base.cpp
  M  src/models/qwen35.cpp
  M  tools/server/server-common.cpp
  M  tools/server/server-context.cpp
  ```

  These are the decode-attention split, the exact speculative-decoding changes and the
  int8 prefill prototype described in the report; they are part of the published tree.

- Excluded from the published tree: the upstream `.git` directory and
  `.github/workflows/`. Everything else in the source tree (including `vendor/`,
  `licenses/`, `docs/`, `scripts/`, examples and tests) is present.
- Build output: `llama_cpp_src\build-cuda\bin` was built from the main tree on
  2026-10-21 12:43 and is published separately as a Windows release zip. It is not
  copied into Git.

## `llama.cpp-w3` compared with `llama.cpp`

`llama.cpp-w3` is a **shelved experiment copy** for the v9 paired 512-k weight-layout
route (report section 57). Diff command used:

```bat
diff -rq llama.cpp llama.cpp-w3 -x .git -x "build*" -x "*.gguf"
```

Files that differ:

| File | What w3 has that the main tree does not |
|---|---|
| `ggml/src/ggml-cuda/mmsb.cu` | New 703-line v9 kernel and in-place repacker for the paired 512-k layout (IQ3_S / IQ3_XXS). |
| `ggml/src/ggml-cuda/mmsb.cuh` | New 35-line interface and env-route declarations. |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | ~192 lines of paired-layout MMQ loaders (`..._iq3_s_mmsb`, `..._iq3_xxs_mmsb`). |
| `ggml/src/ggml-cuda/mmq.cuh` | Selects the mmsb loader through a `mmsb_pair` kernel argument; adds that argument through the launch path. |
| `ggml/src/ggml-cuda/mmq.cu` | Routes eligible `ncols <= 16` launches to `ggml_cuda_mul_mat_mmsb`; adds `GGML_MM_ROUTE` logging. |
| `ggml/src/ggml-cuda/mmvq.cu` | Adds `GGML_MM_ROUTE` logging only. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | mmsb dispatch, fusion-skip guard for packed weights, backend proc addresses for repack/finalize. |
| `src/llama-model.cpp` | Calls the in-place repack for all tensors after load when `GGML_CUDA_MMSB` is set. |
| `patch3.py` | Helper that applies the mmq.cuh portion of the patch (development scratch). |

The main tree also contains the **int8 tensor-core FFN prefill prototype**
(`LLAMA_CUDA_INT8_FFN`) that is absent from `llama.cpp-w3`; w3 is therefore behind the
main tree as well as ahead of it.

### Decision: exclude w3

Nothing in w3 is missing from the published project in the sense of a fix or a working
feature that the main tree needs. It contains only the v9 paired-layout experiment,
which report section 57 measured as **exact and faster in isolation but slower
in-model** (about +3% decode at n=1 with CUDA graphs and a small PP change), and which
improvement_directions.md section 2b records as shelved with one open lead. The
default main-tree path is unchanged by omitting it. The experiment is documented here
so it can be resurrected by porting the files above; keeping a second full source tree
in the published repo would only create an unbuildable duplicate.
