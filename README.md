# Qwen3.8-27B on an RTX 3060: a llama.cpp fork

Inference for **Qwen3.8-27B** (hybrid: 16 attention + 48 GDN layers) on a single
**RTX 3060 12 GiB**, up to **200K context**, with a staged host KV cache, a split-K
q4_0 decode-attention kernel, opt-in weight streaming and exact speculative decoding
(MTP / DFlash). Every performance number in the project docs was measured on the
reference machine.

This is a **permanently diverged fork** of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
It is not intended to be merged upstream and there is no upstream branch to track.

- **Upstream base commit:** `547208dca250ceb3e4c8615598d043f2b602a527`
  (upstream `master`; computed with
  `git -C llama_cpp_src/llama.cpp merge-base HEAD origin/master`).
- **Fork work:** commit `094c9ecce` ("staged host KV: pinned host canonical KV with
  per-layer device staging slots") plus the **uncommitted working-tree changes** that
  are published here. The forked source tree is the repository root.
- Upstream Git history is deliberately **not** included; provenance is recorded in
  [`NOTICE.md`](NOTICE.md), not by a remotes/branches relationship.

The usual llama.cpp CMake build works from the repository root. Project-specific
material (measured report, design notes, launchers, benchmarks) lives in
[`rtx3060/`](rtx3060/).

**Browsable report (GitHub Pages front page):** [`docs/index.html`](docs/index.html).
Enable Pages for this repository with **Source: Deploy from a branch**, branch `main`,
folder `/docs`; the site is then served at `https://<user>.github.io/<repo>/`. The same
file is kept at [`rtx3060/docs/qwen27b_rtx3060_report.html`](rtx3060/docs/qwen27b_rtx3060_report.html).

## What this fork changes

- **Staged host KV (the main change).** The canonical KV cache can live in pinned host
  memory while a small, configurable number of per-layer device slots are filled over
  PCIe during the GDN/MLP compute gap. One sequence per graph; the number of staged
  layers and the GPU KV budget are runtime knobs. New
  `ggml/src/ggml-cuda/staged-kv.cuh`, plus changes in `src/llama-kv-cache.*`,
  `src/llama-context.cpp`, `src/llama-cparams.h`, `src/llama-memory-recurrent.*` and
  `ggml/src/ggml-cuda/ggml-cuda.cu`. Environment:
  `LLAMA_KV_STAGED_LAYERS`, `LLAMA_KV_GPU_BUDGET_MIB`, `LLAMA_KV_STAGE_SLOTS`,
  `LLAMA_KV_STAGE_TRACE`, `LLAMA_KV_STAGE_JOIN`, `LLAMA_KV_STAGE_NODMA`,
  `GGML_KV_STAGE_PROF`.
- **Split-K q4_0 decode attention.** New `fattn-decode-q4.cu` / `.cuh` with a
  token-count split (decode vs. prefill), a q4_0 vec path that avoids the full-context
  f16 scratch, and GQA-packed decoding. Environment:
  `LLAMA_FA_DECODE_Q4`, `LLAMA_FA_DECODE_Q4_SPLITS`, `LLAMA_FA_DECODE_Q4_STAGES`,
  `LLAMA_FA_DECODE_Q4_MAXQ`, `LLAMA_FA_Q4_SPLIT_TOKENS`, `LLAMA_FA_VEC_MAX_NQ`,
  `LLAMA_FA_VEC_QUANT_KV`, `LLAMA_FA_VEC_GQA` (see `ggml/src/ggml-cuda/fattn*.{cu,cuh}`).
- **Exact speculative decoding.** Replay-based rollback for tree/lattice drafts,
  MTP-window control, batch-invariant MMVQ reduction, and DFlash lattice handling in
  `common/speculative.cpp`, `common/sampling.cpp` and the server stats. Environment:
  `LLAMA_RS_REPLAY`, `LLAMA_MTP_WINDOW`, `LLAMA_DFLASH_LATTICE`,
  `LLAMA_MMVQ_BATCH_INVARIANT`, `LLAMA_MMVQ_UNPACK_ONCE`.
- **Weight streaming.** Opt-in double-buffered weight staging
  (`LLAMA_W_STAGE_ENABLE`, `LLAMA_W_STAGE_BLOCKS`, `LLAMA_W_STAGE_MIB`,
  `LLAMA_W_STAGE_SLOT_MIB`, `LLAMA_W_STAGE_PINNED`, `LLAMA_W_STAGE_PREFETCH`,
  `LLAMA_W_STAGE_PERIOD_MS`, `LLAMA_W_STAGE_CHECK`, `LLAMA_W_STAGE_TRACE`), with the
  Gated Delta Net kernels (`gated_delta_net.cu`, `models/delta-net-base.cpp`,
  `models/qwen35.cpp`) participating in the overlap.
- **Prefill layer ranges.** `src/llama-prefill-range.h` and
  `LLAMA_PREFILL_LAYER_RANGE` for offline group-prefill / snapshot workflows.
- **Measurement and diagnostics hooks.** `GGML_MM_TIMING`, `GGML_MM_STATS`,
  `GGML_MM_ROUTE`, `GGML_GDN_TIMING`, `LLAMA_STEP_TIMING`, `LLAMA_DUMP_KV`,
  `LLAMA_KV_STAGE_TRACE`; server-side draft verification counters.
- **Opt-in prototype:** an int8 tensor-core FFN prefill path
  (`LLAMA_CUDA_INT8_FFN`, `LLAMA_CUDA_INT8_FFN_LAYERS`, `LLAMA_CUDA_INT8_MIN_TOKENS`).
  It is off unless enabled and is not the default.

Defaults are unchanged: with none of the environment variables above set, the code
takes the corresponding upstream path (apart from the staged-KV implementation being
present but inactive).

## Repository layout

```
.                       # llama.cpp source tree (build from here)
├── ggml/ src/ common/ tools/ ...   # forked llama.cpp, including uncommitted work
├── rtx3060/
│   ├── docs/
│   │   ├── report.md                     # measured project log (main document)
│   │   ├── improvement_directions.md     # parked ideas with numbers
│   │   ├── qwen27b_rtx3060_report.html   # rendered report
│   │   ├── design/                       # staged-KV, group-prefill, host-KV notes
│   │   └── *.md                          # root-cause / borrow analyses
│   ├── scripts/
│   │   ├── serve.ps1                     # convenient launcher
│   │   └── build-cuda.bat                # Windows CUDA 13.4 / sm_86 build
│   ├── tests/                            # benchmark/launcher/eval scripts
│   └── results/                          # output directory (empty in git)
├── LICENSE                              # MIT, upstream ggml authors
└── NOTICE.md                            # provenance + change summary
```

## Build (Windows, CUDA)

Prerequisites: CMake, Ninja, MSVC 2022, CUDA toolkit (the measured builds used
CUDA 13.4), an NVIDIA driver and a GPU of compute capability 8.6 (RTX 3060).

```bat
:: from the repository root
rtx3060\scripts\build-cuda.bat
```

Or configure it yourself; this is the standard llama.cpp build:

```bat
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target llama-cli llama-server llama-bench
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for a different GPU. The launchers expect binaries
in `build-cuda\bin`.

## Run

Put the GGUF files in the repository root (they are not distributed here). The
production profiles measured in `report.md` are launched with:

```powershell
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile p8k -Port 8091
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile plong -Ctx 32768 -Port 8092
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile s64k -DryRun
```

`-DryRun` prints the exact environment and command line without starting anything.
For an interactive default configuration use `rtx3060\scripts\serve.ps1` (96K context,
q4_0 KV, staged layers, MTP). Every flag in these launchers is cited to the report
section that measured it.

## Models and data

Model weights, GGUF snapshots, KV snapshot files, `nsys` traces, huge logs and the
multi-gigabyte raw `results/` tree are **not** in this repository. Scripts default to
these file names in the repository root:

- `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`
- `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`
- `Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`

Some historical benchmark scripts also expect prompt fixtures under
`rtx3060/results/` or private `comm/` review logs; those paths are documented in
[`rtx3060/tests/README.md`](rtx3060/tests/README.md).

## Windows CUDA release build

A prebuilt Windows x64 CUDA 13.4 build can be published as a GitHub release asset
(the `build-cuda\bin` directory, zipped). It is a build artifact of the working tree at
the time of packaging, not a separate source release. No binaries are committed to
this Git repository.

## Excluded upstream automation

The upstream `.github/workflows/` directory is **not** included. Those workflows are
owned by ggml-org, would try to build/publish under this project's name, and several
would fail because this fork intentionally diverges. Excluded files include
`build-*.yml`, `ci-self-hosted-*.yml`, `server.yml`, `docker.yml`, `winget.yml`,
`release.yml`, `bench.yml`, `editorconfig.yml`, `labeler.yml`, `stale.yml`,
`ai-issues.yml` and every other workflow in that directory. The rest of `.github/`
(issue templates, pull-request template) is retained for convenience.

## License and provenance

MIT, see [`LICENSE`](LICENSE): Copyright (c) 2023-2026 The ggml authors. This fork
keeps the upstream license and adds no different terms. See [`NOTICE.md`](NOTICE.md)
for the exact upstream commit, the list of principal changes and third-party notices.

## Upstream llama.cpp

The remainder of this file is the upstream project description, kept for credit and
for the standard build/feature documentation. The project-specific text above takes
precedence where the two differ.

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
