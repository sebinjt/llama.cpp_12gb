# rtx3060 - Qwen3.8-27B on a 12 GiB RTX 3060

Project material for the llama.cpp fork at the repository root. The source tree itself
is the repo root; this directory only contains measurements, notes, launchers and
benchmarks.

| Path | Contents |
|---|---|
| [docs/report.md](docs/report.md) | **Start here.** Measured project log. |
| [docs/improvement_directions.md](docs/improvement_directions.md) | Parked ideas and their numbers. |
| [docs/qwen27b_rtx3060_report.html](docs/qwen27b_rtx3060_report.html) | Rendered report. |
| [docs/design/](docs/design/) | Staged-KV, group-prefill and host-KV design notes. |
| [docs/source-tree-notes.md](docs/source-tree-notes.md) | What was published and why `llama.cpp-w3` is excluded. |
| [scripts/serve.ps1](scripts/serve.ps1) | Everyday server launcher. |
| [scripts/build-cuda.bat](scripts/build-cuda.bat) | Windows CUDA 13.4 build for sm_86. |
| [tests/serve_prod.ps1](tests/serve_prod.ps1) | Production profiles measured in the report. |
| [tests/README.md](tests/README.md) | Benchmark/eval scripts and their assumptions. |
| `results/` | Output directory (empty in Git). |

## Quick start

```bat
rtx3060\scripts\build-cuda.bat
```

Put the GGUF model files in the repository root, then:

```powershell
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile p8k -DryRun
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile p8k -Port 8091
```

See the top-level [README](../README.md) for the feature list, environment variables,
release-build notes and provenance.
