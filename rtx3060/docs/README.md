# Project documentation

Start with **[report.md](report.md)**. It is the living measured log: setup, every
configuration tried, numbers, failures and the conclusions. Sections 0, 4, 5, 8 and 10
are the short version; sections 26-60 cover the decode-attention, speculative-decoding,
weight-streaming and KV-quant work.

| Document | Contents |
|---|---|
| [report.md](report.md) | Main measured project log (long). |
| [qwen27b_rtx3060_report.html](qwen27b_rtx3060_report.html) | Rendered version of the report for reading in a browser. |
| [improvement_directions.md](improvement_directions.md) | Parked ideas with the numbers behind them and what each would need. |
| [source-tree-notes.md](source-tree-notes.md) | What was published from the working tree, the shelved `llama.cpp-w3` experiment, and why it is excluded. |
| [design/staged-kv-offload-design.md](design/staged-kv-offload-design.md) | Original staged-host-KV design. |
| [design/staged-kv-implementation.md](design/staged-kv-implementation.md) | Implementation notes for the staged KV cache. |
| [design/host-kv-baseline.md](design/host-kv-baseline.md) | Host-KV baseline measurements. |
| [design/final-test-findings.md](design/final-test-findings.md) | Early end-to-end findings. |
| [design/group-prefill.md](design/group-prefill.md) | Offline layer-group prefill and snapshot production. |
| [staged-kv-root-cause.md](staged-kv-root-cause.md) | Why the staged path behaved the way it did. |
| [amd_borrow_plan.md](amd_borrow_plan.md) | What was borrowed from the AMD/other-backend work, and what was rejected. |
| [ninfer-borrow-notes.md](ninfer-borrow-notes.md) | Notes from comparing with the `ninfer` implementation. |
| [experiments_2026-10-17.md](experiments_2026-10-17.md) | Early experiment log. |
| [timeline-explained.md](timeline-explained.md) | Reading the profiler timeline. |

## Path conventions in the historical documents

The documents were written while the fork lived under `llama_cpp_src\llama.cpp` and
the study files lived in a parent directory. When reading commands:

- `llama_cpp_src\llama.cpp\...` is the repository root in this published repo.
- `llama_cpp_src\build-cuda\bin\...` is `build-cuda\bin\...` at the repository root.
- `results\...`, `comm\...`, KV snapshots, `nsys` traces and model GGUFs were
  local measurement artifacts and are **not** included in this repository.
- The launchers under `rtx3060\tests\` and `rtx3060\scripts\` have already been
  adjusted to the published layout; see [../tests/README.md](../tests/README.md).
