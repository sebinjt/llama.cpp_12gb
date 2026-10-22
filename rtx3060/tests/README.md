# Tests, benchmark launchers and evaluation scripts

These are the scripts that produced the measurements in
[../docs/report.md](../docs/report.md). They are research code: the historical scripts
assume a specific set of local model files, prompt fixtures and result directories.
Paths have been adjusted for the published layout, but the data is not shipped.

## Layout assumptions

- Repository root = the llama.cpp source tree = two levels above this directory.
- Built binaries: `<repo>/build-cuda/bin/` (see
  [`../scripts/build-cuda.bat`](../scripts/build-cuda.bat)).
- GGUF models: `<repo>/` (repository root).
- Script outputs: `rtx3060/results/` (empty in Git; created by the scripts).
- Python scripts compute `ROOT` as `rtx3060/` and use `ROOT/..` for the repository
  root. PowerShell `serve_prod.ps1` resolves the repository root from `$PSScriptRoot`.

## Main launcher

```powershell
# production profiles measured in the report
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile p8k  -Port 8091
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile plong -Ctx 32768 -Port 8092
powershell -ExecutionPolicy Bypass -File rtx3060\tests\serve_prod.ps1 -Profile s64k -DryRun
```

`-DryRun` prints the exact environment, route and command line without launching.
`-NoSpec`, `-Ctk/-Ctv`, `-Batch/-Ub`, `-Env NAME=VALUE` and `-ExtraArgs` cover the
A/B checks described in the report.

The simpler interactive launcher is
[`../scripts/serve.ps1`](../scripts/serve.ps1) (96K context, staged layers, MTP).

## Useful benchmark scripts

| Script | Purpose |
|---|---|
| `w2_dflash.py` | Core server benchmark: prompt/prefill/snapshot, DFlash or MTP, JSON + VRAM. |
| `t2_dflash.py` | DFlash context sweep. |
| `kvrun.py` | KV-quant variants layered on `w2_dflash.py`. |
| `stock_bench.py` | Same flow against an unmodified stock llama.cpp binary. |
| `realworld_chat.py` | Multi-turn / long-context checks through the OpenAI API. |
| `dq4_e2e.py`, `dq4_kld.py`, `dq4_saverestore.py` | q4_0 decode-attention exactness/KLD and restore checks. |
| `serve_smoke.py`, `run_staged_server.py`, `run_stock_server.py` | Smoke and comparison runs. |
| `check_spec_vec.py`, `check_group_prefill.py`, `check_group_cache_server.py` | Correctness checks for speculation, grouped prefill and cache restore. |
| `compare_logs.py`, `compare_64k.py`, `summ*.py`, `*_table.py` | Log parsing and report tables. |
| `*_bench.cpp`, `*_bench.cu` | Standalone microbenchmarks (flash attention, MMQ/MMVQ, mmsb, cublas, CPU attention). |
| `build_*.bat` | Build the corresponding microbenchmark; run from `rtx3060/`. |

## Missing fixtures

The original measurement tree also contained files that are not published:

- prompts under `results/` (e.g. `results/dq4/code_short.prompt.txt`),
- KV snapshots under `results/.kv_cache/`,
- `comm/` review logs and `results/w2/` logs referenced by `realworld_chat.py`,
- the stock `llama-b11717-bin-win-cuda-13.4-x64` binary used by `stock_bench.py`.

Scripts that need them accept an explicit path on the command line. The
public-domain benchmark text `tests/pride-and-prejudice.txt` is included because
several scripts use it as a long prompt.

## Notes

- The GPU is a single RTX 3060; do not run concurrent benchmarks. The private
  `comm/GPU_LOCK_RULES.md` used by the authors is not published, so use an equivalent
  external lock if you parallelise work.
- Most environment variables are documented in the top-level README and in the
  report sections named in each script header.
- `_*.cu.txt`, `_*_insert*.cu`, `_w2_int8_patch.py` and `patch_pair_layoutB.py` are
  development patch fragments kept for provenance, not standalone tests.
