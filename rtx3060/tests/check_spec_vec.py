"""Compare wider quantized attention against the existing two-column kernel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / f"check-spec-vec-{int(time.time())}"
OUT.mkdir()
ap = argparse.ArgumentParser()
ap.add_argument("--queries", type=int, nargs="+", default=[3, 4, 5, 8, 17])
ap.add_argument("--staged", type=int, default=8)
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--trace", action="store_true")
args = ap.parse_args()
results = []
for q in args.queries:
    runs = []
    for cols in (2, args.cols):
        stem = OUT / f"q{q}-cols{cols}"
        cmd = [str(ROOT / "../build-cuda/bin/group_prefill.exe"), "--mode", "restore",
               "--model", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf", "--prompt", "code_prompt.txt",
               "--ctx", "4096", "--tokens", str(2111+q-1), "--ubatch", "128", "--staged", str(args.staged),
               "--state", "results/.kv_cache/kv-mtp-short-target.bin", "--generate", "1" if args.trace else "4", "--logits", str(stem)+".f32"]
        if args.trace:
            cmd += ["--trace-dir", str(stem)+"-trace"]
        env = dict(os.environ, LLAMA_FA_VEC_Q4_COLS=str(cols))
        with Path(str(stem)+".log").open("w") as log:
            subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        values = np.fromfile(str(stem)+".f32", dtype=np.float32)
        tokens = [s for s in Path(str(stem)+".log").read_text().splitlines() if s.startswith("TOKEN ")]
        runs.append((values, tokens))
    error = float(np.max(np.abs(runs[0][0]-runs[1][0])))
    if not np.isfinite(error) or error > 1e-4 or runs[0][1] != runs[1][1]:
        raise RuntimeError(f"q={q} error={error} token_equal={runs[0][1] == runs[1][1]}")
    results.append(dict(queries=q, max_logit_error=error, tokens_equal=True))
    print(results[-1], flush=True)
(OUT / "summary.json").write_text(json.dumps(results, indent=2))
print(OUT, flush=True)
