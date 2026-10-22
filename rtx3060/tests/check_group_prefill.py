"""Compare group-prefill state against a full forward, then restore with staged KV."""
import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--output", type=Path, default=ROOT / "results" / ("group-prefill-check-" + str(int(time.time()))))
ap.add_argument("--ubatch", type=int, default=128)
ap.add_argument("--tokens", type=int, default=2048)
ap.add_argument("--trace", action="store_true")
args = ap.parse_args()
OUT = args.output.resolve()
OUT.mkdir(exist_ok=True)
EXE = ROOT / ".." / "build-cuda" / "bin" / "group_prefill.exe"
BASE = [str(EXE), "--model", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf",
        "--prompt", "code_prompt.txt", "--ctx", "4096", "--tokens", str(args.tokens), "--ubatch", str(args.ubatch)]


def run(name, extra_args):
    path = OUT / (name + ".log")
    start = time.monotonic()
    with path.open("w", encoding="utf-8") as f:
        trace = ["--trace-dir", str(OUT / (name + "-trace"))] if args.trace and name in ("split", "reference") else []
        p = subprocess.run(BASE + extra_args + trace, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=600)
    text = path.read_text(encoding="utf-8", errors="replace")
    print(name, "exit", p.returncode, "seconds", round(time.monotonic() - start, 2), flush=True)
    print("\n".join(s for s in text.splitlines() if s.startswith(("BEGIN", "PROGRESS", "SWAP", "SAVED", "COMPLETE", "ERROR"))), flush=True)
    if p.returncode:
        raise RuntimeError(text[-5000:])
    return text


split = OUT / "split.bin"
ref = OUT / "reference.bin"
run("split", ["--out", str(split), "--nextn-out", str(OUT / "split.nextn")])
run("reference", ["--mode", "reference", "--out", str(ref), "--nextn-out", str(OUT / "reference.nextn")])
with split.open("rb") as f:
    split_hash = hashlib.file_digest(f, "sha256").hexdigest()
with ref.open("rb") as f:
    ref_hash = hashlib.file_digest(f, "sha256").hexdigest()
if split_hash != ref_hash:
    raise RuntimeError("split and reference snapshots differ")
with (OUT / "split.nextn").open("rb") as f:
    hidden_hash = hashlib.file_digest(f, "sha256").hexdigest()
with (OUT / "reference.nextn").open("rb") as f:
    if hidden_hash != hashlib.file_digest(f, "sha256").hexdigest():
        raise RuntimeError("target hidden-state histories differ")
rows = {}
for name, snap, staged in [("ref-restored", ref, 0), ("split-restored", split, 0), ("split-staged", split, 8)]:
    logits = OUT / (name + ".f32")
    text = run(name, ["--mode", "restore", "--state", str(snap), "--staged", str(staged), "--logits", str(logits)])
    a = array.array("f")
    a.frombytes(logits.read_bytes())
    if not all(math.isfinite(x) for x in a):
        raise RuntimeError("nonfinite logits")
    rows[name] = (a, [s for s in text.splitlines() if s.startswith("TOKEN ")])

baseline, baseline_tokens = rows["ref-restored"]
summary = {"snapshot_sha256": split_hash, "target_hidden_sha256": hidden_hash}
for name, (values, tokens) in rows.items():
    diffs = [float(a) - b for a, b in zip(values, baseline)]
    summary[name] = {"max_abs": max(map(abs, diffs)), "rms": math.sqrt(sum(x*x for x in diffs)/len(diffs)),
                     "same_tokens": tokens == baseline_tokens, "n_logits": len(values)}
    if len(values) != len(baseline) or not summary[name]["same_tokens"] or summary[name]["max_abs"] > 1e-4:
        raise RuntimeError(f"parity failed: {name} {summary[name]}")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("PASS", json.dumps(summary), flush=True)
