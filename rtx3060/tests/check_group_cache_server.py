"""Check a produced 200K cache across staging depths and through the server slot API."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / ".." / "build-cuda" / "bin"
MODEL = ROOT / ".." / "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--state", type=Path, required=True)
ap.add_argument("--prompt", type=Path, default=ROOT / "big.prompt.txt")
ap.add_argument("--ctx", type=int, default=204800)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
state = args.state.resolve()
prompt = args.prompt.resolve()
rows = {}
for staged in (9, 10):
    logfile, logits = out / f"restore-L{staged}.log", out / f"restore-L{staged}.f32"
    command = [str(BIN / "group_prefill.exe"), "--model", str(MODEL), "--prompt", str(prompt),
               "--mode", "restore", "--state", str(state), "--ctx", str(args.ctx),
               "--ubatch", "128", "--staged", str(staged), "--logits", str(logits)]
    start = time.monotonic()
    with logfile.open("w", encoding="utf-8") as f:
        p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=240,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode:
        raise RuntimeError(logfile.read_text(encoding="utf-8", errors="replace")[-3000:])
    values = array.array("f")
    values.frombytes(logits.read_bytes())
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError("nonfinite logits")
    tokens = [s for s in logfile.read_text(encoding="utf-8").splitlines() if s.startswith("TOKEN ")]
    rows[staged] = (values, tokens)
    print(f"RESTORE L={staged} {time.monotonic()-start:.2f}s", flush=True)
a, ta = rows[9]
b, tb = rows[10]
if len(a) != len(b):
    raise RuntimeError("logit dimensions differ")
max_abs = max(abs(float(x) - y) for x, y in zip(a, b))
summary = {"max_abs_logits_L9_vs_L10": max_abs, "same_16_tokens": ta == tb}
if max_abs > 1e-5 or ta != tb:
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    raise RuntimeError(f"staging parity failed: {summary}")

port = 9962
base = f"http://127.0.0.1:{port}"


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


env = dict(os.environ)
for k in ("LLAMA_PREFILL_LAYER_RANGE", "LLAMA_KV_GPU_BUDGET_MIB"):
    env.pop(k, None)
env.update(LLAMA_KV_STAGED_LAYERS="9", LLAMA_KV_STAGE_SLOTS="2", LLAMA_FA_VEC_QUANT_KV="1")
command = [str(BIN / "llama-server.exe"), "-m", str(MODEL), "-c", str(args.ctx), "-ngl", "99",
           "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "128", "-np", "1",
           "--fit", "off", "--no-context-shift", "--port", str(port),
           "--slot-save-path", str(state.parent) + os.sep]
with (out / "server.log").open("w", encoding="utf-8") as f:
    p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for _ in range(90):
            if p.poll() is not None:
                raise RuntimeError("server exited; inspect server.log")
            try:
                if request("/health").get("status") == "ok":
                    break
            except (OSError, urllib.error.HTTPError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server startup timeout")
        restored = request("/slots/0?action=restore", {"filename": state.name})
        response = request("/completion", {"prompt": prompt.read_bytes().decode("utf-8"), "n_predict": 16,
                                           "temperature": 0, "ignore_eos": True, "cache_prompt": True})
        (out / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
        timings = response["timings"]
        cached = timings.get("cache_n", response.get("tokens_cached", 0))
        if cached < 200000 or timings.get("prompt_n", 0) > 2:
            raise RuntimeError(f"server did not reuse the expected prompt: {timings}")
        summary.update(server_restore=restored, server_timings=timings,
                       content_sha256=hashlib.sha256(response["content"].encode()).hexdigest())
    finally:
        p.terminate()
        p.wait(timeout=30)
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("PASS", json.dumps(summary), flush=True)
