"""Task B helper (worker2 seq 114): one KV-quant config -> JSON, using tests/w2_dflash.py under the hood.

Wraps the common env so a variant is one short spec, e.g.:
  python tests/kvrun.py --ctx 16384 --prompt results/dq4/code_short.prompt.txt --ctk q8_0 --ctv q4_0 \
      --variant novq --out results/kv/b8k_q8q4_novq.json
variants:  base (current env) | novq (no LLAMA_FA_VEC_QUANT_KV) | nq4 (LLAMA_FA_VEC_MAX_NQ=4) |
           noq4 (no LLAMA_FA_DECODE_Q4, i.e. stock FA routing) | f16 (ctk/ctv f16)
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--ctk", default="q4_0")
ap.add_argument("--ctv", default="q4_0")
ap.add_argument("--variant", default="base", choices=["base", "novq", "nq4", "noq4", "noq4_novq"])
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--port", type=int, default=12001)
ap.add_argument("--out", required=True)
ap.add_argument("--model-path", default=os.path.join(ROOT, "..", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"))
a = ap.parse_args()

COMMON = ["LLAMA_RS_REPLAY=1", "LLAMA_MTP_WINDOW=8192", "LLAMA_MMVQ_BATCH_INVARIANT=1",
          "LLAMA_MMVQ_UNPACK_ONCE=0", "LLAMA_STEP_TIMING=1", "GGML_CUDA_PREFER_MMQ=1"]
args = [sys.executable, os.path.join(ROOT, "tests", "w2_dflash.py"),
        "--mode", "dflash", "--drafts", "7",
        "--prompt-file", a.prompt, "--ctx", str(a.ctx), "--n", str(a.n),
        "--ub", "128", "--b", "512", "--draft-ubatch", "16", "--draft-ngl", "99",
        "--model-path", a.model_path, "--ctk", a.ctk, "--ctv", a.ctv,
        "--out", a.out, "--port", str(a.port), "--vram", "--verbosity", "2"]
envs = list(COMMON)
if a.variant in ("base", "nq4"):
    envs += ["LLAMA_FA_DECODE_Q4=1", "LLAMA_FA_VEC_QUANT_KV=1", "LLAMA_FA_VEC_MAX_NQ=128"]
if a.variant == "nq4":
    envs[-1] = "LLAMA_FA_VEC_MAX_NQ=4"
if a.variant == "noq4":
    envs += ["LLAMA_FA_VEC_QUANT_KV=1", "LLAMA_FA_VEC_MAX_NQ=128"]
unset = []
if a.variant in ("novq", "noq4_novq"):
    envs += ["LLAMA_FA_DECODE_Q4=1", "LLAMA_FA_VEC_MAX_NQ=128"]
    unset = ["--unset-env", "LLAMA_FA_VEC_QUANT_KV"]
for e in envs:
    args += ["--env", e]
args += unset
print("variant", a.variant, "ctk", a.ctk, "ctv", a.ctv, "->", a.out)
r = subprocess.run(args, cwd=ROOT)
sys.exit(r.returncode)
