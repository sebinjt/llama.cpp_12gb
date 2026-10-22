"""w2: worker2 runner. Same flow as tests/t2_dflash.py, plus --model-path, --ot (repeatable),
--extra (repeatable raw args) and --no-default-ot. Used for the weight-streaming correctness gate.

  python tests/w2_dflash.py --mode none --prompt-file results/prof/t6_8k_q1.txt --ctx 16384 --n 64 \
      --ot "^blk\.(0|1)\.=CPU" --env LLAMA_CUDA_DISABLE_GRAPHS=1 --env LLAMA_W_STAGE_ENABLE=1 \
      --env LLAMA_W_STAGE_BLOCKS=0,1 --env LLAMA_W_STAGE_CHECK=1 --out results/w2/b1_confirm_a.json --port 9970
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
MODEL = os.path.join(ROOT, "..", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
DRAFT = os.path.join(ROOT, "..", "Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf")
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["dflash", "mtp", "none"], default="none")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--budget", type=int, default=None)
ap.add_argument("--staged", type=int, default=None)
ap.add_argument("--snapshot", default="")
ap.add_argument("--save", default="")
ap.add_argument("--drafts", default="7")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--n-predict-1", type=int, default=None, help="tokens for the prefill call (default 1)")
ap.add_argument("--draft-ubatch", type=int, default=16)
ap.add_argument("--draft-ngl", type=int, default=99)
ap.add_argument("--draft-model-path", default="")
ap.add_argument("--ub", type=int, default=128)
ap.add_argument("--b", type=int, default=512)
ap.add_argument("--output", default="CUDA0", choices=["CUDA0", "CPU"])
ap.add_argument("--model-path", default=MODEL)
ap.add_argument("--ot", action="append", default=[])
ap.add_argument("--no-default-ot", action="store_true")
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--port", type=int, default=9985)
ap.add_argument("--out", required=True)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--vram", action="store_true")
ap.add_argument("--verbosity", default="2")
ap.add_argument("--logprobs", type=int, default=0)
ap.add_argument("--ctk", default="q4_0", help="KV cache type for K (default q4_0: unchanged path)")
ap.add_argument("--ctv", default="q4_0", help="KV cache type for V")
ap.add_argument("--draft-ctk", default="", help="draft-model KV type (empty = leave the launcher default)")
ap.add_argument("--draft-ctv", default="")
ap.add_argument("--unset-env", action="append", default=[], help="env vars to REMOVE (the fork tests some by presence)")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", a.port)) == 0:
        raise RuntimeError(f"port {a.port} in use")

env = dict(os.environ)
env.update({"LLAMA_KV_STAGE_SLOTS": "2", "LLAMA_FA_VEC_QUANT_KV": "1", "LLAMA_FA_VEC_Q4_COLS": "2"})
if a.staged is not None and a.staged > 0:
    env["LLAMA_KV_STAGED_LAYERS"] = str(a.staged)
if a.budget is not None:
    env["LLAMA_KV_GPU_BUDGET_MIB"] = str(a.budget)
for k in a.unset_env:
    env.pop(k, None)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v

cmd = [EXE, "-m", a.model_path, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", a.ctk, "-ctv", a.ctv,
       "-b", str(a.b), "-ub", str(a.ub), "-np", "1", "--fit", "off", "--no-context-shift", "--jinja",
       "--port", str(a.port), "--log-verbosity", a.verbosity, "--slot-save-path", SLOTS]
if not a.no_default_ot:
    cmd += ["-ot", ",".join([f"^output\\.weight$={a.output}"] + a.ot)]
elif a.ot:
    cmd += ["-ot", ",".join(a.ot)]
cmd += a.extra
if a.mode == "mtp":
    cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(max(int(x) for x in a.drafts.split(",")))]
    if a.draft_ubatch:
        cmd += ["--spec-draft-n-ubatch", str(a.draft_ubatch)]
elif a.mode == "dflash":
    cmd += ["--spec-type", "draft-dflash", "--spec-draft-model", (a.draft_model_path or DRAFT),
            "--spec-draft-ngl", str(a.draft_ngl),
            "--spec-draft-type-k", (a.draft_ctk or "q4_0"), "--spec-draft-type-v", (a.draft_ctv or "q4_0"),
            "--spec-draft-n-max", str(max(int(x) for x in a.drafts.split(","))),
            "--spec-draft-n-ubatch", str(a.draft_ubatch)]

outname = os.path.join(ROOT, a.out)
os.makedirs(os.path.dirname(outname), exist_ok=True)
lf = open(outname + ".server.log", "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def post(path, body=None, timeout=7200):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:300]}
    except Exception as e:
        return {"error": repr(e)[:300]}


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


def vram():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0])
    except Exception:
        return None


rec = {"mode": a.mode, "args": vars(a), "runs": {}}
try:
    up = False
    for _ in range(500):
        if p.poll() is not None:
            break
        if (get("/health") or {}).get("status") == "ok":
            up = True
            break
        time.sleep(1)
    if not up:
        print("LOAD FAILED", outname + ".server.log")
        sys.exit(2)
    vram_after_load = vram() if a.vram else None
    prompt = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="").read()

    if a.snapshot:
        r = post("/slots/0?action=restore", {"filename": a.snapshot})
        rec["restore"] = {"ok": "error" not in r, "raw": r if "error" in r else None}
        print("restored", a.snapshot)
    else:
        npred = a.n_predict_1 if a.n_predict_1 is not None else 1
        t0 = time.time()
        r = post("/completion", {"prompt": prompt, "n_predict": npred, "temperature": 0.0,
                                 "cache_prompt": True, "ignore_eos": True})
        tt = r.get("timings", {})
        rec["prefill"] = {"prompt_n": tt.get("prompt_n"), "prompt_ms": tt.get("prompt_ms"), "wall_s": time.time() - t0}
        print(f"prefill {tt.get('prompt_n')} tok in {(tt.get('prompt_ms') or 0)/1e3:.1f} s")

    if a.save:
        sv = post("/slots/0?action=save", {"filename": a.save})
        rec["save"] = {"name": a.save, "ok": not (sv or {}).get("error"),
                       "error": (sv or {}).get("error")}
        print("saved", a.save, rec["save"]["ok"])

    if a.mode == "none":
        t0 = time.time()
        body = {"prompt": prompt, "n_predict": a.n, "temperature": 0.0,
                "cache_prompt": True, "ignore_eos": True, "return_tokens": True}
        if a.logprobs:
            body["logprobs"] = a.logprobs
            body["top_logprobs"] = a.logprobs
        r = post("/completion", body)
        tt = r.get("timings", {})
        toks = r.get("tokens") or []
        if a.logprobs:
            rec["probs"] = r.get("completion_probabilities")
        rec["runs"]["none"] = {"timings": tt, "n_tokens": len(toks),
                               "sha": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16],
                               "tokens": toks, "wall_s": time.time() - t0}
        print(f"none: {tt.get('predicted_n')} tok {tt.get('predicted_per_second', 0):.2f} t/s "
              f"sha={rec['runs']['none']['sha']}")
    else:
        ds = [int(x) for x in a.drafts.split(",")]
        if len(ds) != 1:
            raise SystemExit("one draft length per server: n_max is a launch option, pass --drafts D")
        for d in ds:
            t0 = time.time()
            r = post("/completion", {"prompt": prompt, "n_predict": a.n, "temperature": 0.0,
                                     "cache_prompt": True, "ignore_eos": True, "return_tokens": True})
            tt = r.get("timings", {})
            toks = r.get("tokens") or []
            rec["runs"]["d%d" % d] = {"timings": tt, "n_tokens": len(toks),
                                      "sha": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16],
                                      "tokens": toks, "wall_s": time.time() - t0,
                                      "vram_peak": vram() if a.vram else None}
            print(f"d{d}: {tt.get('predicted_n')} tok {tt.get('predicted_per_second', 0):.2f} t/s "
                  f"drafted={tt.get('draft_n')} acc={tt.get('draft_n_accepted')} sha={rec['runs']['d%d' % d]['sha']}")

    rec["vram_after_load"] = vram_after_load
    if a.vram:
        rec["vram_now"] = vram()
    json.dump(rec, open(outname, "w"), indent=1)
    print("wrote", outname)
finally:
    try:
        p.terminate(); p.wait(timeout=120)
    except Exception:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    lf.close()
