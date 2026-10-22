"""T2: DFlash draft lengths at one context, from a single prefill.

Mode dflash: start the server with draft-dflash, prefill (or restore), save the target state,
then issue one completion per --drafts value against the same prompt (cache_prompt reuses the
prefilled prefix). Writes tokens and timings per draft length.
Mode none: no speculation, restore the saved target state, decode --n tokens as the reference.

  python tests/t2_dflash.py --mode dflash --prompt-file results/charts/prompts/t128k.txt \
      --ctx 131584 --budget 1422 --drafts 4,5,6,7 --n 512 --save t2-xx-128k-dflash.bin \
      --out results/prof/t2_dflash128k.json --port 9985
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
DRAFT = "Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf"
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["dflash", "none"], default="dflash")
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--budget", type=int, default=None)
ap.add_argument("--staged", type=int, default=None)
ap.add_argument("--snapshot", default="")
ap.add_argument("--save", default="")
ap.add_argument("--drafts", default="4,5,6,7")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--draft-ubatch", type=int, default=16)
ap.add_argument("--draft-ngl", type=int, default=99)
ap.add_argument("--draft-model-path", default="")
ap.add_argument("--ub", type=int, default=128)
ap.add_argument("--output", default="CUDA0", choices=["CUDA0", "CPU"])
ap.add_argument("--port", type=int, default=9985)
ap.add_argument("--out", required=True)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--vram", action="store_true")
ap.add_argument("--samplers", default="")
ap.add_argument("--backend-sampling", action="store_true")
ap.add_argument("--verbosity", default="2")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", a.port)) == 0:
        raise RuntimeError(f"port {a.port} in use")

env = dict(os.environ)
env.update({"LLAMA_KV_STAGE_SLOTS": "2", "LLAMA_FA_VEC_QUANT_KV": "1", "LLAMA_FA_VEC_Q4_COLS": "2"})
if a.staged is not None:
    env["LLAMA_KV_STAGED_LAYERS"] = str(a.staged)
if a.budget is not None:
    env["LLAMA_KV_GPU_BUDGET_MIB"] = str(a.budget)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v

cmd = [EXE, "-m", MODEL, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
       "-b", "512", "-ub", str(a.ub), "-np", "1", "--fit", "off", "--no-context-shift", "--jinja",
       "--port", str(a.port), "--log-verbosity", a.verbosity, "--slot-save-path", SLOTS,
       "-ot", f"^output\\.weight$={a.output}"]
if a.samplers:
    cmd += ["--samplers", a.samplers]
if a.backend_sampling:
    cmd += ["--backend-sampling"]
if a.mode == "dflash":
    cmd += ["--spec-type", "draft-dflash", "--spec-draft-model", (a.draft_model_path or DRAFT), "--spec-draft-ngl", str(a.draft_ngl),
            "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
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
        t0 = time.time()
        r = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                                 "cache_prompt": True, "ignore_eos": True})
        tt = r.get("timings", {})
        rec["prefill"] = {"prompt_n": tt.get("prompt_n"), "prompt_ms": tt.get("prompt_ms"), "wall_s": time.time() - t0}
        print(f"prefill {tt.get('prompt_n')} tok in {(tt.get('prompt_ms') or 0)/1e3:.1f} s")

    if a.save:
        sv = post("/slots/0?action=save", {"filename": a.save})
        rec["save"] = {"name": a.save, "ok": not (sv or {}).get("error"),
                       "error": (sv or {}).get("error"), "raw": sv if not (sv or {}).get("error") else None}
        print("saved", a.save, rec["save"]["ok"])

    if a.mode == "none":
        t0 = time.time()
        r = post("/completion", {"prompt": prompt, "n_predict": a.n, "temperature": 0.0,
                                 "cache_prompt": True, "ignore_eos": True, "return_tokens": True})
        tt = r.get("timings", {})
        toks = r.get("tokens") or []
        rec["runs"]["none"] = {"timings": tt, "n_tokens": len(toks),
                               "sha": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16],
                               "tokens": toks, "wall_s": time.time() - t0}
        print(f"none: {tt.get('predicted_n')} tok {tt.get('predicted_per_second', 0):.2f} t/s")
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
