"""Worker2 (seq 110 Task A): runner for the UNMODIFIED stock binary
(llama-b11717-bin-win-cuda-13.4-x64\\llama-server.exe), same flow as tests\\w2_dflash.py:

  health -> optional snapshot restore -> optional prefill (save) -> one decode run -> JSON.

No fork env is set: the stock server only sees what is passed here. Extra env is available via --env
(e.g. a stock GGML_* knob if the build has one).

  python tests\\stock_bench.py --prompt-file results\\dq4\\code_short.prompt.txt --ctx 16384 --spec dflash \
      --n-cpu-ffn 8 --out results\\stock\\a8k_ncffn8.json --port 11001
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "llama-b11717-bin-win-cuda-13.4-x64", "llama-server.exe")
MODEL = os.path.join(ROOT, "..", "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf")
DRAFT = os.path.join(ROOT, "..", "Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf")
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep

ap = argparse.ArgumentParser()
ap.add_argument("--exe", default=EXE)
ap.add_argument("--model-path", default=MODEL)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--ctx", type=int, required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--spec", choices=["dflash", "mtp", "none"], default="dflash")
ap.add_argument("--drafts", default="7")
ap.add_argument("--draft-ubatch", type=int, default=16)
ap.add_argument("--draft-ngl", type=int, default=99)
ap.add_argument("--draft-model-path", default="")
ap.add_argument("--ub", type=int, default=128)
ap.add_argument("--b", type=int, default=512)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--ctk", default="q4_0")
ap.add_argument("--ctv", default="q4_0")
ap.add_argument("--n-cpu-ffn", type=int, default=None)
ap.add_argument("--ot", action="append", default=[])
ap.add_argument("--nkvo", action="store_true")
ap.add_argument("--snapshot", default="")
ap.add_argument("--save", default="")
ap.add_argument("--n-predict-1", type=int, default=None)
ap.add_argument("--logprobs", type=int, default=0)
ap.add_argument("--extra", action="append", default=[])
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--port", type=int, default=11001)
ap.add_argument("--out", required=True)
ap.add_argument("--vram", action="store_true")
ap.add_argument("--verbosity", default="2")
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", a.port)) == 0:
        raise RuntimeError(f"port {a.port} in use")

env = dict(os.environ)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v

cmd = [a.exe, "-m", a.model_path, "-c", str(a.ctx), "-ngl", "99", "-fa", "on",
       "-ctk", a.ctk, "-ctv", a.ctv, "-b", str(a.b), "-ub", str(a.ub), "-t", str(a.threads),
       "-np", "1", "--fit", "off", "--no-context-shift", "--jinja", "--port", str(a.port),
       "--log-verbosity", a.verbosity, "--slot-save-path", SLOTS]
if a.n_cpu_ffn is not None:
    cmd += ["--n-cpu-ffn", str(a.n_cpu_ffn)]
if a.ot:
    cmd += ["-ot", ",".join(a.ot)]
if a.nkvo:
    cmd += ["-nkvo"]
if a.spec == "dflash":
    cmd += ["--spec-type", "draft-dflash", "--spec-draft-model", (a.draft_model_path or DRAFT),
            "--spec-draft-ngl", str(a.draft_ngl), "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
            "--spec-draft-n-max", str(max(int(x) for x in a.drafts.split(","))),
            "--spec-draft-n-ubatch", str(a.draft_ubatch)]
elif a.spec == "mtp":
    cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(max(int(x) for x in a.drafts.split(",")))]
cmd += a.extra

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


rec = {"spec_kind": a.spec, "args": vars(a), "cmd": cmd, "runs": {}}
try:
    up = False
    for _ in range(600):
        if p.poll() is not None:
            break
        if (get("/health") or {}).get("status") == "ok":
            up = True
            break
        time.sleep(1)
    if not up:
        tail = open(outname + ".server.log", encoding="utf-8", errors="replace").read()[-1200:]
        print("LOAD FAILED", outname + ".server.log")
        print(tail)
        sys.exit(2)
    vram_after_load = vram() if a.vram else None
    prompt = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="").read()

    if a.snapshot:
        r = post("/slots/0?action=restore", {"filename": a.snapshot})
        rec["restore"] = {"ok": "error" not in r, "raw": r if "error" in r else None}
        print("restored", a.snapshot, rec["restore"]["ok"])
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
        rec["save"] = {"name": a.save, "ok": not (sv or {}).get("error"), "error": (sv or {}).get("error")}
        print("saved", a.save, rec["save"]["ok"])

    t0 = time.time()
    body = {"prompt": prompt, "n_predict": a.n, "temperature": 0.0, "cache_prompt": True,
            "ignore_eos": True, "return_tokens": True}
    if a.logprobs:
        body["logprobs"] = a.logprobs
        body["top_logprobs"] = a.logprobs
    r = post("/completion", body)
    tt = r.get("timings", {})
    toks = r.get("tokens") or []
    if a.logprobs:
        rec["probs"] = r.get("completion_probabilities")
    rec["runs"]["run"] = {"timings": tt, "n_tokens": len(toks),
                          "sha": hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16],
                          "tokens": toks, "wall_s": time.time() - t0,
                          "vram_peak": vram() if a.vram else None}
    print(f"{a.spec}: {tt.get('predicted_n')} tok {tt.get('predicted_per_second', 0):.2f} t/s "
          f"drafted={tt.get('draft_n')} acc={tt.get('draft_n_accepted')} "
          f"sha={rec['runs']['run']['sha']}")
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
