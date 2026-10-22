"""Profile one decode request (restore or prefill first), optional saved slot, nsys capture + sqlite export.

  python tests/prof_t1.py --prompt-file results/dq4/code_short.prompt.txt --snapshot dq4-short-q4-mtp.bin \
      --ctx 16384 --spec draft-mtp --draft-max 4 --n 128 --out results/prof/t1_mtp8k --port 9971 \
      --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_FA_VEC_MAX_NQ=128 --env LLAMA_RS_REPLAY=1 \
      --env LLAMA_MTP_WINDOW=8192 --env LLAMA_MMVQ_BATCH_INVARIANT=1

nsys is optional (--no-nsys). When enabled: nsys launch the server, nsys start just before the decode
request, nsys stop after it, then export the report to <out>.sqlite for kernel/timeline queries.
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
DRAFT = "Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf"
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--snapshot", default="", help="restore this slot instead of prefilling")
ap.add_argument("--save", default="", help="save the slot to this name after the decode")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--staged", type=int, default=0)
ap.add_argument("--budget", type=int, default=None, help="LLAMA_KV_GPU_BUDGET_MIB")
ap.add_argument("--spec", default="")
ap.add_argument("--draft-max", type=int, default=4)
ap.add_argument("--draft-ubatch", type=int, default=16)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--port", type=int, default=9971)
ap.add_argument("--output", default="CUDA0", choices=["CPU", "CUDA0"])
ap.add_argument("--b", type=int, default=512)
ap.add_argument("--ub", type=int, default=128)
ap.add_argument("--ot", action="append", default=[], help="extra -ot tensor override")
ap.add_argument("--samplers", default="")
ap.add_argument("--backend-sampling", action="store_true")
ap.add_argument("--no-nsys", action="store_true")
ap.add_argument("--no-tokens", action="store_true", help="do not return token ids")
ap.add_argument("--verbosity", default="2")
ap.add_argument("--timeout", type=int, default=7200)
a = ap.parse_args()
BASE = f"http://127.0.0.1:{a.port}"

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", a.port)) == 0:
        raise RuntimeError(f"port {a.port} already in use")

env = dict(os.environ)
env.update({"LLAMA_KV_STAGED_LAYERS": str(a.staged), "LLAMA_KV_STAGE_SLOTS": "2",
            "LLAMA_FA_VEC_QUANT_KV": "1", "LLAMA_FA_VEC_Q4_COLS": "2"})
if a.budget is not None:
    env["LLAMA_KV_GPU_BUDGET_MIB"] = str(a.budget)
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v

cmd = [EXE, "-m", a.model, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
       "-b", str(a.b), "-ub", str(a.ub), "-np", "1", "--fit", "off", "--no-context-shift", "--jinja",
       "--port", str(a.port), "--log-verbosity", str(a.verbosity), "--slot-save-path", SLOTS,
       "-ot", f"^output\\.weight$={a.output}"]
for ot in a.ot:
    cmd += ["-ot", ot]
if a.spec:
    cmd += ["--spec-type", a.spec, "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0",
            "--spec-draft-n-max", str(a.draft_max)]
    if a.draft_ubatch:
        cmd += ["--spec-draft-n-ubatch", str(a.draft_ubatch)]
    if "draft-dflash" in a.spec.split(","):
        cmd += ["--spec-draft-model", DRAFT, "--spec-draft-ngl", "99"]
if a.samplers:
    cmd += ["--samplers", a.samplers]
if a.backend_sampling:
    cmd += ["--backend-sampling"]

session = f"t1_{os.path.basename(a.out)}_{os.getpid()}"
nsys_rep = os.path.join(ROOT, a.out + ".nsys-rep")
nsys_db = os.path.join(ROOT, a.out + ".sqlite")
if not a.no_nsys:
    cmd = [NSYS, "launch", f"--session-new={session}", "-t", "cuda,nvtx",
           "--cuda-graph-trace=node"] + cmd

os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
lf = open(os.path.join(ROOT, a.out + ".server.log"), "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def post(path, body, timeout=None):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout or a.timeout).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:300]}
    except Exception as e:
        return {"error": repr(e)[:300]}


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


rec = {"tag": os.path.basename(a.out), "args": vars(a), "nsys": not a.no_nsys}
try:
    up = False
    for _ in range(400):
        if p.poll() is not None:
            break
        if (get("/health") or {}).get("status") == "ok":
            up = True
            break
        time.sleep(1)
    if not up:
        print(f"LOAD FAILED (see {a.out}.server.log)")
        sys.exit(2)

    prompt = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="").read()
    if a.snapshot:
        r = post("/slots/0?action=restore", {"filename": a.snapshot})
        if "error" in r:
            print("restore failed", r)
            sys.exit(4)
        rec["restore"] = r
    else:
        r = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0.0,
                                 "cache_prompt": True, "ignore_eos": True})
        if "error" in r:
            print("prefill failed", r)
            sys.exit(4)
        tt = r.get("timings", {})
        rec["prefill"] = {"prompt_n": tt.get("prompt_n"), "prompt_ms": tt.get("prompt_ms")}
        print(f"prefill {tt.get('prompt_n')} tok in {(tt.get('prompt_ms') or 0)/1e3:.1f} s")

    if not a.no_nsys:
        subprocess.run([NSYS, "start", f"--session={session}", "-o", os.path.join(ROOT, a.out),
                        "--force-overwrite=true", "--sample=none"], check=True)
    body = {"prompt": prompt, "n_predict": a.n, "temperature": 0.0, "cache_prompt": True,
            "ignore_eos": True}
    if not a.no_tokens:
        body["return_tokens"] = True
    t0 = time.time()
    r = post("/completion", body)
    wall = time.time() - t0
    if not a.no_nsys:
        subprocess.run([NSYS, "stop", f"--session={session}"], check=True)

    if "error" in r:
        print("decode failed", r)
        sys.exit(5)
    t = r.get("timings", {})
    toks = r.get("tokens")
    rec.update({"wall_s": wall, "timings": t, "predicted_n": t.get("predicted_n"),
                "tg_tps": t.get("predicted_per_second"), "draft_n": t.get("draft_n"),
                "draft_n_accepted": t.get("draft_n_accepted"),
                "n_tokens": len(toks) if toks else None,
                "tokens_sha256_prefix": None})
    if toks:
        import hashlib
        rec["tokens_sha256_prefix"] = hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16]
        rec["tokens"] = toks

    if a.save:
        sv = post("/slots/0?action=save", {"filename": a.save})
        rec["save"] = {"name": a.save, "ok": not (sv or {}).get("error"), "error": (sv or {}).get("error")}

    json.dump(rec, open(os.path.join(ROOT, a.out + ".json"), "w"), indent=1)
    print(f"decode {t.get('predicted_n')} tok {t.get('predicted_per_second', 0):.2f} t/s "
          f"wall={wall:.3f}s drafted={t.get('draft_n')} accepted={t.get('draft_n_accepted')} "
          f"tokens={len(toks) if toks else '-'} sha={rec['tokens_sha256_prefix']}")

    if not a.no_nsys:
        for cmd_ in ([NSYS, "export", "--type", "sqlite", "--force-overwrite", "true",
                      "-o", nsys_db, nsys_rep],
                     [NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                      "-o", os.path.join(ROOT, a.out + ".kern.csv"), nsys_rep]):
            try:
                subprocess.run(cmd_, check=True, timeout=600)
            except Exception as e:
                print("post-process failed:", e)
    sys.exit(0)
finally:
    try:
        p.terminate(); p.wait(timeout=60)
    except Exception:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    lf.close()
