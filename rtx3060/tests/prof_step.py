"""Profile decode steps with Nsight Systems: llama-server under `nsys launch`, prefill (or restore) first, then capture only
the decode request (`nsys start` ... `nsys stop`), export to sqlite.

  python tests/prof_step.py --prompt-file results/charts/prompts/t8k.txt --ctx 16384 --spec draft-mtp --draft-max 4 \
      --n 64 --out results/prof/mtp8k --env LLAMA_FA_DECODE_Q4=1 ...
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe"
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep
PORT = 9965
BASE = f"http://127.0.0.1:{PORT}"

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--snapshot", default="", help="restore instead of prefilling")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--staged", type=int, default=0)
ap.add_argument("--spec", default="")
ap.add_argument("--draft-max", type=int, default=4)
ap.add_argument("--draft-ubatch", type=int, default=16)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--no-nsys", action="store_true")
a = ap.parse_args()

env = dict(os.environ)
env.update({"LLAMA_KV_STAGED_LAYERS": str(a.staged), "LLAMA_KV_STAGE_SLOTS": "2", "LLAMA_FA_VEC_QUANT_KV": "1",
            "LLAMA_FA_VEC_Q4_COLS": "2"})
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = [EXE, "-m", MODEL, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "128",
       "-np", "1", "--fit", "off", "--no-context-shift", "--jinja", "--port", str(PORT), "--log-verbosity", "2",
       "--slot-save-path", SLOTS, "-ot", "^output\\.weight$=CUDA0"]
if a.spec:
    cmd += ["--spec-type", a.spec, "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0", "--spec-draft-n-max", str(a.draft_max),
            "--spec-draft-n-ubatch", str(a.draft_ubatch)]
session = "dq4prof"
if not a.no_nsys:
    cmd = [NSYS, "launch", f"--session-new={session}", "-t", "cuda", "--cuda-graph-trace=node"] + cmd
os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
lf = open(os.path.join(ROOT, a.out + ".server.log"), "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def post(path, body, timeout=7200):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:300]}


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


try:
    for _ in range(400):
        if (get("/health") or {}).get("status") == "ok":
            break
        time.sleep(1)
    else:
        print("LOAD FAILED"); sys.exit(2)
    prompt = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="").read()
    if a.snapshot:
        r = post("/slots/0?action=restore", {"filename": a.snapshot})
        if "error" in r:
            print("restore failed", r); sys.exit(4)
    else:
        r = post("/completion", {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "cache_prompt": True})
        print("prefill:", r.get("timings", {}).get("prompt_n"), "tokens", round((r.get("timings", {}).get("prompt_ms") or 0) / 1e3, 1), "s")
    if not a.no_nsys:
        subprocess.run([NSYS, "start", f"--session={session}", "-o", os.path.join(ROOT, a.out), "--force-overwrite=true"], check=True)
    t0 = time.time()
    r = post("/completion", {"prompt": prompt, "n_predict": a.n, "temperature": 0.0, "cache_prompt": True, "ignore_eos": True})
    wall = time.time() - t0
    if not a.no_nsys:
        subprocess.run([NSYS, "stop", f"--session={session}"], check=True)
    t = r.get("timings", {})
    info = {"wall_s": wall, "timings": t, "args": vars(a)}
    json.dump(info, open(os.path.join(ROOT, a.out + ".json"), "w"), indent=1)
    print(f"decode {t.get('predicted_n')} tok {t.get('predicted_per_second', 0):.2f} t/s prompt_n={t.get('prompt_n')} "
          f"drafted={t.get('draft_n')} accepted={t.get('draft_n_accepted')} wall={wall:.2f}s")
finally:
    try:
        p.terminate(); p.wait(timeout=60)
    except Exception:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    lf.close()
