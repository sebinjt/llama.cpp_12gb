"""End-to-end decode A/B on a restored 200K snapshot (staged KV), for the LLAMA_FA_DECODE_Q4 kernel.

One server per configuration; each rep restores the snapshot and decodes --n greedy tokens.
Prints tok/s per rep, the token ids' hash, and writes the token ids to --out (json) for comparison.

  python tests/dq4_e2e.py --staged 9 --env LLAMA_FA_DECODE_Q4=1 --tag dq4-L9 --out results/dq4/dq4-L9.json
"""
import argparse, hashlib, json, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep
PORT = 9962
BASE = f"http://127.0.0.1:{PORT}"

ap = argparse.ArgumentParser()
ap.add_argument("--staged", type=int, default=9)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--ctx", type=int, default=204800)
ap.add_argument("--snapshot", default="kv-200k-code-u2048-q4.bin")
ap.add_argument("--prompt-file", default=os.path.join("results", "code200k.prompt.txt"))
ap.add_argument("--env", action="append", default=[], help="KEY=VALUE added to the server environment")
ap.add_argument("--spec", default="", help="e.g. draft-mtp")
ap.add_argument("--draft-max", type=int, default=0)
ap.add_argument("--draft-ubatch", type=int, default=0)
ap.add_argument("--server-arg", action="append", default=[], help="extra llama-server argument (repeatable)")
ap.add_argument("--output", default="CUDA0", choices=["CPU", "CUDA0"])
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default="")
ap.add_argument("--verbosity", default="2")
ap.add_argument("--suffix-file", default="", help="text appended after the snapshot prompt (prefilled on top of the restore)")
ap.add_argument("--stop-at-eos", action="store_true")
ap.add_argument("--n-probs", type=int, default=0, help="return top-k probabilities per token (CPU cost per token)")
ap.add_argument("--save", default="", help="save the slot to this snapshot name after the last rep")
ap.add_argument("--skip-manifest", action="store_true", help="snapshot has no manifest (made by --save)")
ap.add_argument("--no-restore", action="store_true", help="fresh prefill instead of restoring the snapshot")
ap.add_argument("--load-only", action="store_true", help="only check that the server loads")
a = ap.parse_args()

with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError(f"port {PORT} in use")

with open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="") as pf:
    prompt = pf.read()
manifest = {} if (a.no_restore or a.skip_manifest) else json.load(open(os.path.join(SLOTS, a.snapshot + ".json"), encoding="utf-8"))
if not a.no_restore and not a.skip_manifest and manifest.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
    print("snapshot prompt mismatch"); sys.exit(4)
if a.suffix_file:
    with open(os.path.join(ROOT, a.suffix_file), encoding="utf-8", newline="") as sf:
        prompt += sf.read()
if not a.no_restore and not a.skip_manifest and "draft-mtp" in a.spec.split(",") and manifest.get("kind") != "target+mtp":
    print("draft-mtp needs a target+mtp snapshot"); sys.exit(4)

env = dict(os.environ)
env["LLAMA_KV_STAGED_LAYERS"] = str(a.staged)
env["LLAMA_KV_STAGE_SLOTS"] = "2"
env["LLAMA_FA_VEC_QUANT_KV"] = "1"
env["LLAMA_FA_VEC_Q4_COLS"] = "2"
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v

cmd = [EXE, "-m", MODEL, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0",
       "-b", "512", "-ub", "128", "-np", "1", "--fit", "off", "--no-context-shift", "--jinja", "--port", str(PORT),
       "--log-verbosity", a.verbosity, "--slot-save-path", SLOTS, "-ot", f"^output\\.weight$={a.output}"]
if a.spec:
    cmd += ["--spec-type", a.spec, "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0"]
    if a.draft_max:
        cmd += ["--spec-draft-n-max", str(a.draft_max)]
    if a.draft_ubatch:
        cmd += ["--spec-draft-n-ubatch", str(a.draft_ubatch)]

cmd += a.server_arg
os.makedirs(os.path.join(ROOT, "results", "dq4"), exist_ok=True)
log = os.path.join(ROOT, "results", "dq4", f"server_{a.tag}.log")
lf = open(log, "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


def post(path, body=None, timeout=7200):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(BASE + path, data, {"Content-Type": "application/json"}, method="POST")
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace")[:300]}
    except Exception as e:
        return {"error": repr(e)[:300]}


res = {"tag": a.tag, "staged": a.staged, "env": a.env, "spec": a.spec, "draft_max": a.draft_max, "reps": []}
code = 0
try:
    up = False
    for _ in range(300):
        if p.poll() is not None:
            break
        if (get("/health") or {}).get("status") == "ok":
            up = True
            break
        time.sleep(1)
    if not up:
        print(f"[{a.tag}] LOAD FAILED (see {log})")
        code = 2
        sys.exit(2)
    if a.load_only:
        print(f"[{a.tag}] loaded")
        sys.exit(0)
    for rep in range(a.reps):
        r = {} if a.no_restore else post("/slots/0?action=restore", {"filename": a.snapshot})
        if a.no_restore:
            post("/slots/0?action=erase")
        if r is None or "error" in r:
            print(f"[{a.tag}] restore failed: {(r or {}).get('error')}")
            code = 4
            sys.exit(4)
        r = post("/completion", {"prompt": prompt, "n_predict": a.n, "temperature": 0.0, "cache_prompt": not a.no_restore,
                                 "ignore_eos": not a.stop_at_eos, "return_tokens": True, "n_probs": a.n_probs})
        if "error" in r:
            print(f"[{a.tag}] decode error: {r['error'][:200]}")
            code = 3
            sys.exit(3)
        t = r.get("timings", {})
        toks = r.get("tokens") or []
        n = t.get("predicted_n", 0) or 0
        rec = {"tok_s": t.get("predicted_per_second"), "ms_per_tok": (t.get("predicted_ms") or 0) / max(1, n),
               "predicted_n": n, "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "tokens": toks, "tokens_sha": hashlib.sha256(json.dumps(toks[:64]).encode()).hexdigest()[:16],
               "text_head": (r.get("content") or "")[:120], "content": r.get("content") or "", "probs": r.get("completion_probabilities") or []}
        res["reps"].append(rec)
        acc = ""
        if rec["draft_n"]:
            acc = f" drafted={rec['draft_n']} accepted={rec['draft_n_accepted']} acc={rec['draft_n_accepted']/rec['draft_n']:.3f}"
        print(f"[{a.tag}] rep{rep} cache_n={rec['cache_n']} prompt_n={rec['prompt_n']} PP {(rec['prompt_ms'] or 0)/1e3:.1f}s TG {n} tok "
              f"{rec['tok_s'] or 0:.3f} t/s ({rec['ms_per_tok']:.1f} ms/tok) first64_sha={rec['tokens_sha']}{acc}", flush=True)
    if a.save:
        sv = post("/slots/0?action=save", {"filename": a.save})
        print(f"[{a.tag}] saved slot -> {a.save}: {sv}")
finally:
    try:
        p.terminate()
        p.wait(timeout=30)
    except Exception:
        subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)
    lf.close()
    if a.out and res["reps"]:
        with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
