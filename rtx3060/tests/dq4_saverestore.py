"""Slot save/restore in the middle of MTP decoding (after a verify step) must not change tokens.

Per mode: uninterrupted greedy run of 2N tokens, vs N tokens -> slot save -> erase -> restore -> N more tokens.
  python tests/dq4_saverestore.py --env LLAMA_RS_REPLAY=1
"""
import argparse, json, os, socket, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
SLOTS = os.path.join(ROOT, "results", ".kv_cache") + os.sep
PORT = 9964
BASE = f"http://127.0.0.1:{PORT}"

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", default=os.path.join("results", "dq4", "code_short.prompt.txt"))
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--draft-max", type=int, default=3)
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--tag", default="sr")
a = ap.parse_args()


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError("port in use")
env = dict(os.environ)
env.update({"LLAMA_KV_STAGED_LAYERS": "0", "LLAMA_FA_VEC_QUANT_KV": "1", "LLAMA_FA_VEC_Q4_COLS": "2", "LLAMA_FA_DECODE_Q4": "1"})
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
cmd = [EXE, "-m", MODEL, "-c", "16384", "-ngl", "99", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "-b", "512", "-ub", "128",
       "-np", "1", "--fit", "off", "--no-context-shift", "--port", str(PORT), "--log-verbosity", "2", "--slot-save-path", SLOTS,
       "--spec-type", "draft-mtp", "--spec-draft-type-k", "q4_0", "--spec-draft-type-v", "q4_0", "--spec-draft-n-max", str(a.draft_max)]
lf = open(os.path.join(ROOT, "results", "dq4", f"server_{a.tag}.log"), "w", encoding="utf-8", errors="replace")
p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
try:
    for _ in range(300):
        if (get("/health") or {}).get("status") == "ok":
            break
        time.sleep(1)
    text = open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="").read()
    P = post("/tokenize", {"content": text, "add_special": True})["tokens"]
    req = lambda prompt, n, cache: post("/completion", {"prompt": prompt, "n_predict": n, "temperature": 0.0, "cache_prompt": cache,
                                                         "ignore_eos": True, "return_tokens": True})
    post("/slots/0?action=erase", {})
    full = req(P, 2 * a.n, False)
    post("/slots/0?action=erase", {})
    r1 = req(P, a.n, True)
    t1 = r1["tokens"]
    snap = f"dq4-saverestore.bin"
    sv = post("/slots/0?action=save", {"filename": snap})
    post("/slots/0?action=erase", {})
    rs = post("/slots/0?action=restore", {"filename": snap})
    r2 = req(P + t1, a.n, True)
    t2 = r2["tokens"]
    ft = full["tokens"]
    same = ft == t1 + t2
    d = next((i for i in range(min(len(ft), len(t1 + t2))) if ft[i] != (t1 + t2)[i]), None)
    tm1, tm2 = r1.get("timings", {}), r2.get("timings", {})
    print(f"[{a.tag}] uninterrupted {len(ft)} vs {len(t1)}+{len(t2)} after save/restore: {'IDENTICAL' if same else f'first diff at {d}'}; "
          f"part1 drafted={tm1.get('draft_n')} accepted={tm1.get('draft_n_accepted')}; restore cache_n={tm2.get('cache_n')} "
          f"prompt_n={tm2.get('prompt_n')}; saved {sv.get('n_saved')} tokens, restored {rs.get('n_restored')}")
finally:
    p.terminate()
    try:
        p.wait(timeout=30)
    except Exception:
        subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)
    lf.close()
