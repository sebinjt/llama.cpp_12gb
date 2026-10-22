"""Decode-shaped, teacher-forced top-k comparison of two server configurations (attention-kernel A/B).

For one fixed token sequence T (greedy output of the reference config), every config is queried token by token:
prompt + T[:i] -> next-token top-k distribution (n_probs), with prompt caching so each step decodes 1-2 tokens.
Reports top-1 agreement and a KL divergence estimate on the top-k support (renormalized).

  python tests/dq4_kld.py --n 128 --out results/dq4/kld.json
"""
import argparse, json, math, os, socket, subprocess, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "..", "build-cuda", "bin", "llama-server.exe")
MODEL = "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
PORT = 9963
BASE = f"http://127.0.0.1:{PORT}"

ap = argparse.ArgumentParser()
ap.add_argument("--prompt-file", default=os.path.join("results", "dq4", "code_short.prompt.txt"))
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--configs", default="vec,dq4", help="first is the reference; vec, dq4, stock (no QUANT_KV: MMA for 2+ tokens)")
ap.add_argument("--out", default=os.path.join("results", "dq4", "kld.json"))
a = ap.parse_args()

# KV-quant configs (worker2 seq 114 Task B): "_ctk"/"_ctv" set the KV types, the rest is env. All of them run
# with LLAMA_FA_VEC_QUANT_KV unset (the MMA route), which is the configuration the 8K sweep showed is not penalised.
ap2 = {"vec": {}, "dq4": {"LLAMA_FA_DECODE_Q4": "1"}, "stock": {"LLAMA_FA_VEC_QUANT_KV": ""},
       "f16":  {"LLAMA_FA_VEC_QUANT_KV": "", "_ctk": "f16",  "_ctv": "f16"},
       "q4q4": {"LLAMA_FA_VEC_QUANT_KV": "", "_ctk": "q4_0", "_ctv": "q4_0"},
       "q8q4": {"LLAMA_FA_VEC_QUANT_KV": "", "_ctk": "q8_0", "_ctv": "q4_0"},
       "q8q8": {"LLAMA_FA_VEC_QUANT_KV": "", "_ctk": "q8_0", "_ctv": "q8_0"}}


def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def get(path):
    try:
        return json.loads(urllib.request.urlopen(BASE + path, timeout=5).read())
    except Exception:
        return None


def start(extra_env, tag):
    extra_env = dict(extra_env)
    ctk = extra_env.pop("_ctk", "q4_0")
    ctv = extra_env.pop("_ctv", "q4_0")
    env = dict(os.environ)
    env.update({"LLAMA_KV_STAGED_LAYERS": "0", "LLAMA_FA_VEC_QUANT_KV": "1", "LLAMA_FA_VEC_Q4_COLS": "2"})
    env.update(extra_env)
    for k in [k for k, v in env.items() if v == ""]:
        del env[k]  # an empty value means unset (the gates test getenv != nullptr)
    cmd = [EXE, "-m", MODEL, "-c", str(a.ctx), "-ngl", "99", "-fa", "on", "-ctk", ctk, "-ctv", ctv,
           "-b", "512", "-ub", "128", "-np", "1", "--fit", "off", "--no-context-shift", "--port", str(PORT),
           "--log-verbosity", "2"]
    lf = open(os.path.join(ROOT, "results", "dq4", f"server_kld_{tag}.log"), "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(300):
        if p.poll() is not None:
            break
        if (get("/health") or {}).get("status") == "ok":
            return p, lf
        time.sleep(1)
    p.kill()
    raise RuntimeError(f"{tag}: server failed to load")


def stop(p, lf):
    try:
        p.terminate()
        p.wait(timeout=30)
    except Exception:
        subprocess.run(["taskkill", "/F", "/PID", str(p.pid)], capture_output=True)
    lf.close()


def topk(r):
    cp = r["completion_probabilities"][0]
    lst = cp.get("top_logprobs") or cp.get("top_probs")
    out = {}
    for e in lst:
        out[e["id"]] = math.exp(e["logprob"]) if "logprob" in e else e["prob"]
    return out


with socket.socket() as probe:
    if probe.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError(f"port {PORT} in use")
with open(os.path.join(ROOT, a.prompt_file), encoding="utf-8", newline="") as pf:
    prompt_text = pf.read()

CONFIGS = [(c, ap2[c]) for c in a.configs.split(",")]
res = {"n": a.n, "k": a.k, "configs": {}}
T = None
P = None
for tag, extra in CONFIGS:
    p, lf = start(extra, tag)
    try:
        if P is None:
            P = post("/tokenize", {"content": prompt_text, "add_special": True})["tokens"]
        if T is None:
            r = post("/completion", {"prompt": P, "n_predict": a.n, "temperature": 0.0, "cache_prompt": True,
                                     "ignore_eos": True, "return_tokens": True})
            T = r["tokens"]
            res["T"] = T
        dists = []
        t0 = time.time()
        for i in range(a.n):
            r = post("/completion", {"prompt": P + T[:i], "n_predict": 1, "temperature": 0.0, "cache_prompt": True,
                                     "n_probs": a.k, "ignore_eos": True})
            dists.append(topk(r))
        res["configs"][tag] = [{str(k): v for k, v in d.items()} for d in dists]
        print(f"{tag}: {a.n} steps in {time.time() - t0:.1f} s", flush=True)
    finally:
        stop(p, lf)

reftag = CONFIGS[0][0]
ref = res["configs"][reftag]
for tag in res["configs"]:
    if tag == reftag:
        continue
    cur = res["configs"][tag]
    agree = 0
    klds = []
    maxdp = 0.0
    for d0, d1 in zip(ref, cur):
        a0 = max(d0, key=d0.get)
        a1 = max(d1, key=d1.get)
        agree += a0 == a1
        keys = set(d0) | set(d1)
        eps = 1e-10
        s0 = sum(d0.get(k, eps) for k in keys)
        s1 = sum(d1.get(k, eps) for k in keys)
        klds.append(sum((d0.get(k, eps) / s0) * math.log((d0.get(k, eps) / s0) / (d1.get(k, eps) / s1)) for k in keys))
        maxdp = max(maxdp, max(abs(d0.get(k, 0) - d1.get(k, 0)) for k in keys))
    print(f"{tag} vs {reftag}: top-1 agreement {agree}/{len(ref)}, KLD(top-{a.k} support) mean {sum(klds)/len(klds):.3e} "
          f"max {max(klds):.3e}, max |dp| {maxdp:.3e}; ref greedy matches T: "
          f"{sum(1 for i, d in enumerate(ref) if max(d, key=d.get) == str(T[i]))}/{len(ref)}")
    res[f"summary_{tag}"] = {"top1": agree, "kld_mean": sum(klds) / len(klds), "kld_max": max(klds), "max_dp": maxdp}
with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as f:
    json.dump(res, f)
