"""Real-world checks of the production configs through llama-server, as a user would run it.

OpenAI-compatible client (/v1/chat/completions, streaming). Every request records the server's own
timings (prompt_n / prompt_ms / predicted_ms / predicted_per_second / draft_n / draft_n_accepted /
cache_n) plus a client-side TTFT wall clock, the finish reason, and GPU VRAM (sampled in the background).

Scenarios (worker2 seq 109, reviewer seq 107):
  1  dynamic-config check: 8 varied ~4K-context prompts, max_tokens 512, temperature 0. The same run is
     repeated against the same profile started at two contexts (plong at -c 32768 and -c 204800) and the
     two tables are compared: the goal is the same speed at ~4K either way.
  2  growing multi-turn conversation built from real repo text, ~2-8K new tokens per turn up to ~199K,
     on plong. Per turn: cached tokens (only the new turn should prefill), new-turn latency, decode t/s,
     VRAM.
  3  exactness through the server: 2 prompts at temperature 0 against a spec server and a --no-spec
     server (same profile); the generated text must be identical.
  4  sampling: temperature 0.7 / top_p 0.9, 5 requests per profile; no crash, coherent text, acceptance
     reported.
  5  stability summary for a scenario-4/1/2 run: request count, peak VRAM, server warnings/errors.

The GPU lock and the server launch are the caller's job (comm/GPU_LOCK_RULES.md, tests/serve_prod.ps1).

  python tests/realworld_chat.py --scenario 1 --base-url http://127.0.0.1:8091 --out results/rw/s1_32k.json
  python tests/realworld_chat.py --scenario 3 --base-url A --base-url2 B --out results/rw/s3.json
  python tests/realworld_chat.py --scenario 2 --base-url A --turns 30 --out results/rw/s2.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------- plumbing
class VramSampler(threading.Thread):
    """Peak VRAM (nvidia-smi, MiB) while a scenario runs."""

    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.peak, self.samples, self._stop = period, 0, 0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout.splitlines()
                v = int(out[0].strip())
                self.peak = max(self.peak, v)
                self.samples += 1
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        return self.peak


def vram_now():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10).stdout.splitlines()[0])
    except Exception:
        return None


def chat(base, messages, max_tokens, temperature=0.0, top_p=None, seed=None, timeout=7200, label=""):
    """One streaming chat completion. Returns text + server timings + client-side timings."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "cache_prompt": True}
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.time()
    ttft = None
    parts, reasoning, timings, usage, finish, n_chunks, raw_tail = [], [], None, None, None, 0, None
    first_kind = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                n_chunks += 1
                raw_tail = j
                if j.get("timings"):
                    timings = j["timings"]
                if j.get("usage"):
                    usage = j["usage"]
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    # this build streams the thinking part as reasoning_content and the answer as content
                    for kind, bucket in (("reasoning_content", reasoning), ("content", parts)):
                        piece = d.get(kind)
                        if piece:
                            if ttft is None:
                                ttft = time.time() - t0
                                first_kind = kind
                            bucket.append(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}"
    except Exception as e:
        err = repr(e)[:400]
    wall = time.time() - t0
    text = "".join(parts)
    think = "".join(reasoning)
    t = timings or {}
    return {
        "label": label, "error": err, "text": text, "reasoning": think, "wall_s": round(wall, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None, "first_delta": first_kind,
        "answer_missing": (not text), "finish_reason": finish,
        "prompt_tokens": (usage or {}).get("prompt_tokens", t.get("prompt_n")),
        "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", t.get("cache_n")),
        "completion_tokens": (usage or {}).get("completion_tokens", t.get("predicted_n")),
        "prompt_ms": t.get("prompt_ms"), "prompt_tps": t.get("prompt_per_second"),
        "predicted_ms": t.get("predicted_ms"), "decode_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
        "verify_steps": t.get("draft_verification_steps"),
        "accepted_per_step": (round(t["predicted_n"] / t["draft_verification_steps"], 3)
                              if t.get("draft_verification_steps") else None),
        "timings_raw": t or None, "last_chunk": raw_tail, "n_chunks": n_chunks,
    }


def tok_est(s):
    return len(s) // 4


def read(rel, start=None, end=None):
    p = os.path.join(ROOT, rel)
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[(start or 1) - 1:(end or len(lines))])


def clip(text, chars):
    """~chars characters of real repo text (no synthetic padding)."""
    return text[:chars]


# ----------------------------------------------------------------------------- scenario 1
def scenario1(base, max_tokens, chars):
    """8 varied ~4K-context prompts; one table row each."""
    kv = read("../src/llama-kv-cache.cpp")
    gpu = read("../ggml/src/ggml-cuda/ggml-cuda.cu")
    rep55 = read("report.md", 3235, 3400)
    log = read("results/w2/s106_c64dflash7_b300_S612.json.server.log")
    drv = read("tests/w2_dflash.py")
    rd = read("report.md", 1783, 2102)  # section 26

    prompts = [
        ("code-review",
         "You are reviewing a C++ patch. Below is a CUDA/llama.cpp source excerpt. List the concrete bugs or "
         "race conditions you can find, most severe first, each with the line-level reason.\n\n```cpp\n"
         + clip(gpu, chars) + "\n```"),
        ("math-reasoning",
         "Below are measured benchmark rows. Work through the arithmetic step by step and say whether the "
         "stated percentages follow from the raw numbers; show each division.\n\n" + clip(rep55, chars)),
        ("summarise",
         "Summarise the following report section for an engineer who has not read it: what was measured, what "
         "the numbers were, and what the conclusion is. Use at most 200 words.\n\n" + clip(rep55, chars)),
        ("translate",
         "Translate the following technical text into German, keeping all numbers, identifiers and section "
         "references exactly as they are.\n\n" + clip(rep55, chars)),
        ("json-extract",
         "Extract every STEPTIME line from this server log into a JSON array. Each element must have exactly "
         "the keys draft, pre, eval, sync, post, total, ops with numeric values (floats, ms). Output only the "
         "JSON array.\n\n" + clip(log, chars)),
        ("long-form",
         "Write a detailed design note for streaming quantized model weights into GPU slots with two "
         "double-buffered slots, covering slot pairing, prefetch, the pass boundary, and the failure modes. "
         "Be concrete and specific.\n\nContext from an existing implementation:\n" + clip(kv, chars)),
        ("code-fix",
         "The python script below drives a llama-server benchmark. Point out every failure mode you would "
         "fix before anyone runs it unattended (timeouts, cleanup, port clashes, silent errors), and give the "
         "minimal diff for each.\n\n```python\n" + clip(drv, chars) + "\n```"),
        ("number-check",
         "Below is a performance table. Recompute every ratio and flag any cell where the stated t/s does not "
         "equal tokens/step divided by ms/step (within 0.2%), showing your arithmetic.\n\n" + clip(rd, chars)),
    ]
    rows = []
    for name, prompt in prompts:
        r = chat(base, [{"role": "user", "content": prompt}], max_tokens, 0.0, label=name)
        r["prompt_chars"] = len(prompt)
        rows.append(r)
        print(f"  {name:<14} prompt={_f(r['prompt_tokens'], '>6')} tok  pp={_f(round(r['prompt_tps'], 1) if r['prompt_tps'] else None, '>7.1f')} t/s  "
              f"dec={_f(round(r['decode_tps'], 2) if r['decode_tps'] else None, '>6.2f')} t/s  ttft={r['ttft_s']}  "
              f"acc/step={r['accepted_per_step']}  "
              f"{'ERR ' + str(r['error']) if r['error'] else r['finish_reason']}")
    return rows


# ----------------------------------------------------------------------------- scenario 2
def _f(v, spec="d"):
    return format(v, spec) if v is not None else "-"


def scenario2(base, max_tokens, target_tokens, turn_tokens, out=None, n_ctx=204800):
    """Growing multi-turn conversation: each turn appends 2-8K tokens of real repo text plus a question."""
    # content pool: fixed ~36K-char windows walked over the biggest real files in the repo, so every
    # turn's new text is genuinely new (no invented filler).
    big = ["report.md",
           "../ggml/src/ggml-cuda/ggml-cuda.cu",
           "../src/llama-kv-cache.cpp",
           "comm/files/seq92_worker2.md", "comm/files/seq88_worker2.md",
           "improvement_directions.md", "staged-kv-root-cause.md", "nimble_notes.md",
           "tests/int8_fwht_bench.cu", "tests/w2_dflash.py", "dev/staged-kv-implementation.md",
           "../src/llama-context.cpp",
           "../src/llama-graph.cpp"]
    win = turn_tokens * 4
    pool = []
    for rel in big:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for off in range(0, len(text) - win // 2, win):
            pool.append(text[off:off + win])
    questions = [
        "Summarise this excerpt in 3 bullet points.",
        "What does this excerpt claim, and which numbers support it?",
        "Point out anything in this excerpt that looks inconsistent with the rest of the document.",
        "Explain this excerpt to a new engineer in plain terms.",
    ]
    messages = [{"role": "system", "content": "You are a terse, precise assistant. Answer in at most 150 words."}]
    rows, used, turn = [], 0, 0
    stop_reason = "pool exhausted"
    while used < target_tokens and turn < len(pool):
        if used + turn_tokens + max_tokens + 64 > n_ctx:
            stop_reason = f"next turn would exceed the {n_ctx}-token context"
            break
        chunk = pool[turn]
        want_chars = turn_tokens * 4
        if len(chunk) > want_chars:
            chunk = chunk[:want_chars]
        used += tok_est(chunk)
        messages.append({"role": "user",
                         "content": f"CONTEXT (turn {turn + 1}):\n{chunk}\n\n{questions[turn % len(questions)]}"})
        r = chat(base, messages, max_tokens, 0.0, label=f"turn{turn + 1}")
        r["turn"] = turn + 1
        r["approx_ctx_tokens"] = used
        r["vram_after"] = vram_now()
        rows.append(r)
        # keep the assistant turn in the history: the conversation really grows
        messages.append({"role": "assistant", "content": r["text"] or "(empty)"})
        print(f"  turn {turn + 1:>2}: ctx~{used:>7} prompt={_f(r['prompt_tokens'], '>6')} "
              f"cached={_f(r['cached_tokens'], '>6')} new={_f(round((r['prompt_ms'] or 0) / 1000, 1), '>5.1f')}s "
              f"ttft={r['ttft_s']} dec={_f(round(r['decode_tps'], 2) if r['decode_tps'] else None, '>5.2f')} t/s "
              f"vram={r['vram_after']} {'ERR ' + str(r['error']) if r['error'] else r['finish_reason']}")
        turn += 1
        if out:                      # incremental, so a crash late in a 25-minute run keeps the data
            json.dump({"turns": rows, "stop_reason": stop_reason}, open(out + ".partial.json", "w"), indent=1)
        if r["error"]:
            stop_reason = "request error"
            break
    # deliberate one-past-the-limit request: the server must reject it cleanly, not crash or shift context
    over = chat(base, messages + [{"role": "user", "content": pool[-1][:turn_tokens * 4]}], max_tokens, 0.0,
                label="over_limit")
    print(f"  over-limit probe: {'ERR ' + str(over['error'])[:120] if over['error'] else 'accepted (finish=' + str(over['finish_reason']) + ')'}")
    rows.append({"over_limit_probe": True, "error": over["error"], "prompt_tokens": over["prompt_tokens"],
                 "finish_reason": over["finish_reason"], "text_len": len(over.get("text") or "")})
    if out:
        json.dump({"turns": rows, "stop_reason": stop_reason}, open(out + ".partial.json", "w"), indent=1)
    print(f"  stop reason: {stop_reason}")
    return rows


# ----------------------------------------------------------------------------- scenario 3
def s3_prompts():
    p1 = ("List the exact steps, in order, that a llama-server process takes when it loads a Q4_0 KV cache "
          "and creates two staging slots. Be specific and complete.")
    p2 = read("report.md", 3235, 3300) + "\n\nIn one paragraph, what is the single most important number in this table?"
    return [p1, p2]


def scenario3(base, max_tokens, phase):
    """One side of the spec vs no-spec A/B through the server (temperature 0). Sequential by design: the
    two servers do not fit on the card together, so the texts are recorded and compared afterwards."""
    rows = []
    for i, prompt in enumerate(s3_prompts(), 1):
        r = chat(base, [{"role": "user", "content": prompt}], max_tokens, 0.0, label=f"{phase}_p{i}")
        rows.append({"prompt": i, "phase": phase, "run": r})
        print(f"  {phase} prompt {i}: {r['decode_tps'] or 0:.2f} t/s ({r['completion_tokens']} tok) "
              f"{'ERR ' + r['error'] if r['error'] else r['finish_reason']}")
    return rows


def s3_compare(files):
    """Compare the two recorded sides; identical text is the pass condition."""
    sides = {}
    for f in files:
        full = f if os.path.isabs(f) else os.path.join(ROOT, f)
        d = json.load(open(full))
        for row in d["runs"][0]:
            sides.setdefault(row["prompt"], {})[row["phase"]] = row["run"]
    rows = []
    for p in sorted(sides):
        spec, nospec = sides[p].get("spec"), sides[p].get("nospec")
        if not spec or not nospec:
            continue
        same = spec["text"] == nospec["text"] and spec.get("reasoning") == nospec.get("reasoning")
        rows.append({"prompt": p, "identical": same,
                     "spec_tokens": spec["completion_tokens"], "nospec_tokens": nospec["completion_tokens"],
                     "spec_tps": spec["decode_tps"], "nospec_tps": nospec["decode_tps"],
                     "spec_text": spec["text"], "nospec_text": nospec["text"],
                     "spec_reasoning_len": len(spec.get("reasoning") or ""),
                     "nospec_reasoning_len": len(nospec.get("reasoning") or ""),
                     "spec_reasoning": (spec.get("reasoning") or "")[:4000],
                     "nospec_reasoning": (nospec.get("reasoning") or "")[:4000],
                     "first_diff": next((i for i, (x, y) in enumerate(zip(spec["text"], nospec["text"]))
                                         if x != y), None)})
        print(f"  prompt {p}: identical={same} spec_tok={spec['completion_tokens']} "
              f"nospec_tok={nospec['completion_tokens']}")
    return rows


# ----------------------------------------------------------------------------- scenario 4
def scenario4(base, count, max_tokens):
    """Sampling: temperature 0.7 / top_p 0.9 - spec must work and stay coherent."""
    prompts = [
        "Write a short paragraph about why double-buffered DMA slots help on a single GPU.",
        "Give three concrete ways to cut per-step PCIe traffic on a 12 GiB card.",
        "Explain speculative decoding to a colleague in two sentences.",
        "Name four metrics you would log for a local LLM server and why.",
        "Describe one failure mode of CUDA graphs in a streaming pipeline.",
    ]
    rows = []
    for i in range(count):
        r = chat(base, [{"role": "user", "content": prompts[i % len(prompts)]}], max_tokens, 0.7, top_p=0.9,
                 seed=1000 + i, label=f"sample{i + 1}")
        r["vram_after"] = vram_now()
        rows.append(r)
        print(f"  sample {i + 1}: dec={_f(round(r['decode_tps'], 2) if r['decode_tps'] else None, '>6.2f')} t/s "
              f"acc/step={r['accepted_per_step']} chars={len(r['text'])} "
              f"{'ERR ' + str(r['error']) if r['error'] else r['finish_reason']}")
    return rows


# ----------------------------------------------------------------------------- scenario 5
def summarise(log_path, runs):
    """Stability: request count from the recorded runs, server restarts, log warnings/errors."""
    n_req = 0
    errs = 0
    for rows in runs:
        for r in rows:
            if not isinstance(r, dict):
                continue
            if "spec" in r:
                n_req += 2
                errs += sum(1 for k in ("spec", "nospec") if r[k].get("error"))
            elif "decode_tps" in r:
                n_req += 1
                errs += 1 if r.get("error") else 0
    diag = {"warn": [], "error": []}
    if log_path and os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if " E " in line or line.lstrip().startswith("E "):
                    diag["error"].append(line.rstrip()[:300])
                elif " W " in line or line.lstrip().startswith("W "):
                    diag["warn"].append(line.rstrip()[:300])
    return {"requests": n_req, "request_errors": errs, "server_restarts": 0,
            "log_warnings": diag["warn"][:40], "log_errors": diag["error"][:40],
            "log_warning_count": len(diag["warn"]), "log_error_count": len(diag["error"])}


def load_runs(paths):
    """Re-read earlier scenario JSONs so scenario 5 can aggregate a whole session."""
    runs = []
    for p in paths:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        try:
            d = json.load(open(full))
        except Exception as e:
            print(f"  (skip {p}: {e})")
            continue
        runs.extend(d.get("runs") or [])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--inputs", default="", help="scenario 5: comma-separated scenario JSONs to aggregate")
    ap.add_argument("--server-log", default="", help="scenario 5: llama-server log to scan for W/E lines")
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--base-url2", default="", help="scenario 3: the no-spec server")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chars", type=int, default=14000, help="scenario 1: context chars per prompt (~4K tokens)")
    ap.add_argument("--turn-tokens", type=int, default=7500, help="scenario 2: new tokens per turn")
    ap.add_argument("--target-tokens", type=int, default=199000, help="scenario 2: stop after this much context")
    ap.add_argument("--turns", type=int, default=0, help="scenario 2: hard cap on turns (0 = by target)")
    ap.add_argument("--count", type=int, default=5, help="scenario 4: requests")
    ap.add_argument("--phase", default="", choices=["", "spec", "nospec"],
                    help="scenario 3: run only this side (the two servers do not fit on the card together)")
    ap.add_argument("--n-ctx", type=int, default=204800, help="scenario 2: the server's -c, to stop before the limit")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vram = VramSampler()
    vram.start()
    t0 = time.time()
    runs = []
    summary = None
    try:
        if a.scenario == 5:
            runs = load_runs([p for p in a.inputs.split(",") if p])
            summary = summarise(os.path.join(ROOT, a.server_log) if a.server_log else "", [runs])
            print(f"  requests={summary['requests']} errors={summary['request_errors']} "
                  f"restarts={summary['server_restarts']} log W={summary['log_warning_count']} "
                  f"log E={summary['log_error_count']}")
        elif a.scenario == 1:
            runs = [scenario1(a.base_url, a.max_tokens, a.chars)]
        elif a.scenario == 2:
            runs = [scenario2(a.base_url, a.max_tokens,
                              a.turn_tokens * a.turns if a.turns else a.target_tokens, a.turn_tokens,
                              out=out, n_ctx=a.n_ctx)]
        elif a.scenario == 3:
            if a.inputs:
                runs = [s3_compare([p for p in a.inputs.split(",") if p])]
            elif a.base_url2:
                sys.exit("scenario 3 runs one server at a time: use --phase spec / --phase nospec and --inputs to compare")
            elif a.phase:
                runs = [scenario3(a.base_url, a.max_tokens, a.phase)]
            else:
                sys.exit("scenario 3 needs --phase spec|nospec (or --inputs to compare two recorded runs)")
        elif a.scenario == 4:
            runs = [scenario4(a.base_url, a.count, a.max_tokens)]
    finally:
        peak = vram.stop()
    rec = {"scenario": a.scenario, "label": a.label or None, "base_url": a.base_url,
           "base_url2": a.base_url2 or None, "max_tokens": a.max_tokens,
           "sim_s": round(time.time() - t0, 1), "vram_peak_mib": peak,
           "runs": runs}
    if summary is not None:
        rec["summary"] = summary
    json.dump(rec, open(out, "w"), indent=1)
    print(f"peak VRAM {peak} MiB, wall {rec['sim_s']} s -> {out}")


if __name__ == "__main__":
    main()
