"""Print the real-world (tests/realworld_chat.py) results as the tables used in the report."""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    p = os.path.join(ROOT, "results/rw", name)
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(v, spec):
    return format(v, spec) if v is not None else "-"


def s1():
    a, b = load("s1_c32768.json"), load("s1_c204800.json")
    if not (a and b):
        return
    A = {r["label"]: r for r in a["runs"][0]}
    B = {r["label"]: r for r in b["runs"][0]}
    print("scenario 1 - same prompts, same profile (plong), only -c differs; fresh server, cache cold")
    print(f"{'prompt':<15}{'cached':>7}{'PP 32K':>8}{'PP 200K':>9}{'PP d':>7}{'dec 32K':>9}{'dec 200K':>10}{'dec d':>7}")
    for k in A:
        if k not in B:
            continue
        x, y = A[k], B[k]
        print(f"{k:<15}{fmt(x['cached_tokens'], '>7')}{fmt(x['prompt_tps'], '>8.1f')}{fmt(y['prompt_tps'], '>9.1f')}"
              f"{100 * (y['prompt_tps'] / x['prompt_tps'] - 1):>6.1f}%{fmt(x['decode_tps'], '>9.2f')}"
              f"{fmt(y['decode_tps'], '>10.2f')}{100 * (y['decode_tps'] / x['decode_tps'] - 1):>6.1f}%")
    pa = [A[k]["prompt_tps"] for k in A if k in B]
    pb = [B[k]["prompt_tps"] for k in A if k in B]
    da = [A[k]["decode_tps"] for k in A if k in B]
    db = [B[k]["decode_tps"] for k in A if k in B]
    print(f"{'mean':<15}{'':>7}{st.mean(pa):>8.1f}{st.mean(pb):>9.1f}{100 * (st.mean(pb) / st.mean(pa) - 1):>6.1f}%"
          f"{st.mean(da):>9.2f}{st.mean(db):>10.2f}{100 * (st.mean(db) / st.mean(da) - 1):>6.1f}%")
    print(f"peak VRAM: 32K {a['vram_peak_mib']} MiB, 200K {b['vram_peak_mib']} MiB\n")


def s2():
    d = load("s2_grow199k.json")
    if not d:
        return
    rows = d["runs"][0]
    print("scenario 2 - growing multi-turn (plong, -c 204800), each turn appends ~10K tokens of real repo text")
    print(f"{'turn':>4}{'ctx~':>8}{'prompt':>8}{'cached':>8}{'new tok':>8}{'new s':>7}{'TTFT s':>8}"
          f"{'dec t/s':>8}{'acc/step':>9}{'VRAM':>7}")
    for r in rows:
        if r.get("over_limit_probe"):
            print(f"  over-limit probe: {'ERR ' + str(r['error'])[:110] if r['error'] else 'accepted'}")
            continue
        new = (r["prompt_tokens"] - (r["cached_tokens"] or 0)) if r["prompt_tokens"] and r["cached_tokens"] is not None else None
        print(f"{r['turn']:>4}{fmt(r.get('approx_ctx_tokens'), '>8')}{fmt(r['prompt_tokens'], '>8')}"
              f"{fmt(r['cached_tokens'], '>8')}{fmt(new, '>8')}{fmt((r['prompt_ms'] or 0) / 1000, '>7.1f')}"
              f"{fmt(r['ttft_s'], '>8.2f')}{fmt(r['decode_tps'], '>8.2f')}{fmt(r['accepted_per_step'], '>9.2f')}"
              f"{fmt(r['vram_after'], '>7')}")
    print(f"stop reason: {d['runs'][0][-1].get('over_limit_probe') and 'context limit reached (clean HTTP 400)'}"
          f"  peak VRAM {d['vram_peak_mib']} MiB  wall {d['sim_s']} s\n")


def s3():
    d = load("s3_compare.json")
    if not d:
        return
    print("scenario 3 - spec vs no-spec through the server (p8k, temperature 0, 768 tokens)")
    for r in d["runs"][0]:
        print(f"  prompt {r['prompt']}: identical={r['identical']}  spec {r['spec_tps']:.2f} t/s "
              f"({r['spec_tokens']} tok)  no-spec {r['nospec_tps']:.2f} t/s ({r['nospec_tokens']} tok)  "
              f"reasoning {r['spec_reasoning_len']}/{r['nospec_reasoning_len']} chars  "
              f"first_diff={r['first_diff']}")
    print()


def s4():
    print("scenario 4 - sampling temperature 0.7 / top_p 0.9")
    for name, tag in (("s4_p8k.json", "p8k (-c 16384)"), ("s4_plong.json", "plong (-c 204800)"),
                      ("s4_s64k.json", "s64k (IQ3_S, -c 65792)")):
        d = load(name)
        if not d:
            continue
        rows = d["runs"][0]
        dec = [r["decode_tps"] for r in rows if r["decode_tps"]]
        acc = [r["accepted_per_step"] for r in rows if r["accepted_per_step"]]
        txt = sum(1 for r in rows if r["text"])
        print(f"  {tag:<24} n={len(rows)} dec {st.mean(dec):.2f} t/s (min {min(dec):.2f}) "
              f"acc/step {st.mean(acc):.2f}  answers {txt}/{len(rows)}  errs {sum(1 for r in rows if r['error'])} "
              f"peak {d['vram_peak_mib']} MiB")
    print()


def s5():
    ins = [p for p in ("s1_c32768.json", "s1_c204800.json", "s1_s64k.json", "s2_grow199k.json",
                       "s3_p8k_spec.json", "s3_p8k_nospec.json", "s4_p8k.json", "s4_plong.json", "s4_s64k.json")]
    n = err = 0
    peaks = []
    for name in ins:
        d = load(name)
        if not d:
            continue
        peaks.append(d["vram_peak_mib"])
        for r in d["runs"][0]:
            if r.get("over_limit_probe"):
                continue
            n += 1
            err += 1 if r.get("error") else 0
    print(f"scenario 5 - stability: requests={n} request_errors={err} max_peak_vram={max(peaks)} MiB "
          f"across {len(peaks)} sessions")
    for log in ("srv_plong32k_final.log", "srv_plong200k.log", "srv_p8k.log", "srv_p8k_nospec.log",
                "srv_s64k_final.log"):
        p = os.path.join(ROOT, "results/rw", log)
        if not os.path.exists(p):
            continue
        e = w = 0
        for line in open(p, encoding="utf-8", errors="replace"):
            if " E " in line:
                e += 1
            elif " W " in line:
                w += 1
        print(f"  {log:<28} log errors={e} warnings={w}")


if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for fn in ([s1, s2, s3, s4, s5] if what == "all" else [globals()[what]]):
        fn()
