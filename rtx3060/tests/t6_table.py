"""Per-prompt t/s, tok/step table for the 200K mode comparison (T6) plus the mean and spread.

Usage: python tests/t6_table.py [glob]
"""
import glob
import json
import statistics
import sys
from collections import defaultdict


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/prof/t6_200k_s*_*.json"
    rows = defaultdict(dict)
    for f in sorted(glob.glob(pat)):
        name = f.split("/")[-1].replace("t6_200k_", "").replace(".json", "")
        if "_" not in name:
            continue
        prompt, mode = name.rsplit("_", 1)
        try:
            d = json.load(open(f))
            t = d["runs"]["d7"]["timings"]
        except Exception:
            continue
        rows[prompt][mode] = {
            "tps": t["predicted_per_second"],
            "ms_step": t["predicted_ms"] / t["draft_verification_steps"],
            "tok_step": t["draft_mean_accepted_length"],
            "steps": t["draft_verification_steps"],
            "sha": d["runs"]["d7"]["sha"],
        }
    print(f"{'prompt':5s} {'mode':5s} {'t/s':>6s} {'ms/step':>8s} {'tok/step':>8s} {'steps':>6s}  sha")
    agg = defaultdict(list)
    for p in sorted(rows):
        for m in ("mmvq", "mmq"):
            if m not in rows[p]:
                print(f"{p:5s} {m:5s}   (missing)")
                continue
            r = rows[p][m]
            print(f"{p:5s} {m:5s} {r['tps']:6.2f} {r['ms_step']:8.1f} {r['tok_step']:8.2f} {r['steps']:6d}  {r['sha']}")
            agg[m].append(r)
    print()
    for m in ("mmvq", "mmq"):
        if not agg[m]:
            continue
        tps = [r["tps"] for r in agg[m]]
        tks = [r["tok_step"] for r in agg[m]]
        mss = [r["ms_step"] for r in agg[m]]
        print(f"{m}: mean {statistics.mean(tps):6.2f} t/s (median {statistics.median(tps):6.2f}, "
              f"min {min(tps):6.2f}, max {max(tps):6.2f}, n={len(tps)}) | "
              f"tok/step mean {statistics.mean(tks):.3f} | ms/step mean {statistics.mean(mss):.1f}")
    if len(agg["mmvq"]) and len(agg["mmq"]):
        both = [p for p in rows if "mmvq" in rows[p] and "mmq" in rows[p]]
        diffs = [(rows[p]["mmq"]["tps"] / rows[p]["mmvq"]["tps"] - 1.0) * 100 for p in both]
        wins = sum(1 for d in diffs if d > 0)
        print(f"paired: MMQ vs MMVQ per prompt: " + ", ".join(f"{p}:{d:+.1f}%" for p, d in zip(both, diffs)))
        print(f"        mean {statistics.mean(diffs):+.2f}% (MMQ wins {wins}/{len(diffs)} prompts)")


if __name__ == "__main__":
    main()
