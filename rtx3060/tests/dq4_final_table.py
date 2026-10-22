"""Summarize the final sweep (results/dq4/F-*.json) into a table."""
import json, os, glob
rows = []
for t in (2, 3):
    ref = None
    names = [f"F-t{t}-nomtp-vec-L9", f"F-t{t}-nomtp-dq4-L9", f"F-t{t}-mtp-flagsoff-L14-d3"] + [f"F-t{t}-best-L11-d{d}" for d in (2, 3, 4, 5)]
    for n in names:
        f = f"results/dq4/{n}.json"
        if not os.path.exists(f):
            rows.append(f"{n:30s} missing"); continue
        x = json.load(open(f))["reps"][0]
        N = x["predicted_n"]; dn = x["draft_n"] or 0; da = x["draft_n_accepted"] or 0
        passes = N - da
        toks = x["tokens"]
        if ref is None: ref = toks
        d = next((i for i in range(min(len(ref), len(toks))) if ref[i] != toks[i]), None)
        rows.append(f"{n:30s} {x['tok_s']:7.3f} t/s  ms/tok {x['ms_per_tok']:6.1f}  tok/pass {N/passes:4.2f}  ms/pass {x['ms_per_tok']*N/passes:6.1f}  "
                    f"drafted {dn:4d} accepted {da:4d} acc {da/max(1,dn):.3f}  first diff vs vec no-MTP: {d}")
    rows.append("")
print("\n".join(rows))
