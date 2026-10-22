"""Summarize dq4_e2e result jsons: prompt_n, tokens, t/s, draft stats, sha."""
import json, glob, os, sys

pat = sys.argv[1] if len(sys.argv) > 1 else "results/dq4/*.json"
for p in sorted(glob.glob(pat)):
    try:
        d = json.load(open(p))
    except Exception as e:
        continue
    reps = d.get("reps") or []
    if not reps:
        continue
    for i, r in enumerate(reps):
        n = r.get("predicted_n") or 0
        acc = ""
        if r.get("draft_n"):
            acc = f" drafted={r['draft_n']} acc={r['draft_n_accepted']} ({r['draft_n_accepted']/r['draft_n']:.3f})"
        print(f"{os.path.basename(p):34s} rep{i} pn={r.get('prompt_n')} tgn={n} "
              f"t/s={r.get('tok_s') or 0:.3f} ms/tok={r.get('ms_per_tok') or 0:.2f}{acc} sha={r.get('tokens_sha')}")
