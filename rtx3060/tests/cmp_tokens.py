"""Compare decoded token ids against a baseline run; report the first differing token.

Handles both _cfg.py result jsons ({args, response:{tokens:[...]}}) and dq4_e2e.py jsons ({reps:[{tokens:[...]}]}).
Usage: python tests/cmp_tokens.py <baseline.json> <other.json> [...]
"""
import json, os, sys


def tokens_of(path):
    d = json.load(open(path))
    if "response" in d:
        return d["response"].get("tokens") or []
    reps = d.get("reps") or []
    if reps:
        return reps[-1].get("tokens") or []
    return []


base_path = sys.argv[1]
base = tokens_of(base_path)
print(f"baseline {os.path.basename(base_path)}: {len(base)} tokens")
for p in sys.argv[2:]:
    t = tokens_of(p)
    n = min(len(base), len(t))
    first = next((i for i in range(n) if base[i] != t[i]), None)
    tag = "IDENTICAL" if first is None and len(t) == len(base) else (
        f"differs at token {first}" if first is not None else f"length {len(t)} vs {len(base)}")
    print(f"  {os.path.basename(p):34s} {len(t):4d} tok  {tag}")
