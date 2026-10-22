"""Parse MMTIME/FATIME logs into per-pass, per-class breakdowns (target verify pass vs speculative drafter).

Passes are delimited by the LM head (`output.weight`).  A pass that contains blk.0.attn_qkv.weight is the
target model's forward; a pass with `fc` / `selector_hidden` / `*_conv_proj` is the DFlash drafter; a pass
with blk.64.* is the MTP drafter.  FA calls carry n_head: 24 = target (16 attn layers of Qwen3.8-27B),
32 = drafter (DFlash cross-attention heads).

Usage: python tests/parse_step_timing.py <server.log> [--json out.json] [--skip N] [--tail N]
"""
import json
import re
import sys
from collections import defaultdict

mm_pat = re.compile(r"^MMTIME ([\d.]+) name=(\S+) n=(\d+) K=(\d+)")
fa_pat = re.compile(r"^FATIME ([\d.]+) n_kv=(\d+) n_q=(\d+) n_head=(\d+)")


def kind(name):
    if name == "output.weight":
        return "lm_head"
    if name.startswith("fc") or name.startswith("selector_hidden"):
        return "draft_head"
    if "conv_proj" in name:
        return "draft_conv"
    if "attn_inp_k_rot" in name:
        return "rot"
    if ".ssm_" in name:
        return "gdn_mm"
    if ".attn_" in name:
        return "attn_mm"
    if ".ffn_" in name:
        return "ffn_mm"
    if name.startswith("blk.64."):
        return "mtp_mm"
    return "other_mm"


CLASSES = ["ffn_mm", "attn_mm", "gdn_mm", "lm_head", "rot", "draft_head", "draft_conv", "mtp_mm", "other_mm"]


def parse(path):
    passes, cur = [], None
    for line in open(path, errors="replace"):
        m = mm_pat.match(line)
        if m:
            ms, name, n, K = float(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
            if cur is None:
                cur = {"ops": [], "fa": []}
            cur["ops"].append((name, n, K, ms))
            if name == "output.weight":
                passes.append(cur)
                cur = None
            continue
        f = fa_pat.match(line)
        if f and cur is not None:
            cur["fa"].append((float(f.group(1)), int(f.group(2)), int(f.group(3)), int(f.group(4))))
    for p in passes:
        names = [o[0] for o in p["ops"]]
        if any(nm.startswith("blk.0.attn_qkv") for nm in names):
            p["type"] = "target"
        elif any(nm.startswith("fc") or nm.startswith("selector_hidden") or "conv_proj" in nm for nm in names):
            p["type"] = "draft_dflash"
        elif any(nm.startswith("blk.64.") for nm in names):
            p["type"] = "draft_mtp"
        else:
            p["type"] = "unknown"
        p["n"] = max(o[1] for o in p["ops"])
        p["mm"] = defaultdict(float)
        p["mm_ops"] = defaultdict(int)
        for name, n, K, ms in p["ops"]:
            p["mm"][kind(name)] += ms
            p["mm_ops"][kind(name)] += 1
        p["mm_total"] = sum(o[3] for o in p["ops"])
        p["fa_calls"] = len(p["fa"])
        p["fa"] = sum(x[0] for x in p["fa"])
        p["nops"] = len(p["ops"])
    return passes


def show(lst, label):
    if not lst:
        return 0.0
    print(f"\n{label}: {len(lst)} passes, n={sorted({p['n'] for p in lst})}, "
          f"matmul ops/pass {sum(p['nops'] for p in lst)/len(lst):.0f}, "
          f"FA calls/pass {sum(p['fa_calls'] for p in lst)/len(lst):.1f}")
    print(f"  {'class':11s} {'ms/pass':>8s} {'ops/pass':>9s} {'ms/op':>7s}")
    tot = 0.0
    for c in CLASSES:
        v = sum(p["mm"].get(c, 0.0) for p in lst) / len(lst)
        o = sum(p["mm_ops"].get(c, 0) for p in lst) / len(lst)
        if v or o:
            print(f"  {c:11s} {v:8.3f} {o:9.1f} {v/max(o,1e-9):7.4f}")
            tot += v
    fa = sum(p["fa"] for p in lst) / len(lst)
    print(f"  {'matmul sum':11s} {tot:8.3f}")
    print(f"  {'flash_attn':11s} {fa:8.3f}")
    print(f"  {'TIMED sum':11s} {tot+fa:8.3f}")
    return tot + fa


def main():
    path = sys.argv[1]
    ps = parse(path)
    cnt = defaultdict(int)
    for p in ps:
        cnt[p["type"]] += 1
    print(f"passes: {len(ps)}  " + " ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 2
    def steady(lst):
        return lst[skip:len(lst) - 1] if len(lst) > skip + 3 else lst
    tgt = steady([p for p in ps if p["type"] == "target"])
    drf = steady([p for p in ps if p["type"].startswith("draft")])
    t = show(tgt, "TARGET forward (verify)")
    d = show(drf, "DRAFTER pass")
    if tgt and drf:
        print(f"\nPER STEP: target {t:.2f} ms + {len(drf)/max(len(tgt),1):.2f} drafter passes "
              f"{d*len(drf)/max(len(tgt),1):.2f} ms = {t + d*len(drf)/max(len(tgt),1):.2f} ms timed GPU")
    if "--json" in sys.argv:
        json.dump([{k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in p.items()} for p in ps],
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
