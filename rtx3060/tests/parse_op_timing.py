"""Parse MMTIME/FATIME logs into per-pass breakdowns.

Segments: a full target pass starts at blk.0.attn_qkv.weight and ends at output.weight;
a draft pass starts at blk.64.nextn.eh_proj.weight and ends at output.weight.
"""
import collections, re, sys

pat = re.compile(r"^(MMTIME|FATIME) ([\d.]+)(.*)$")
kv = re.compile(r"name=(\S+) n=(\d+) K=(\d+)")
fa = re.compile(r"n_kv=(\d+) n_q=(\d+) n_head=(\d+)")


def kind(name):
    if name == "output.weight":
        return "lm_head"
    if name.startswith("blk.64."):
        return "mtp_block"
    if "attn_inp_k_rot" in name:
        return "rot"
    if ".ssm_" in name:
        return "gdn"
    if ".attn_" in name:
        return "attn_weights"
    if ".ffn_" in name:
        return "ffn"
    return "other"


def segment(path):
    segs = []
    cur = None
    for line in open(path, errors="replace"):
        m = pat.match(line)
        if not m:
            continue
        typ, ms, rest = m.group(1), float(m.group(2)), m.group(3)
        if typ == "MMTIME":
            mm = kv.search(rest)
            if not mm:
                continue
            name, n, K = mm.group(1), int(mm.group(2)), int(mm.group(3))
            if name == "blk.0.attn_qkv.weight":
                cur = {"type": "target", "n": n, "rows": [], "fa": [], "start": None}
                segs.append(cur)
            elif name == "blk.64.nextn.eh_proj.weight":
                cur = {"type": "draft", "n": n, "rows": [], "fa": [], "start": None}
                segs.append(cur)
            if cur is None:
                continue
            cur["rows"].append((name, n, ms))
            if name == "output.weight":
                cur = None
        else:
            mm = fa.search(rest)
            if mm and cur is not None:
                cur["fa"].append((int(mm.group(1)), int(mm.group(2)), ms))
    return segs


def summarize(path, show=8):
    segs = segment(path)
    print(f"\n== {path} : {len(segs)} segments")
    by_type = collections.defaultdict(list)
    for s in segs:
        by_type[(s["type"], s["n"])].append(s)
    print(f"  {'type':<8}{'n':>3}{'count':>7}{'mm_ms':>9}{'fa_ms':>8}{'tot_ms':>9}")
    for (t, n), ss in sorted(by_type.items()):
        mm = sum(r[2] for s in ss for r in s["rows"] if kind(r[0]) != "lm_head")
        lm = sum(r[2] for s in ss for r in s["rows"] if kind(r[0]) == "lm_head")
        fa = sum(f[2] for s in ss for f in s["fa"])
        print(f"  {t:<8}{n:>3}{len(ss):>7}{mm:>9.2f}{fa:>8.2f}{mm+lm+fa:>9.2f}")

    for key in [("target", 5), ("draft", 1), ("target", 3), ("target", 2), ("target", 1)]:
        ss = by_type.get(key)
        if not ss:
            continue
        print(f"\n  -- {key[0]} n={key[1]}, last pass breakdown ({len(ss)} passes, showing last):")
        s = ss[-1]
        agg = collections.defaultdict(float)
        cnt = collections.Counter()
        for name, n, ms in s["rows"]:
            agg[kind(name)] += ms
            cnt[kind(name)] += 1
        fa = sum(f[2] for f in s["fa"])
        tot = sum(agg.values()) + fa
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"    {k:<13}{cnt[k]:>4} calls {v:8.2f} ms {100*v/tot:5.1f}%")
        print(f"    {'flash_attn':<13}{len(s['fa']):>4} calls {fa:8.2f} ms {100*fa/tot:5.1f}%")
        print(f"    {'TOTAL':<13}{'':>4}       {tot:8.2f} ms")
        # top tensors
        top = sorted(s["rows"], key=lambda r: -r[2])[:show]
        print("    top tensors: " + ", ".join(f"{n.split('.weight')[0]}={ms:.2f}" for n, _, ms in top))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
