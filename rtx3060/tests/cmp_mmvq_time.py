"""Compare two mmvq_cols_bench outputs: per type/shape/ncols us/matmul, new vs old."""
import sys, collections

def load(path):
    d = {}
    # Tee-Object writes UTF-16 on Windows PowerShell; drop the NULs so the fields parse
    raw = open(path, "rb").read()
    text = raw.decode("utf-16", "ignore") if raw[:2] == b"\xff\xfe" else raw.decode("utf-8", "ignore")
    for line in text.splitlines():
        p = line.replace("\x00", "").split()
        if len(p) != 5 or p[0] == "type":
            continue
        ty, shape, nc, us, usc = p
        d[(ty, shape, int(nc))] = (float(us), float(usc))
    return d

old = load(sys.argv[1])
new = load(sys.argv[2])
print(f"{'type':9s} {'shape':11s} {'nc':>2s} {'old us':>8s} {'new us':>8s} {'delta':>7s}")
by_nc = collections.defaultdict(list)
for k in sorted(old, key=lambda k: (k[0], k[1], k[2])):
    if k not in new:
        continue
    o, n = old[k][0], new[k][0]
    dl = 100.0 * (n - o) / o
    by_nc[k[2]].append((k[0], k[1], dl))
    print(f"{k[0]:9s} {k[1]:11s} {k[2]:2d} {o:8.1f} {n:8.1f} {dl:+6.1f}%")
print()
print("mean delta per ncols (all types / IQ3 types only, the ones the flag covers):")
for nc in sorted(by_nc):
    rows = by_nc[nc]
    iq3 = [d for t, s, d in rows if t.startswith("iq3")]
    print(f"  nc={nc}: all {sum(d for _, _, d in rows) / len(rows):+6.1f}%   "
          f"iq3 {sum(iq3) / len(iq3):+6.1f}%")
