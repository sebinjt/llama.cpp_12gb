"""Column-invariance check for the cols bench dumps.

For one prefix (e.g. "mmq"), compare column j's output across every ncols dump that
contains it (ncols 1,2,4,5,8): it must be bit-identical if the kernel is batch-invariant.
Also compare the full dumps of two prefixes (e.g. mmvq vs mmq) per column.

  python tests/cmp_cols_columns.py results/prof/cols mmq
  python tests/cmp_cols_columns.py results/prof/cols mmvq mmq
"""
import glob, os, re, struct, sys

D = sys.argv[1]
prefixes = sys.argv[2:]
COLS = [1, 2, 4, 5, 8]


def load(path):
    raw = open(path, "rb").read()
    return struct.unpack("<%df" % (len(raw) // 4), raw)


def bits(x):
    return struct.pack("<f", x)


for pa in prefixes:
    files = glob.glob(os.path.join(D, pa + "_*.f32"))
    by = {}
    for f in files:
        m = re.match(rf"{re.escape(pa)}_(.+)_(\d+)_(\d+)_c(\d+)\.f32$", os.path.basename(f))
        if m:
            by.setdefault((m.group(1), int(m.group(2)), int(m.group(3))), {})[int(m.group(4))] = f
    print(f"\n== {pa}: column invariance across ncols")
    print(f"{'type':10s} {'shape':13s} {'col':>3s} {'dumps':>5s} {'bit_diffs':>9s} {'max_rel':>10s}")
    bad = 0
    for (ty, i, o), d in sorted(by.items()):
        ref = load(d[max(d)])
        for col in range(max(d)):
            if not any(col < nc for nc in d):
                continue
            diffs = 0
            rel = 0.0
            n = 0
            for nc, f in d.items():
                if col >= nc:
                    continue
                vals = load(f)[col * o:(col + 1) * o]
                refv = ref[col * o:(col + 1) * o]
                n += 1
                for x, y in zip(refv, vals):
                    if bits(x) != bits(y):
                        diffs += 1
                    rel = max(rel, abs(x - y) / max(abs(x), abs(y), 1e-30))
            if diffs:
                bad += 1
            print(f"{ty:10s} {i}x{o:<7d} {col:3d} {n:5d} {diffs:9d} {rel:10.3e}"
                  f"{'  <-- DIFFERS' if diffs else ''}")
    print("VERDICT:", "column-invariant" if bad == 0 else f"{bad} column(s) differ")

if len(prefixes) == 2:
    a, b = prefixes
    print(f"\n== {a} vs {b}: per-column diff")
    for fa in sorted(glob.glob(os.path.join(D, a + "_*.f32"))):
        tail = os.path.basename(fa)[len(a) + 1:]
        fb = os.path.join(D, b + "_" + tail)
        if not os.path.exists(fb):
            print(f"{tail:30s} MISSING")
            continue
        va, vb = load(fa), load(fb)
        if len(va) != len(vb):
            print(f"{tail:30s} SIZE")
            continue
        diffs = sum(1 for x, y in zip(va, vb) if bits(x) != bits(y))
        rel = max((abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(va, vb)), default=0.0)
        print(f"{tail:30s} n={len(va):7d} bits={diffs:7d} max_rel={rel:.3e}")
