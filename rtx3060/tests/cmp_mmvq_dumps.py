"""Compare MMVQ bench dumps bit for bit: <a>_<type>_<in>_<out>_c<n>.f32 vs <b>_...."""
import glob, os, struct, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join("tests", "bin", "mmvq")
pa = sys.argv[2] if len(sys.argv) > 2 else "old"
pb = sys.argv[3] if len(sys.argv) > 3 else "new"
as_ = sorted(glob.glob(os.path.join(D, pa + "_*.f32")))
if not as_:
    sys.exit(f"no {pa}_*.f32 in " + D)

print(f"{'dump':34s} {'n':>7s} {'bit_diffs':>9s} {'max_rel':>10s}")
bad = 0
for ap_ in as_:
    tail = os.path.basename(ap_)[len(pa) + 1:]
    bp = os.path.join(D, pb + "_" + tail)
    if not os.path.exists(bp):
        print(f"{tail:34s} MISSING {pb}")
        bad += 1
        continue
    a = open(ap_, "rb").read()
    b = open(bp, "rb").read()
    if len(a) != len(b):
        print(f"{tail:34s} SIZE {len(a)} vs {len(b)}")
        bad += 1
        continue
    n = len(a) // 4
    fa = struct.unpack(f"<{n}f", a)
    fb = struct.unpack(f"<{n}f", b)
    bits = sum(1 for x, y in zip(fa, fb) if struct.pack("<f", x) != struct.pack("<f", y))
    rel = max((abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(fa, fb)), default=0.0)
    if bits:
        bad += 1
    print(f"{tail:34s} {n:7d} {bits:9d} {rel:10.3e}{'  <-- DIFFERS' if bits else ''}")
print("\nVERDICT:", "all bit-identical" if bad == 0 else f"{bad} dump(s) differ")
