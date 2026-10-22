# Compare two dec_fa_bench output dumps: python dec_fa_cmp.py a.f32 b.f32 [more pairs...]
import struct, sys
def load(p):
    d = open(p, "rb").read()
    return struct.unpack(f"{len(d)//4}f", d)
a = sys.argv[1:]
for i in range(0, len(a) - 1, 2):
    x, y = load(a[i]), load(a[i + 1])
    n = min(len(x), len(y))
    md = max(abs(x[j] - y[j]) for j in range(n))
    mx = max(abs(v) for v in x[:n]) or 1.0
    nan = sum(1 for v in y[:n] if v != v)
    rms = (sum((x[j] - y[j]) ** 2 for j in range(n)) / n) ** 0.5
    rmsx = (sum(v * v for v in x[:n]) / n) ** 0.5 or 1.0
    print(f"{a[i]} vs {a[i+1]}: n={n} max|diff|={md:.3e} max|a|={mx:.3e} rel={md/mx:.3e} rms_rel={rms/rmsx:.3e} nan={nan}")
