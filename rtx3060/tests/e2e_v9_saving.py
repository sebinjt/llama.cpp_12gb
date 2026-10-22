"""ms saved per 8K verify step if v9 replaces MMQ@8 for the IQ3_S/IQ3_XXS tensors, using the same-run
x-of-floor numbers measured in seq 81 (comm/files/seq81_v9.md). floor = bytes / 325 GB/s."""
import collections, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf as G
from gguf import GGUFReader

# x of the weight-read floor at ncols=8, measured 2026-10-21 in the same session
# shape -> (MMQ@8 IQ3_S, v9 IQ3_S exact, v9 IQ3_S +split, MMQ@8 XXS, v9 XXS exact, v9 XXS +split)
X = {
 (5120,17408):  (2.097, 1.956, 1.752, 2.049, 2.219, 1.781),
 (17408,5120):  (2.621, 2.710, 1.624, 2.629, 3.056, 1.848),
 (5120,10240):  (2.096, 1.653, 1.667, 2.052, 1.815, 1.769),
 (6144,5120):   (2.363, 2.565, 1.696, 2.372, 2.833, 1.881),
}
TYI = {"IQ3_S": (0,1,2), "IQ3_XXS": (3,4,5)}
BW = 325e9
path = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
r = GGUFReader(path)
buckets = collections.Counter()
for t in r.tensors:
    n = 1
    for d in t.shape: n *= int(d)
    bs = G.GGML_QUANT_SIZES.get(t.tensor_type)
    nb = (n // bs[0]) * bs[1] if bs else n * 4
    if len(t.shape) == 2 and t.tensor_type.name in TYI:
        buckets[(t.tensor_type.name, int(t.shape[0]), int(t.shape[1]))] += nb
tot = sum(buckets.values())
print(f"# {path}: grid-type (IQ3_S+IQ3_XXS) tensors = {tot/2**30:.2f} GiB of measured-shape bytes")
for mode, idx in (("exact (no split)", 1), ("with split-K", 2)):
    saved_ms = 0.0
    per = collections.Counter()
    for (ty, i, o), nb in buckets.items():
        if (i, o) not in X: continue
        base = 0 if ty == "IQ3_S" else 3
        xm, xv = X[(i, o)][base], X[(i, o)][idx if ty == "IQ3_S" else idx + 3]
        floor_ms = nb / BW * 1e3
        saved_ms += floor_ms * (xm - xv)
        per[(ty, i, o)] += floor_ms * (xm - xv)
    print(f"\n## {mode}: {saved_ms:.2f} ms per verify step  ({100*saved_ms/101.0:+.1f}% of a 101 ms step)")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"   {k[0]:8s} {k[1]}x{k[2]:<6d} {v:+.2f} ms")
