"""Per-(type,shape) weight-byte census of a GGUF, to turn per-tensor x-of-floor deltas into ms/step."""
import collections, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf as G
from gguf import GGUFReader
path = sys.argv[1]
r = GGUFReader(path)
tot = collections.Counter()
for t in r.tensors:
    n = 1
    for d in t.shape: n *= int(d)
    bs = G.GGML_QUANT_SIZES.get(t.tensor_type)
    nbytes = (n // bs[0]) * bs[1] if bs else n * 4
    sh = tuple(int(x) for x in t.shape)
    tot[(t.tensor_type.name, sh)] += nbytes
s = sum(tot.values())
print(f"# {path}  total {s/2**30:.2f} GiB")
agg = collections.Counter()
for (ty, sh), b in tot.items():
    if len(sh) == 2: agg[(ty, sh)] += b
for (ty, sh), b in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"{ty:9s} {sh[0]:>6d}x{sh[1]:<6d} {b/2**20:8.1f} MiB  {100*b/s:5.1f}%")
