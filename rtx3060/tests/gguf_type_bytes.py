"""Weight bytes per quant type for a GGUF model."""
import collections, sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf as G
from gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
r = GGUFReader(path)
tot = collections.Counter()
for t in r.tensors:
    n = 1
    for d in t.shape:
        n *= int(d)
    bs = G.GGML_QUANT_SIZES.get(t.tensor_type)
    tot[t.tensor_type.name] += (n // bs[0]) * bs[1] if bs else n * 4
s = sum(tot.values())
print(f"{path}: total {s / 2**30:.2f} GiB")
for ty, b in tot.most_common():
    print(f"  {ty:10s} {b / 2**20:9.1f} MiB  {100 * b / s:5.1f}%")
