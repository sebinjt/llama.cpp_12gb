"""Dump selected GGUF metadata keys. Usage: gguf_meta.py <model.gguf> [substr ...]"""
import struct, sys

path = sys.argv[1]
want = [s.lower() for s in sys.argv[2:]] or ["dflash", "arch", "target", "block", "selector", "nextn", "rope"]

f = open(path, "rb")
def rd(fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]
def rstr():
    return f.read(rd("Q")).decode("utf8", "replace")
def val(t):
    if t == 0: return rd("B")
    if t == 1: return rd("b")
    if t == 2: return rd("H")
    if t == 3: return rd("h")
    if t == 4: return rd("I")
    if t == 5: return rd("i")
    if t == 6: return rd("f")
    if t == 7: return rd("B")
    if t == 8: return rstr()
    if t == 9:
        et = rd("I"); n = rd("Q")
        return [val(et) for _ in range(n)]
    if t == 10: return rd("Q")
    if t == 11: return rd("q")
    if t == 12: return rd("d")
    raise ValueError(t)

assert f.read(4) == b"GGUF"
ver = rd("I"); nt = rd("Q"); nk = rd("Q")
meta = {}
for _ in range(nk):
    k = rstr(); t = rd("I"); meta[k] = val(t)
print(f"{path}: gguf v{ver}, {nt} tensors, {nk} metadata")
for k in sorted(meta):
    if any(w in k.lower() for w in want):
        print(f"  {k} = {meta[k]!r}"[:300])
