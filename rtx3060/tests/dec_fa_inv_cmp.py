# Compare DEC_FA_INV dumps: rows for nq=1..4 vs nq=5 at each n_kv prefix.  usage: dec_fa_inv_cmp.py dumpdir n_kv...
import struct, sys, os
def L(p):
    d = open(p, "rb").read()
    return struct.unpack(f"{len(d)//4}f", d)
for n in sys.argv[2:]:
    b = L(os.path.join(sys.argv[1], f"i{n}_24_4_5.f32"))
    print(n, "differing values vs nq=5 for nq=1..4:", [sum(1 for i, v in enumerate(L(os.path.join(sys.argv[1], f"i{n}_24_4_{q}.f32"))) if v != b[i]) for q in range(1, 5)])
