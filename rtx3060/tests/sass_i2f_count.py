"""Count I2FP / IADD3 / FADD in mul_mat_q kernels of a cuobjdump -sass dump.

Usage: python tests/sass_i2f_count.py <sass file> [type ...]
Prints one row per mul_mat_q<type, J> instantiation so an A/B of the source change can be read off directly.
"""
import re
import sys

path = sys.argv[1]
want_types = [int(t) for t in sys.argv[2:]] or [16, 17, 18, 21, 22, 23]

txt = open(path, encoding="utf-8", errors="replace").read()
blocks = re.split(r"\n\s*Function : ", txt)

rows = []
for b in blocks:
    name = b.split("\n")[0].strip()
    if "mul_mat_q" not in name:
        continue
    m = re.search(r"IL9ggml_type(\d+)ELi(\d+)E", name)
    if not m:
        continue
    ty, j = int(m.group(1)), int(m.group(2))
    if ty not in want_types:
        continue
    rows.append((ty, j, len(re.findall(r"^\s*/\*[0-9a-f]{4}\*/\s+I2FP", b, re.M)),
                 len(re.findall(r"^\s*/\*[0-9a-f]{4}\*/\s+IADD3", b, re.M)),
                 len(re.findall(r"^\s*/\*[0-9a-f]{4}\*/\s+FADD", b, re.M))))

print(f"{'type':>4s} {'J':>4s} {'I2FP':>6s} {'IADD3':>6s} {'FADD':>6s}  (instantiation)")
for ty, j, i2f, iadd3, fadd in sorted(rows):
    print(f"{ty:4d} {j:4d} {i2f:6d} {iadd3:6d} {fadd:6d}")
print(f"{len(rows)} kernel(s)")
