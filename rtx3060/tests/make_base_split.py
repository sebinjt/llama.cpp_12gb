"""Split a prompt into a long base and a suffix that covers the DFlash 2048-token window.

The base is prefilled once and saved; each later run restores it and lets the server
prefill only the suffix when the full prompt is sent.
"""
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--full", required=True)
ap.add_argument("--base-out", required=True)
ap.add_argument("--suffix-out", required=True)
ap.add_argument("--suffix-chars", type=int, default=12000)
a = ap.parse_args()

text = open(a.full, encoding="utf-8", newline="").read()
split = max(0, len(text) - a.suffix_chars)
nl = text.rfind("\n", 0, split)
if nl > 0 and split - nl < 4000:
    split = nl + 1
base, suffix = text[:split], text[split:]
assert base + suffix == text
open(a.base_out, "w", encoding="utf-8", newline="").write(base)
open(a.suffix_out, "w", encoding="utf-8", newline="").write(suffix)
print(f"full {len(text)} chars -> base {len(base)} chars ({os.path.basename(a.base_out)}), "
      f"suffix {len(suffix)} chars ({os.path.basename(a.suffix_out)})")
