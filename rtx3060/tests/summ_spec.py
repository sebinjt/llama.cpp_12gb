"""Summarize the _cfg.py 8K DFlash/MTP sweeps: t/s, tokens/step, acceptance, VRAM."""
import glob, json, os, sys

paths = sys.argv[1:] or sorted(glob.glob("results/dq4/D8k-dflash*.json")) + \
    ["results/dq4/D8k-nomtp.json", "results/dq4/D8k-mtp4.json"]
print(f"{'run':16s} {'spec':12s} {'n_max':>5s} {'t/s':>7s} {'ms/tok':>7s} {'PP t/s':>7s} "
      f"{'drafted':>7s} {'accept':>7s} {'acc%':>6s} {'tok/step':>8s} {'vram':>6s}")
for p in paths:
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    r = d.get("response", {})
    t = r.get("timings", {})
    a = d.get("args", {})
    n = max(1, t.get("predicted_n") or 1)
    dn = t.get("draft_n") or 0
    da = t.get("draft_n_accepted") or 0
    steps = t.get("draft_verification_steps") or 0
    print(f"{a.get('tag', os.path.basename(p)):16s} {a.get('spec') or '-':12s} {str(a.get('draft_max') or '-'):>5s} "
          f"{t.get('predicted_per_second') or 0:7.2f} {(t.get('predicted_ms') or 0) / n:7.2f} "
          f"{t.get('prompt_per_second') or 0:7.1f} "
          f"{dn:7d} {da:7d} {100 * da / dn if dn else 0:5.1f}% "
          f"{n / steps if steps else 0:8.2f} {d.get('vram_peak_mib') or 0:6d}")
