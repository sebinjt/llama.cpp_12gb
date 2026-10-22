import json, sys, glob, os
rows = []
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/w2/s106_*.json'
for f in sorted(glob.glob(pat)):
    d = json.load(open(f))
    for k, r in d['runs'].items():
        t = r.get('timings') or {}
        if not t:
            continue
        st = t.get('draft_verification_steps') or 1
        rows.append(dict(run=os.path.basename(f)[:-5], m=k,
                         tps=t['predicted_per_second'],
                         msp=t['predicted_ms'] / st,
                         tks=t['predicted_n'] / st,
                         draft=t.get('draft_n'), acc=t.get('draft_n_accepted'), steps=st,
                         tok=r['n_tokens'],
                         sha=r['sha'], vload=d.get('vram_after_load'), vpk=r.get('vram_peak')))
hdr = f"{'run':<34}{'m':>4}{'t/s':>7}{'ms/st':>8}{'tok/st':>8}{'drf':>5}{'acc':>4}{'stp':>5}{'tok':>5}  {'sha':<18}{'vload':>6}{'vpk':>6}"
print(hdr)
for r in rows:
    print(f"{r['run']:<34}{r['m']:>4}{r['tps']:>7.2f}{r['msp']:>8.1f}{r['tks']:>8.3f}"
          f"{str(r['draft']):>5}{str(r['acc']):>4}{r['steps']:>5}{r['tok']:>5}  {r['sha']:<18}"
          f"{str(r['vload']):>6}{str(r['vpk']):>6}")
