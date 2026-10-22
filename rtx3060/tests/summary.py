"""Summarize results/*/response.json including per-process GPU dedicated/shared memory."""
import json
from pathlib import Path

rows = []
for d in sorted(Path('results').iterdir()):
    if not d.is_dir() or d.name.startswith('.'):
        continue
    f = d / 'response.json'
    if not f.exists():
        continue
    try:
        r = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    t = r.get('timings') or {}
    g = r.get('gpu_mem_mib') or {}
    rows.append((d.name, t.get('prompt_n'), t.get('prompt_per_second'),
                 t.get('predicted_n'), t.get('predicted_per_second'),
                 g.get('dedicated_mib'), g.get('shared_mib'), r.get('kv_cache_restored')))

w = max(len(r[0]) for r in rows)
fmt = f'{{:{w}}}  {{:>7}} {{:>7}} {{:>5}} {{:>7}} {{:>9}} {{:>10}} {{:>6}}'
print(fmt.format('run', 'pp_n', 'pp t/s', 'tg_n', 'tg t/s', 'VRAM MiB', 'shared MiB', 'cached'))
for name, pn, pp, gn, tg, ded, shr, cached in rows:
    print(fmt.format(name,
                     pn if pn is not None else '-',
                     f'{pp:.1f}' if pp else '-',
                     gn if gn is not None else '-',
                     f'{tg:.2f}' if tg else '-',
                     f'{ded:.0f}' if ded else '-',
                     f'{shr:.0f}' if shr else '-',
                     'yes' if cached else 'no'))
