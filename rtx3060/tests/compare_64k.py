"""Decode comparison at a fixed 64K filled context, identical prompt for both servers.

Tries to restore the cached KV first (so prefill isn't repeated); falls back to a real
prefill if the server rejects the cache file.

Usage: python tests/compare_64k.py --port 9931 --label stock-ngl48
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=9931)
P.add_argument('--label', required=True)
P.add_argument('--ctx-tokens', type=int, default=65536)
P.add_argument('--n-predict', type=int, default=128)
P.add_argument('--restore', default='')
A = P.parse_args()
BASE = f'http://127.0.0.1:{A.port}'


def api(path, payload=None, timeout=7200):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


source = (ROOT / 'tests/pride-and-prejudice.txt').read_text(encoding='utf-8-sig')
tokens = api('/tokenize', {'content': source, 'add_special': True})['tokens'][:A.ctx_tokens]

restored = False
if A.restore:
    try:
        api(f'/slots/0?action=restore', {'filename': A.restore})
        restored = True
        print(f'restored {A.restore}', flush=True)
    except urllib.error.HTTPError as e:
        print(f'restore rejected ({e.code}): {e.read()[:200]!r} -> will prefill', flush=True)

print(f'{A.label}: prompt {len(tokens):,} tokens, n_predict {A.n_predict}', flush=True)
t0 = time.time()
r = api('/completion', {'prompt': tokens, 'n_predict': A.n_predict, 'temperature': 0,
                        'top_k': 1, 'seed': 42, 'ignore_eos': True, 'cache_prompt': True})
wall = time.time() - t0
t = r['timings']
row = {'label': A.label, 'ctx': len(tokens), 'restored': restored,
       'prompt_n': t.get('prompt_n'), 'prompt_tps': round(t.get('prompt_per_second') or 0, 1),
       'tg_tps': round(t['predicted_per_second'], 2),
       'ms_per_token': round(1000 / t['predicted_per_second'], 2),
       'generated': t.get('predicted_n'), 'wall_s': round(wall, 1)}
out = ROOT / 'results' / f'cmp64k-{A.label}.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(row, indent=2))
print(json.dumps(row, indent=2), flush=True)
