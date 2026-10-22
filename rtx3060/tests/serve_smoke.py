"""Smoke-test a running server: restore a cached long prefill, then time a real decode
at that context length. Usage: python tests/serve_smoke.py [port] [ctx]"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
port = sys.argv[1] if len(sys.argv) > 1 else '8090'
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 98304
base = f'http://127.0.0.1:{port}'


def api(path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

# Find the cached prefill saved for this context size and read its token count from the name.
found = sorted((ROOT / 'results/.kv_cache').glob(f'kv-*-c{ctx}-q4_0-q4_0-p*.bin'))
if not found:
    print(f'no cached prefill for ctx={ctx} in results/.kv_cache; run the benchmark once to create it')
    raise SystemExit(1)
cache = found[-1].name
prompt_tokens = int(cache.rsplit('-p', 1)[1].split('.')[0])
try:
    api('/slots/0?action=restore', {'filename': cache})
    print(f'restored {cache}')
except Exception as exc:
    print(f'restore failed: {exc}')
    raise SystemExit(1)

tokens = api('/tokenize', {'content': (ROOT / 'tests/pride-and-prejudice.txt').read_text(encoding='utf-8-sig'),
                           'add_special': True})['tokens'][:prompt_tokens]
print(f'context: {len(tokens):,} tokens')

t0 = time.time()
r = api('/completion', {'prompt': tokens, 'n_predict': 32, 'temperature': 0, 'seed': 42,
                        'ignore_eos': True, 'cache_prompt': True})
t = r['timings']
print(f'wall={time.time() - t0:.1f}s  prompt_n={t["prompt_n"]}  predicted_n={t["predicted_n"]}')
print(f'decode={t["predicted_per_second"]:.2f} t/s  ({1000 / t["predicted_per_second"]:.1f} ms/token)')
print('output:', repr(r.get('content', '')[:300]))
