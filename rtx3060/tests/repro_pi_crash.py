"""Reproduce the pi-agent crash: a slot holding a large cached prompt, then a request
with a different ~6.5K-token prompt (forces seq_rm of the whole cache + prefill).

Usage: python tests/repro_pi_crash.py [port]
Writes repro.log; exits non-zero if the server dies."""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
port = sys.argv[1] if len(sys.argv) > 1 else '9931'
base = f'http://127.0.0.1:{port}'
open('repro.log', 'w').close()


def log(msg):
    print(msg, flush=True)
    with open('repro.log', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def api(path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def alive():
    try:
        return api('/health', timeout=5).get('status') == 'ok'
    except Exception:
        return False


# 1. Fill the slot with a large cached prompt.
fresh = len(sys.argv) > 2 and sys.argv[2] == 'fresh'
caches = sorted((ROOT / 'results/.kv_cache').glob('kv-*-c98304-q4_0-q4_0-p*.bin'))
if caches and not fresh:
    api('/slots/0?action=restore', {'filename': caches[-1].name})
    log(f'restored {caches[-1].name}')

# 2. A different prompt of pi's size -> seq_rm of everything cached.
text = (ROOT / 'tests/pride-and-prejudice.txt').read_text(encoding='utf-8-sig')
other = text[300000:360000]                       # a different slice, not a prefix
tokens = api('/tokenize', {'content': other, 'add_special': True})['tokens'][:6557]
log(f'new prompt: {len(tokens)} tokens (pi sent 6557)')

if fresh:
    # No state file involved: build a large slot the ordinary way first.
    warm = api('/tokenize', {'content': text[:400000], 'add_special': True})['tokens'][:20000]
    log(f'warming slot with {len(warm)} tokens (no restore) ...')
    api('/completion', {'prompt': warm, 'n_predict': 8, 'temperature': 0, 'ignore_eos': True})
    log('warm-up done')

log('POST /v1/chat/completions stream=true ...')
t0 = time.time()
req = urllib.request.Request(
    base + '/v1/chat/completions',
    json.dumps({'messages': [{'role': 'user', 'content': other[:4000] + '\n\nmake an svg pelican'}],
                'max_tokens': 256, 'stream': True, 'temperature': 0}).encode(),
    {'Content-Type': 'application/json'})
chunks = 0
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            chunks += 1
            if chunks == 1:
                log(f'  first chunk after {time.time() - t0:.1f}s')
except Exception as exc:
    log(f'STREAM FAILED after {chunks} chunks / {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}')

log(f'stream done: {chunks} chunks in {time.time() - t0:.1f}s')
time.sleep(2)
ok = alive()
log(f'server alive: {ok}')
sys.exit(0 if ok else 1)
