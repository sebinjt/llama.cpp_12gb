"""Baseline: stock llama.cpp (llama_cpp/) with simple layer offloading, same model,
ctx, KV types and MTP settings as serve.ps1. Measures decode at a long context.

Usage: python tests/run_stock_server.py --ngl 61 --ctx 98304
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser()
P.add_argument('--ngl', type=int, default=61)
P.add_argument('--ctx', type=int, default=98304)
P.add_argument('--prompt-tokens', type=int, default=96256)
P.add_argument('--mtp', type=int, default=2)
P.add_argument('--generate', type=int, default=128)
P.add_argument('--port', type=int, default=9931)
P.add_argument('--type-k', default='q4_0')
P.add_argument('--type-v', default='q4_0')
A = P.parse_args()

out = ROOT / 'results' / f'stock-ngl{A.ngl}-c{A.ctx}-mtp{A.mtp}'
out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{A.port}'


def api(path, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


cmd = [str(ROOT / 'llama_cpp/llama-server.exe'), '-m', str(ROOT / '..' / 'Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'),
       '-c', str(A.ctx), '-ngl', str(A.ngl), '-fa', 'on', '-ctk', A.type_k, '-ctv', A.type_v,
       '-b', '512', '-ub', '128', '-np', '1', '--fit', 'off', '--no-context-shift',
       '--cache-ram', '0', '--jinja', '--reasoning', 'off', '--host', '127.0.0.1', '--port', str(A.port),
       '--log-verbosity', '4', '--ctx-checkpoints', '1', '--slot-save-path', str(ROOT / 'results/.kv_cache')]
if A.mtp:
    cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(A.mtp),
            '--spec-draft-type-k', A.type_k, '--spec-draft-type-v', A.type_v]
env = dict(os.environ)
for k in ('LLAMA_KV_STAGED_LAYERS', 'LLAMA_KV_STAGE_SLOTS', 'LLAMA_KV_STAGE_NODMA'):
    env.pop(k, None)
(out / 'command.json').write_text(json.dumps({'command': cmd}, indent=2))
log = open(out / 'server.log', 'wb')
proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}; see {out}/server.log')
        try:
            if api('/health', timeout=1).get('status') == 'ok':
                break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError('startup timed out')
    print(f'[{time.strftime("%H:%M:%S")}] server ready (ngl={A.ngl})', flush=True)

    source = (ROOT / 'tests/pride-and-prejudice.txt').read_text(encoding='utf-8-sig')
    tokens = api('/tokenize', {'content': source, 'add_special': True})['tokens'][:A.prompt_tokens]
    print(f'[{time.strftime("%H:%M:%S")}] prefilling {len(tokens):,} tokens (this is the slow part)', flush=True)
    t0 = time.time()
    r = api('/completion', {'prompt': tokens, 'n_predict': A.generate, 'temperature': 0,
                            'seed': 42, 'ignore_eos': True, 'cache_prompt': True})
    t = r['timings']
    r['wall'] = time.time() - t0
    (out / 'response.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(f'[{time.strftime("%H:%M:%S")}] pp={t["prompt_per_second"]:.1f} t/s  '
          f'tg={t["predicted_per_second"]:.2f} t/s  ({1000/t["predicted_per_second"]:.1f} ms/token)', flush=True)
    print('output:', repr(r.get('content', '')[:200]), flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
