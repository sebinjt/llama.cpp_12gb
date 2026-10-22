"""Real llama-server benchmark; never a synthetic transport surrogate.
Example: python tests/run_staged_server.py --layers 8 --ctx 131072 --prompt-tokens 130560 --mtp 3
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser()
P.add_argument('--layers', type=int, default=8)
P.add_argument('--ctx', type=int, default=131072)
P.add_argument('--prompt-tokens', type=int, default=130560)
P.add_argument('--mtp', type=int, default=0)
P.add_argument('--draft-type-k', default='f16', help='MTP draft layer KV type (default f16 = upstream)')
P.add_argument('--draft-type-v', default='f16')
P.add_argument('--type-k', default='q4_0')
P.add_argument('--type-v', default='q4_0')
P.add_argument('--generate', type=int, default=128)
P.add_argument('--port', type=int, default=8097)
P.add_argument('--batch', type=int, default=512)
P.add_argument('--ubatch', type=int, default=128)
P.add_argument('--model', default='Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf')
P.add_argument('--reuse', action='store_true', help='second request reuses the long prefix')
P.add_argument('--budget-mib', type=int, default=0)
P.add_argument('--slots', type=int, default=4, help='device staging slot depth (LLAMA_KV_STAGE_SLOTS)')
P.add_argument('--kv-cache', action='store_true', help='reuse a disk prefill KV cache for this config (skips prefill)')
P.add_argument('--refresh-cache', action='store_true', help='redo prefill and overwrite the disk KV cache')
P.add_argument('--verbose', type=int, default=4)
A = P.parse_args()
placement = f'budget{A.budget_mib}' if A.budget_mib else str(A.layers)
name = f'staged-{placement}-s{A.slots}-c{A.ctx}-p{A.prompt_tokens}-{A.type_k}-{A.type_v}-mtp{A.mtp}'
if A.mtp and (A.draft_type_k, A.draft_type_v) != ('f16', 'f16'):
    name += f'-d{A.draft_type_k}-{A.draft_type_v}'
out = ROOT / 'results' / name
out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{A.port}'
def api(path, payload=None, timeout=2400):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def gpu_mem(pid):
    """Per-process WDDM dedicated vs shared GPU memory, in MiB."""
    ps = ("$i = Get-CimInstance Win32_PerfRawData_GPUPerformanceCounters_GPUProcessMemory | "
          f"Where-Object {{ $_.Name -like 'pid_{pid}_*' }}; "
          "$d = ($i | Measure-Object -Property DedicatedUsage -Sum).Sum; "
          "$s = ($i | Measure-Object -Property SharedUsage -Sum).Sum; "
          "'{0} {1}' -f [math]::Round($d/1MB,1), [math]::Round($s/1MB,1)")
    try:
        r = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                           capture_output=True, text=True, timeout=60)
        d, s = r.stdout.split()
        return {'dedicated_mib': float(d), 'shared_mib': float(s)}
    except Exception:
        return {}

book = ROOT / 'tests' / 'pride-and-prejudice.txt'
if not book.exists():
    with urllib.request.urlopen('https://www.gutenberg.org/cache/epub/1342/pg1342.txt', timeout=60) as response:
        book.write_bytes(response.read())
cmd = [str(ROOT / '../build-cuda/bin/llama-server.exe'), '-m', str(ROOT / A.model),
       '-c', str(A.ctx), '-ngl', '99', '-fa', 'on', '-ctk', A.type_k, '-ctv', A.type_v,
       '-b', str(A.batch), '-ub', str(A.ubatch), '-np', '1', '--fit', 'off',
       '--no-context-shift', '--ctx-checkpoints', '1', '--cache-ram', '0',
       '--host', '127.0.0.1', '--port', str(A.port), '--reasoning', 'off', '--log-verbosity', str(A.verbose)]
if A.mtp:
    cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', str(A.mtp),
            '--spec-draft-type-k', A.draft_type_k, '--spec-draft-type-v', A.draft_type_v]
# Disk KV cache: the KV content depends only on model + ctx + K/V types + prompt,
# not on how many layers are staged, so one cached prefill serves every placement.
slot_dir = ROOT / 'results' / '.kv_cache'
slot_dir.mkdir(parents=True, exist_ok=True)
cmd += ['--slot-save-path', str(slot_dir)]
cached = f'kv-{Path(A.model).stem}-c{A.ctx}-{A.type_k}-{A.type_v}-p{A.prompt_tokens}.bin'
have_cache = (slot_dir / cached).exists() and not A.refresh_cache
env = dict(os.environ)
for key in ('LLAMA_DUMP_KV', 'GGML_CUDA_PREFER_MMQ', 'LLAMA_KV_STAGE_TRACE', 'GGML_CUDA_DISABLE_GRAPHS'):
    env.pop(key, None)
env.pop('LLAMA_KV_GPU_BUDGET_MIB', None)
if A.budget_mib:
    env.pop('LLAMA_KV_STAGED_LAYERS', None)
    env['LLAMA_KV_GPU_BUDGET_MIB'] = str(A.budget_mib)
else:
    env['LLAMA_KV_STAGED_LAYERS'] = str(A.layers)
env['LLAMA_KV_STAGE_SLOTS'] = str(A.slots)
(out / 'command.json').write_text(json.dumps({'command': cmd, 'layers': A.layers, 'budget_mib': A.budget_mib}, indent=2))
log = open(out / 'server.log', 'wb')
proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}; see {out}/server.log')
        try:
            if api('/health', timeout=1).get('status') == 'ok': break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError('server startup timed out')
    print('Server ready:', name, flush=True)
    if A.prompt_tokens > 4096:
        source = book.read_text(encoding='utf-8-sig')
    else:
        source = ('Write a Python function that computes the nth Fibonacci number with matrix exponentiation. '
                  'Include a docstring, complexity, and unit tests.\n') * (A.prompt_tokens // 24 + 30)
    tokens = api('/tokenize', {'content': source, 'add_special': True})['tokens']
    if len(tokens) < A.prompt_tokens:
        raise RuntimeError(f'book has only {len(tokens)} tokens, need {A.prompt_tokens}')
    tokens = tokens[:A.prompt_tokens]
    # Prompt is a contiguous, real book excerpt, not random token IDs.
    (out / 'prompt-info.json').write_text(json.dumps({'source': str(book), 'tokens': len(tokens)}, indent=2))
    restored = False
    if have_cache:
        try:
            api('/slots/0?action=restore', {'filename': cached})
            restored = True
            print(f'Restored prefill KV from {cached} (prefill skipped)', flush=True)
        except Exception as exc:
            print(f'KV cache restore failed ({exc}); prefill will run', flush=True)
    payload = {'prompt': tokens, 'n_predict': A.generate, 'temperature': 0,
               'seed': 42, 'ignore_eos': True, 'cache_prompt': True}
    t0 = time.time()
    response = api('/completion', payload)
    response['test_wall_seconds'] = time.time() - t0
    response['kv_cache_restored'] = restored
    response['gpu_mem_mib'] = gpu_mem(proc.pid)
    (out / 'response.json').write_text(json.dumps(response, indent=2), encoding='utf-8')
    print(json.dumps(response.get('timings', {}), indent=2), flush=True)
    print('Output excerpt:', repr(response.get('content', '')[:240]), flush=True)
    if not have_cache:
        try:
            api('/slots/0?action=save', {'filename': cached})
            print(f'Saved prefill KV to {cached}', flush=True)
        except Exception as exc:
            print(f'KV cache save failed: {exc}', flush=True)
    if A.reuse:
        second = api('/completion', payload)
        (out / 'response-reuse.json').write_text(json.dumps(second, indent=2), encoding='utf-8')
        print('REUSE:', json.dumps(second.get('timings', {})), flush=True)
finally:
    proc.terminate()
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
    log.close()
