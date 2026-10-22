"""Compare decode throughput: stock llama.cpp (layer offload) vs staged host KV.

Sources:
  llama_cpp/sweep-mtp-*.txt, llama_cpp/boundary-mtp-*.txt   stock server sweeps (one ctx per file)
  logs/staged-serving/*.err.log                             our staged server, real served requests
"""
import glob
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# "eval time = 414.22 ms / 4 tokens (...)"  -- may be wrapped, so collapse whitespace first
EVAL = re.compile(r'eval time\s*=\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs)')
PLIVE = re.compile(r'task (\d+) \| n_gen =\s*(\d+), tg =\s*([\d.]+) t/s')
PROMPT = re.compile(r'task (\d+) \| new prompt.*task\.n_tokens = (\d+)')
NGL = re.compile(r'-ngl\s+(\d+)')
CTXARG = re.compile(r'(?:--ctx-size|-c)\s+(\d+)')


def collapse(text):
    return re.sub(r'\s+', ' ', text)


def stock_rows():
    out = []
    for f in sorted(glob.glob(str(ROOT / 'llama_cpp/sweep-mtp-*.txt')) +
                    glob.glob(str(ROOT / 'llama_cpp/boundary-mtp-*.txt'))):
        p = pathlib.Path(f)
        raw = p.read_text(encoding='utf-8', errors='replace')
        text = collapse(raw)
        # generation rates: the *last* eval time block is the final generation
        rates = []
        for ms, n in EVAL.findall(text):
            ms, n = float(ms), int(n)
            if n >= 20 and ms > 0:
                rates.append(n / (ms / 1000.0))
        if not rates:
            continue
        m = re.search(r'(\d+)$', p.stem)
        ctx = int(m.group(1)) if m else None
        c = CTXARG.search(text)
        if ctx is None and c:
            ctx = int(c.group(1))
        ngl = NGL.search(raw)
        kv = 'q8_0' if 'q8' in p.name else 'q4_0'
        out.append({'ctx': ctx, 'kv': kv, 'ngl': int(ngl.group(1)) if ngl else None,
                    'tg': max(rates), 'samples': len(rates), 'file': p.name})
    return sorted(out, key=lambda r: (r['kv'], r['ctx'] or 0))


def staged_rows():
    out = []
    for f in (glob.glob(str(ROOT / 'logs/staged-serving/*.err.log'))):
        p = pathlib.Path(f)
        raw = p.read_text(encoding='utf-8', errors='replace')
        prompt_of = {m[0]: int(m[1]) for m in PROMPT.findall(raw)}
        final = {}
        for task, n, tg in PLIVE.findall(raw):
            n, tg = int(n), float(tg)
            if task not in final or n > final[task][0]:
                final[task] = (n, tg)
        for task, (n, tg) in final.items():
            if n >= 20:
                out.append({'ctx': prompt_of.get(task, 0) + n, 'tg': tg, 'n': n,
                            'prompt': prompt_of.get(task)})
    return sorted(out, key=lambda r: r['ctx'])


stock = stock_rows()
print('=' * 76)
print('STOCK llama.cpp b11681 - layer offload (server sweeps in llama_cpp/)')
print('=' * 76)
print(f'{"file":40s} {"kv":>5s} {"ngl":>4s} {"ctx":>7s} {"tg t/s":>7s}')
for r in stock:
    print(f'{r["file"]:40s} {r["kv"]:>5s} {str(r["ngl"] or "-"):>4s} {str(r["ctx"]):>7s} {r["tg"]:>7.2f}')

staged = staged_rows()
print()
print('=' * 76)
print('STAGED host KV - real served requests (logs/staged-serving)')
print('=' * 76)
print(f'{"prompt":>8s} {"gen":>6s} {"ctx":>7s} {"tg t/s":>7s}')
for r in staged:
    print(f'{str(r["prompt"]):>8s} {r["n"]:>6d} {r["ctx"]:>7d} {r["tg"]:>7.2f}')
