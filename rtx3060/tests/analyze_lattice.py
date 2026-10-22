"""Offline analysis of the DFlash selector lattice vs what the target accepted.

Reads a server log produced with LLAMA_DFLASH_LATTICE=1 (LATP = lattice rows, LACC = target token vs
draft token per verified row) and answers, per step:
  - how many draft tokens were accepted, and where the first mismatch was;
  - at that mismatch position, is the target's token among the lattice's top-k candidates, and at what rank
    (rank 0 = what the greedy walk took, so rank>=1 is a candidate a second path could have used);
  - is the mismatch position the least-confident position of the greedy path (smallest top1-top2 margin),
    which is the position the B=2 policy would branch at;
  - what the B=2 path would be (diverge at the least-confident position, take the 2nd candidate there,
    then follow argmax conditioned on the new predecessor) and whether a 3rd candidate would have helped.

Usage: python tests/analyze_lattice.py <server.log> [--min-steps N]
"""
import re
import sys
from collections import Counter, defaultdict

lat_pat = re.compile(r"^LATP (\d+) (\d+) (\d+) ids(.*?) sc(.*)$")
acc_pat = re.compile(r"^LACC (\d+) (\d+) (.*)$")


def parse(path):
    lat = defaultdict(dict)
    acc = {}
    for line in open(path, errors="replace"):
        m = lat_pat.match(line.strip())
        if m:
            step, pos, tk = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ids = [int(x) for x in m.group(4).split(",") if x]
            sc = [float(x) for x in m.group(5).split(",") if x]
            assert len(sc) == tk * tk, (len(sc), tk)
            lat[step][pos] = (ids, [sc[p * tk:(p + 1) * tk] for p in range(tk)])
            continue
        m = acc_pat.match(line.strip())
        if m:
            step, n = int(m.group(1)), int(m.group(2))
            rows = []
            for tok in m.group(3).split():
                a, b = tok.split(",")
                rows.append((int(a), int(b)))
            acc[step] = rows
    return lat, acc


def main():
    path = sys.argv[1]
    lat, acc = parse(path)
    steps = sorted(set(acc) & set(lat))
    print(f"steps with both dumps: {len(steps)}")

    full = 0                    # drafted tokens all accepted (n_draft accepted)
    fail_rows = Counter()       # accepted count -> steps
    mismatch_rank = Counter()   # rank of the target's token among the lattice candidates at the mismatch pos
    hedge_possible = 0          # mismatch where a rank>=1 candidate matches the target token
    lowconf_hit = 0             # mismatch position == least-confident position of the greedy path
    third_would_help = 0        # mismatch where only rank>=2 matches
    margin_rank = []            # (margin of the chosen, margin of the hedge candidate) for failing steps
    path1_tokens = 0

    for st in steps:
        rows = acc[st]
        n_draft = sum(1 for _, d in rows if d >= 0)
        # first mismatch row
        f = None
        for i, (s, d) in enumerate(rows):
            if d < 0:
                break
            if s != d:
                f = i
                break
        if f is None:
            full += 1
            continue
        fail_rows[f] += 1
        pos = f + 1                      # block position of the mismatching draft token
        if pos not in lat[st]:
            continue
        ids, scores = lat[st][pos]
        target = rows[f][0]
        # rank of the target token in the candidate list (conditioned on the greedy predecessor)
        rank = ids.index(target) if target in ids else None
        if rank is not None and rank >= 1:
            hedge_possible += 1
            if rank >= 2:
                third_would_help += 1
        mismatch_rank[rank] += 1

        # least-confident position along the greedy path: margin of the greedy choice vs the runner-up,
        # using the scores conditioned on the greedy predecessor chain
        pred = 0
        margins = {}
        for p in sorted(lat[st]):
            ids_p, sc_p = lat[st][p]
            row = sc_p[pred]
            top = sorted(range(len(row)), key=lambda k: -row[k])[:2]
            margins[p] = row[top[0]] - row[top[1]]
            pred = top[0]
        if margins:
            lp = min(margins, key=lambda p: margins[p])
            if lp == pos:
                lowconf_hit += 1
            margin_rank.append((margins[lp], margins.get(pos, float("nan"))))
            # B=2 path: diverge at the least-confident position, take the runner-up, then follow argmax
            pred = 0
            toks = []
            for p in sorted(lat[st]):
                ids_p, sc_p = lat[st][p]
                row = sc_p[pred]
                order = sorted(range(len(row)), key=lambda k: -row[k])
                pick = order[1] if p == lp else order[0]
                toks.append(ids_p[pick])
                pred = pick
            path1_tokens += len(toks)

    nfail = sum(fail_rows.values())
    print(f"\nsteps: {len(steps)}  full accept: {full} ({100.0*full/len(steps):.1f}%)  "
          f"with a mismatch: {nfail}")
    print("first-mismatch row distribution:", dict(sorted(fail_rows.items())))
    print(f"\nAt the first mismatch, the target's token in the lattice candidates of that position:")
    for r in sorted([k for k in mismatch_rank if k is not None]) + [None]:
        c = mismatch_rank.get(r, 0)
        label = f"rank {r}" if r is not None else "not in top-k"
        print(f"  {label:12s} {c:4d} steps ({100.0*c/max(nfail,1):5.1f}% of failures)")
    print(f"\nB=2 hedge (a rank>=1 candidate equals the target token at the mismatch position):")
    print(f"  {hedge_possible}/{nfail} failing steps ({100.0*hedge_possible/max(nfail,1):.1f}%) -> "
          f"a second path would have accepted at least one more token in those steps")
    print(f"  of those, {third_would_help} needed rank>=2, i.e. a third candidate would also have helped")
    print(f"\nPolicy check (branch at the least-confident position):")
    print(f"  the mismatch position WAS the least-confident position in {lowconf_hit}/{nfail} "
          f"({100.0*lowconf_hit/max(nfail,1):.1f}%) of failures")
    if margin_rank:
        avg_lp = sum(m[0] for m in margin_rank) / len(margin_rank)
        avg_mm = sum(m[1] for m in margin_rank if m[1] == m[1]) / max(len([m for m in margin_rank if m[1] == m[1]]), 1)
        print(f"  mean margin at the least-confident position {avg_lp:.3f}, at the mismatch position {avg_mm:.3f}")
    print(f"\nB=2 upper-bound on tokens/step: draft {path1_tokens/max(len(steps),1):.2f} tokens per step in path 1, "
          f"a full accept adds 1 bonus; compare with the current accepted tokens/step.")


if __name__ == "__main__":
    main()
