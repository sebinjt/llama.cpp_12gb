"""Per-step accounting for a decode run: matmul / FA / GDN / whole-graph / residual.

Reads a server log produced with GGML_MM_TIMING + GGML_FA_TIMING + GGML_GDN_TIMING + GGML_GRAPH_TIMING
and GGML_CUDA_DISABLE_GRAPHS=1, keeps only the decode phase (after the last big prefill pass) and
reports, per eval pass and per step, the summed GPU time of each timer plus the residual that the
whole-graph timer leaves over (norms, rope, elementwise, launches).

  python tests/summ_step.py <server.log> [--json out.json]
"""
import json
import re
import sys

mm = re.compile(r"^MMTIME ([\d.]+) name=(\S+) n=(\d+) K=(\d+)")
fa = re.compile(r"^FATIME ([\d.]+) n_kv=(\d+) n_q=(\d+) n_head=(\d+)")
gdn = re.compile(r"^GDNTIME ([\d.]+) name=(\S+) n_tok=(\d+) n_seq=(\d+) H=(\d+) S_v=(\d+)")
gt = re.compile(r"^GRAPHTIME ([\d.]+) nodes=(\d+) first=(\S+)")


def kind(name):
    if name == "output.weight":
        return "lm_head"
    if name.startswith("fc") or name.startswith("selector_hidden"):
        return "draft_head"
    if "conv_proj" in name:
        return "draft_conv"
    if "attn_inp_k_rot" in name:
        return "rot"
    if ".ssm_" in name:
        return "gdn_mm"
    if ".attn_" in name:
        return "attn_mm"
    if ".ffn_" in name:
        return "ffn_mm"
    return "other_mm"


def main():
    path = sys.argv[1]
    lines = open(path, errors="replace").read().splitlines()
    # decoding starts after the last prefill-sized op (n >= 16 columns); the server also does a small
    # warm-up eval before the prompt, so "first small op" is not a reliable marker
    dec_start = 0
    for i, l in enumerate(lines):
        m = mm.match(l)
        if m and int(m.group(3)) >= 16:
            dec_start = i + 1
        m = fa.match(l)
        if m and int(m.group(3)) >= 16:
            dec_start = i + 1
    tot_mm = tot_fa = tot_gdn = tot_graph = 0.0
    cls = {}
    steps = 0
    gdn_n = 0
    graph_calls = 0
    nodes_max = 0
    nq = []
    for l in lines[dec_start:]:
        m = mm.match(l)
        if m:
            ms, name, n = float(m.group(1)), m.group(2), int(m.group(3))
            tot_mm += ms
            k = kind(name)
            cls[k] = cls.get(k, 0.0) + ms
            if name == "blk.0.attn_qkv.weight":
                steps += 1
                nq.append(n)
            continue
        m = fa.match(l)
        if m:
            tot_fa += float(m.group(1)); continue
        m = gdn.match(l)
        if m:
            tot_gdn += float(m.group(1)); gdn_n += 1; continue
        m = gt.match(l)
        if m:
            ms, nd = float(m.group(1)), int(m.group(2))
            # decode-phase graph calls only: a decode eval's graph is far smaller than a prefill chunk
            if nd < 1200:
                tot_graph += ms; graph_calls += 1; nodes_max = max(nodes_max, nd)
            continue
    steps = max(steps, 1)
    print(f"log: {path}")
    print(f"decode: {steps} target passes, avg n={sum(nq)/len(nq):.2f}, graph calls {graph_calls} "
          f"(largest decode graph {nodes_max} nodes)")
    print(f"  per step (ms): matmul {tot_mm/steps:8.3f}  FA {tot_fa/steps:6.3f}  GDN {tot_gdn/steps:6.3f}  "
          f"graph-sum {tot_graph/steps:8.3f}")
    print(f"  matmul classes (ms/step):")
    for k in sorted(cls, key=lambda x: -cls[x]):
        print(f"    {k:11s} {cls[k]/steps:8.3f}")
    resid = (tot_graph - tot_mm - tot_fa - tot_gdn) / steps
    print(f"  residual (graph-sum - matmul - FA - GDN) = {resid:7.3f} ms/step "
          f"({100.0*resid/max(tot_graph/steps,1e-9):.1f}% of graph-sum)")
    if gdn_n:
        print(f"  GDN: {gdn_n/steps:.1f} ops/step, {tot_gdn/gdn_n:.4f} ms/op")
    if "--json" in sys.argv:
        json.dump({"steps": steps, "mm": tot_mm, "fa": tot_fa, "gdn": tot_gdn, "graph": tot_graph,
                   "cls": cls, "graph_calls": graph_calls},
                  open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
