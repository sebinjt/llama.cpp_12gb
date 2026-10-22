# Temporal view of one decode token — why components ADD instead of overlap

Source: nsys GPU traces of Qwen3.8-27B-GSQ-RCO-IQ2_S (ngl 99, q8_0 KV, flash-attn, no MTP)
- 4K ctx (llama-cli):  nsys_cli.nsys-rep            (decode ~31 ms/token)
- 72K ctx / 32K real (llama-cli): nsys_32k.nsys-rep (decode ~62-72 ms/token)
- 32K core path (llama-bench):    nsys_bench32.nsys-rep (no server wrapper)

## Per-token timeline (32K attention, llama-cli/server path; ~62-72 ms wall, ~42-44 ms GPU busy = ~60% duty)

Class key: M=weight-stream matmul  A=flash-attention  D=KV/q8 dequant  G=GDN recurrence
           S=small ops (norm/silu/quant/rope)  C=copies/syncs  .=GPU idle

compute block of one token (from token_32k.svg / ASCII, ~1 char = 0.35 ms):

 42.0| MMMMMSMMM.........D AAMMMMMMMMMMMMMDDAMMMMMMMMMMMMM.DAAMMMMMSMMMMMMDDASMMMMMMMMMMM.DAAMM.MMMMMMMMMMDAAMM.MMMSMMMMM |
 84.0| .DAGMM.MMMMMMMMM.DDAMMMSMMMMMSMMMDDAMMM.MMSMMMMMMDDAMMMMMM_MMMMMMDDA.MMMMMMMMMSMMMDDAMMMMMM.MMMSMMDDAMMMMMMSMMMMMMDDAMMM |
126.0| .MMMMMMSMMMDDAMMMMMMMMM_MMMDDAMMMMMMMC..............

read: one large M-blob (several layers of weight-stream matmul), then right before every attention
layer a D (KV q8_0 dequant) + A (flash attention) pair; then more M-blocks. 16 D/A slots in the
token, spaced ~3.5-4 ms = every 4th of the 64 layers = the 16 full-attention layers.

4K token for contrast (mostly M-blocks, attention invisible: A total = 0.04-0.35 ms/token):

  0.00| DDDMMMMSMMSGD.DDDMMMDDDMMMMM.DMMMMMMM.MDDDMMMMSMMSDMMMSMMMSGMMSMMMMM... (M 32.8ms, A 0.35ms, G 2.0ms, D 4.5ms)

## Budget of the ~62-72 ms/token at 32K (16-token aggregate, lm-server path)

  component                                  GPU ms  % of busy   note
  weight-stream matmuls (M)                   24.0      58%     48 GDN ffn_up/gate/down + 16 attn q/k/v/o + lm_head
  flash attention (A)                          6.8      16%     reads ALL 32K cells of K,V (per token)
  KV q8_0 dequant (D)                          2.0       5%     per-block scale+shift while reading KV
  D2H logits + small copies (C)                3.1       7%     llama.cpp output sync to host
  norms/silu/quant in+bias (S)                 5.0      12%     ~2600 ops, each tiny but serialized
  GDN recurrence (G)                           1.0       2%     fused gdn_single + conv — cheap
  -------------------------------------------------------------
  GPU busy                                    41.7     100%     (~= 9.9 GiB/token at ~190 GB/s effective)
  host idle / driver bubbles               ~20-30            GPU has nothing to do: single-stream serialization
                                                                + server data marshalling + per-launch overhead

  llama-bench (no server) at same 32K ctx: same compute but ONLY ~21 MB / 2 ms of copies in the tail
  (vs server path ~140 MB D2H) -> the big copy bursts are llama-server/wrapper data-delivery, not model math.

## Why it is sequential, not parallel

1. Single CUDA compute stream. llama.cpp keeps one stream per backend; kernels are enqueued in graph
   order — no concurrent kernel overlap by design.
2. Hard dependency chain per token: layer i+1's matmuls consume layer i's output; attention at layer i
   consumes the K/V just produced by the projections of THAT layer. Nothing to run ahead of itself.
3. Even "independent" chunks (e.g. rms_norm of next block, silu/activation of a finished gemm) are
   tiny (~10-60 us) and serialized behind the previous one — they ADD ~5 ms in aggregate rather than
   hiding under the big matmuls.
4. Attention and weights both consume the SAME DRAM. "Hiding" attention under weight streaming does
   not reduce the 9.9 GiB/token read; on a shared-bus system overlap can only fill *idle bubbles*, it
   cannot make a bandwidth-bound sum cheaper. That's why the components genuinely ADD.
5. The residual ~40% idle is host/driver-side: per-token launch + WDDM submission + llama-server
   D2H marshalling + auto-regressive fashion (token t+1 cannot start until t is sampled) -> idle
   bubbles at token boundaries. This is the only part overlap could reclaim, and it's host, not GPU.

## Why 16.4 instead of "30"

- 30 assumed 12 GiB @ 360 GB/s: real file 8.61 GiB -> 41.8 t/s ceiling; achieved ~190 GB/s effective
  on M=1 low-bit mmvq -> ~24 ms/token just for weights.
- + ~11 ms attention+KV at 32K (serial, shared DRAM), + ~7 ms small-ops+copies, + host bubbles.
- Sum = ~62 ms/token = 16 t/s. The GPU is DRAM-shared + host-paced, not free of losses.

Artifacts: token_32k.svg, token_4k.svg, timeline.py, nsys_32k.nsys-rep, nsys_cli.nsys-rep, nsys_bench32.nsys-rep