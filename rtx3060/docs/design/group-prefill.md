# Offline layer-group prefill

The producer runs layers 0-31 over the entire prompt, retains the raw FP32 residual
stream in host RAM, then reloads the model with layers 32-63 resident and runs that
half over the boundary activations. Only the active half's attention KV is in VRAM.
Inactive weights and KV remain in host memory. Both halves use the existing MMA
prefill kernels. No staged-KV CUDA code is changed by this feature.

The final artifact is a standard llama.cpp sequence-state file: all 16 attention
layers' Q4_0 K/V, all 48 GDN layers' recurrent/conv state, position metadata and the
original token prefix. It can be loaded through the staged server's slot restore API.
It is a **target-only** cache. It contains no MTP draft state or speculative state.

## Build

From the repository root in a normal Windows terminal:

```powershell
cmd /c tests\build_group_prefill.bat
```

This builds `llama`, `llama-server`, and `group_prefill.exe` in the existing CUDA
build. It preserves the prior uncommitted staged-KV and MTP changes.

## Produce a cache

```powershell
python tests\prefill_cache.py --prompt big.prompt.txt --out results\.kv_cache\kv-200k-group-u2048-q4.bin --ctx 204800 --ubatch 2048 --save-nextn
```

The runner creates a log and JSON manifest beside the snapshot. The manifest records
the model size/mtime, prompt and snapshot hashes, executable/library hashes, timing,
and sampled peak GPU memory. Existing snapshots are not overwritten. Failed writes
use a `.partial` suffix; only a completed file is renamed to the requested name.
The binary hashes identify the producer; a decoder rebuild alone does not invalidate
the cache. Revalidate compatibility if KV representation or state semantics change.

By default, the producer caches all but the final prompt token. The server can then
consume that token to recover logits without rewinding GDN state or requiring a
server checkpoint sidecar. Use the exact original prompt when restoring.

`--tokens N` caches a shorter prefix for development. It leaves the remainder of the
original prompt uncached; this is not a claim that the whole original file was cached.

At 200K, boundary activations occupy about 3.9 GiB of host RAM. An additional roughly
4 GiB temporary sequence-state copy is needed during the model swap. Inactive KV and
the GGUF mapping also consume host memory. Intermediate state does not need disk space.

## Restore into the staged server

Launch the normal server with the same model, Q4_0 K/V, one sequence and sufficient
context. For the 200K layout, use vector quantized attention, nine staged layers and
two staging slots. Specify an absolute slot-save directory ending in a separator.

```python
requests.post(base + "/slots/0?action=restore", json={"filename": "kv-200k-group-u2048-q4.bin"}).raise_for_status()
response = requests.post(base + "/completion", json={
    "prompt": open("big.prompt.txt", encoding="utf-8", newline="").read(),
    "n_predict": 32, "temperature": 0, "cache_prompt": True,
}).json()
print(response["timings"])
```

The expected reused prefix is 204,364 tokens for the current `big.prompt.txt`; only
one prompt token should be processed. Restore itself is not prefill and must not be
reported as prefill throughput. Check `cache_n` and `prompt_n` on every experiment.
Preserve CRLF/LF when reading the prompt: normalizing line endings can change the
token sequence and invalidate prefix reuse.

## Retain target hidden states for MTP

Add `--save-nextn` to the Python producer to retain the normalized target hidden
state at every cached token, without repeating the target pass later. The companion
`SNAPSHOT.nextn.f32` is about 3.9 GiB at 200K. It is not an initialized MTP draft cache;
a separate one-layer MTP initialization step is still needed before measuring
restored MTP acceptance.

The companion has a 24-byte little-endian header: magic `QNEXTH01` (8 bytes), row
count (u64), row width (u32), and GGML type (u32, zero for F32). Row-major FP32 values
follow. Rows are h[0] through h[N-1], not shifted; the MTP input for token i pairs
that token with h[i-1], with the normal initial carry state at i=0. The manifest
records the companion path, size and SHA256.

The short regression also checks that this entire hidden-state companion is
byte-identical between split and ordinary full-model forwards.

## Validation

```powershell
python tests\check_group_prefill.py
python tests\check_group_cache_server.py --state results\.kv_cache\kv-200k-group-u2048-q4.bin --output results\group-prefill-200k-u2048-check
```

The short check compares the *entire saved snapshot* against normal full-model
prefill at the same microbatch size, then compares restored logits and 16 greedy
tokens with zero and eight staged layers. The long check compares restored logits
and 16 tokens at nine and ten staged layers and verifies reuse through the actual
server slot endpoint.

The first implementation used the ordinary embedding-input path for the second
group. That incorrectly treated text activations as multimodal embeddings, changing
M-RoPE coordinates and reading a position layout unsuitable for that batch. The
corrected input carries original token IDs and text positions alongside a dedicated
raw residual-stream input. At 2,048 tokens and ubatch 128, the split and full snapshots
have identical SHA256 `ae2e06d898d79e57971048b24572553880b00b8dc2362f9b0d7814ba6530868f`.
All 248,320 restored logits and all 16 generated tokens match exactly, including
restoration with eight staged layers (`results/group-prefill-check-v2/summary.json`).

A second regression checks a 2,048-token batch plus a three-token tail. This caught
an unanchored `output.*` override that also matched `attn_output.weight`. It put
attention projections in host memory and selected CPU math for tiny batches. The
override is now `^output\.weight$`, and the 2,051-token split/full snapshots and
hidden-state histories match byte for byte, with identical restored logits/tokens
(`results/group-prefill-check-1792405238/summary.json`). `_cfg.py` uses the anchored
pattern, and `serve.ps1` expands the `output=CPU` shorthand to that exact pattern.

Both recorded 200K runs predate that placement correction. Their final batches
contain 76 and 1,612 tokens, respectively, so the affected projections used GPU
math, as in the passing full-batch controls. Those artifacts pass direct consumer
checks. The placement-only correction was tested with the short boundary regression;
the full 200K timing was not repeated after it.

## Implementation boundaries

- The private `LLAMA_PREFILL_LAYER_RANGE=begin:end` environment option is read at
  context construction. Without it, the model runs its normal full graph.
- Only the dense 64-layer Qwen35 model, one sequence, Q4_0 K/V and splits on complete
  four-layer blocks are supported by the producer.
- Initial group state is saved before unloading its context/model. On the second
  group, restoring that state and clearing metadata without clearing tensor data
  preserves the completed layers while rebuilding identical positions.
- The second group is prefill-only. Do not use a partial context for generation.
  Reload the final snapshot in a normal full-model context.
- No upstream submission, commit or push is part of this work.
