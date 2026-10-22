#!/bin/bash
# worker2 seq106 runner. Locks are held by the caller for the whole batch; each run records nvidia-smi first.
# usage: w2_run106.sh OUT PROMPT CTX MODE DRAFTS STAGED|B:BUDGET SNAP|none BLOCKS PORT [extra --env ...]
set -u
cd /c/Users/sebinj/qwen_inference || exit 1
OUT="$1"; PROMPT="$2"; CTX="$3"; MODE="$4"; DRAFTS="$5"; STAGED="$6"; SNAP="$7"; BLOCKS="$8"; PORT="$9"
shift 9
COMMON="LLAMA_FA_DECODE_Q4=1 LLAMA_FA_VEC_MAX_NQ=128 LLAMA_RS_REPLAY=1 LLAMA_MTP_WINDOW=8192 LLAMA_MMVQ_BATCH_INVARIANT=1 LLAMA_MMVQ_UNPACK_ONCE=0 LLAMA_STEP_TIMING=1 GGML_CUDA_PREFER_MMQ=1 LLAMA_W_STAGE_ENABLE=1 LLAMA_W_STAGE_SLOT_MIB=48 LLAMA_W_STAGE_PREFETCH=1"
if [ "$BLOCKS" != "none" ]; then COMMON="$COMMON LLAMA_W_STAGE_BLOCKS=$BLOCKS"; fi
ARGS=()
for e in $COMMON; do ARGS+=(--env "$e"); done
for e in "$@"; do ARGS+=(--env "$e"); done

STAGE_ARGS=()
case "$STAGED" in
  B:*) STAGE_ARGS+=(--budget "${STAGED#B:}"); STAGE_ARGS+=(--staged 0);;
  *)   STAGE_ARGS+=(--staged "$STAGED");;
esac
SNAP_ARGS=()
if [ "$SNAP" != "none" ]; then SNAP_ARGS+=(--snapshot "$SNAP"); fi
OT_ARGS=()
if [ "$BLOCKS" != "none" ]; then
  IFS=',' read -ra L <<< "$BLOCKS"
  PAT=$(IFS='|'; echo "${L[*]}")
  OT_ARGS+=(--ot "^blk\.($PAT)\.ffn_down\.weight=CPU")
fi

{
  echo "=== seq106 run start $(date -Iseconds)"
  echo "lock: $(cat comm/GPU_LOCK 2>/dev/null || echo 'NOT HELD')"
  echo "compute-apps before: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')[end]"
  echo "mem before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
} > "results/w2/$OUT.lock.txt"

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && { echo "GPU BUSY - refusing"; exit 9; }

python tests/w2_dflash.py --mode "$MODE" --prompt-file "$PROMPT" --ctx "$CTX" \
  "${STAGE_ARGS[@]}" "${SNAP_ARGS[@]}" --drafts "$DRAFTS" --n "${W2N:-128}" --ub 32 --b 512 \
  --draft-ubatch 16 --draft-ngl 99 --output CUDA0 \
  --model-path "D:/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf" \
  "${OT_ARGS[@]}" "${ARGS[@]}" --vram --verbosity "${W2VERB:-2}" \
  --out "results/w2/$OUT.json" --port "$PORT" 2>&1 | tee "results/w2/$OUT.run.log"
echo "=== run end $(date -Iseconds); mem after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> "results/w2/$OUT.lock.txt"
