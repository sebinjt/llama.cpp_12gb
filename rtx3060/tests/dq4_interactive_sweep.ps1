# Interactive sweep: restore the 200K target+MTP snapshot, prefill a fresh task suffix (226 tokens), generate 512.
Set-Location C:\Users\sebinj\qwen_inference
python tests\dq4_e2e.py --staged 11 --snapshot kv-200k-code-u2048-q4-mtp.bin --suffix-file results\dq4\task3.suffix.txt --spec draft-mtp --draft-max 4 --draft-ubatch 16 --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_RS_REPLAY=1 --env LLAMA_MTP_WINDOW=8192 --env LLAMA_FA_VEC_MAX_NQ=128 --n 32 --reps 1 --tag "I-probe-L11"
foreach ($t in @(3,2)) {
  foreach ($L in @(12,13)) {
    foreach ($d in @(3,4,5)) {
      python tests\dq4_e2e.py --staged $L --snapshot kv-200k-code-u2048-q4-mtp.bin --suffix-file "results\dq4\task$t.suffix.txt" --spec draft-mtp --draft-max $d --draft-ubatch 16 --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_RS_REPLAY=1 --env LLAMA_MTP_WINDOW=8192 --env LLAMA_FA_VEC_MAX_NQ=128 --n 512 --reps 1 --tag "I-t$t-L$L-d$d" --out "results\dq4\I-t$t-L$L-d$d.json"
    }
  }
  python tests\dq4_e2e.py --staged 9 --snapshot kv-200k-code-u2048-q4-mtp.bin --suffix-file "results\dq4\task$t.suffix.txt" --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_FA_VEC_MAX_NQ=128 --n 512 --reps 1 --tag "I-t$t-nomtp-dq4-L9" --out "results\dq4\I-t$t-nomtp-dq4-L9.json"
}
