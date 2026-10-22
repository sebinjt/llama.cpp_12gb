Set-Location C:\Users\sebinj\qwen_inference
$e=@("--n","512","--reps","1","--env","LLAMA_FA_VEC_MAX_NQ=128","--env","LLAMA_FA_DECODE_Q4=1","--env","LLAMA_RS_REPLAY=1","--env","LLAMA_MTP_WINDOW=8192","--env","LLAMA_MMVQ_BATCH_INVARIANT=1","--env","LLAMA_MMVQ_CHUNKED_FALLBACK=1","--draft-ubatch","16","--snapshot","kv-200k-code-u2048-q4-mtp.bin","--verbosity","4")
foreach ($L in @(11,12)) {
  foreach ($s in @("task3","task3long")) {
    python tests\dq4_e2e.py --staged $L @e --suffix-file "results\dq4\$s.suffix.txt" --spec draft-mtp --draft-max 4 --tag "B3-$s-L$L-d4" --out "results\dq4\B3-$s-L$L-d4.json"
  }
}
python tests\dq4_e2e.py --staged 11 @e --suffix-file "results\dq4\task3long.suffix.txt" --tag "B3-task3long-L11-nomtp" --out "results\dq4\B3-task3long-L11-nomtp.json"
python tests\dq4_e2e.py --staged 12 @e --suffix-file "results\dq4\task3long.suffix.txt" --spec draft-mtp --draft-max 4 --server-arg=-ub --server-arg=256 --server-arg=-b --server-arg=512 --n 64 --tag "B3-task3long-L12-d4-ub256" --out "results\dq4\B3-task3long-L12-d4-ub256.json"
python tests\dq4_e2e.py --staged 11 @e --suffix-file "results\dq4\task3long.suffix.txt" --spec draft-mtp --draft-max 4 --server-arg=-ub --server-arg=256 --server-arg=-b --server-arg=512 --n 64 --tag "B3-task3long-L11-d4-ub256" --out "results\dq4\B3-task3long-L11-d4-ub256.json"
