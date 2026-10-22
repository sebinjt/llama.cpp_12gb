Set-Location C:\Users\sebinj\qwen_inference
foreach ($t in @(3,2)) {
  $c=@("--staged","12","--snapshot","kv-200k-code-task$t-mtp.bin","--skip-manifest","--suffix-file","results\dq4\task$t.suffix.txt","--n","512","--reps","1","--env","LLAMA_FA_VEC_MAX_NQ=128","--env","LLAMA_FA_DECODE_Q4=1","--env","LLAMA_RS_REPLAY=1","--env","LLAMA_MTP_WINDOW=8192","--draft-ubatch","16")
  foreach ($d in @(3,4,5)) { python tests\dq4_e2e.py @c --spec draft-mtp --draft-max $d --tag "A5-t$t-d$d" --out "results\dq4\A5-t$t-d$d.json" }
}
