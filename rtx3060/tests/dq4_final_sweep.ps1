# Final sweep: 2 coding tasks x 512 tokens on restored 200K snapshots (see comm\files\seq34_final.txt)
Set-Location C:\Users\sebinj\qwen_inference
foreach ($t in @(2,3)) {
  $c = @("--snapshot","kv-200k-code-task$t-mtp.bin","--skip-manifest","--suffix-file","results\dq4\task$t.suffix.txt","--n","512","--reps","1","--env","LLAMA_FA_VEC_MAX_NQ=128")
  python tests\dq4_e2e.py --staged 9 @c --tag "F-t$t-nomtp-vec-L9" --out "results\dq4\F-t$t-nomtp-vec-L9.json"
  python tests\dq4_e2e.py --staged 9 @c --env LLAMA_FA_DECODE_Q4=1 --tag "F-t$t-nomtp-dq4-L9" --out "results\dq4\F-t$t-nomtp-dq4-L9.json"
  python tests\dq4_e2e.py --staged 14 @c --spec draft-mtp --draft-max 3 --tag "F-t$t-mtp-flagsoff-L14-d3" --out "results\dq4\F-t$t-mtp-flagsoff-L14-d3.json"
  foreach ($d in @(2,3,4,5)) {
    python tests\dq4_e2e.py --staged 11 @c --spec draft-mtp --draft-max $d --draft-ubatch 16 --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_RS_REPLAY=1 --env LLAMA_MTP_WINDOW=8192 --tag "F-t$t-best-L11-d$d" --out "results\dq4\F-t$t-best-L11-d$d.json"
  }
}
