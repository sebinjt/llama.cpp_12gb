Set-Location C:\Users\sebinj\qwen_inference
$base=@("--staged","0","--ctx","16384","--prompt-file","results\dq4\code_short.prompt.txt","--snapshot","dq4-short-q4-mtp.bin","--n","512","--reps","1")
$cfgs=@{ "dq4"=@("--env","LLAMA_FA_DECODE_Q4=1"); "rpl"=@("--env","LLAMA_RS_REPLAY=1"); "none"=@(); "vecmax8"=@("--env","LLAMA_FA_VEC_MAX_NQ=8") }
foreach ($k in $cfgs.Keys) {
  $e=$cfgs[$k]
  python tests\dq4_e2e.py @base @e --tag "A-$k-nomtp" --out "results\dq4\A-$k-nomtp.json"
  foreach ($d in @(3,4)) { python tests\dq4_e2e.py @base @e --spec draft-mtp --draft-max $d --tag "A-$k-d$d" --out "results\dq4\A-$k-d$d.json" }
}
