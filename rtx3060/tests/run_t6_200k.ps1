# T6: 200K mode comparison over 5 task suffixes, both matmul routes. Restores the 200K base each run.
$ErrorActionPreference = "Continue"
Set-Location C:\Users\sebinj\qwen_inference
foreach ($s in @("s1","s2","s3","s4","s5")) {
  foreach ($m in @("mmvq","mmq")) {
    $env:GGML_CUDA_PREFER_MMQ = if ($m -eq "mmq") { "1" } else { "0" }
    $port = 10040 + [int]$s.Substring(1) * 2 + [int]($m -eq "mmq")
    python tests/t2_dflash.py --mode dflash --prompt-file "results/prof/t6_p200k_$s.txt" `
      --ctx 204800 --budget 1000 --drafts 7 --n 512 --draft-ubatch 16 --ub 128 `
      --snapshot t2-200k-base.bin --port $port `
      --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_FA_VEC_MAX_NQ=128 --env LLAMA_RS_REPLAY=1 `
      --env LLAMA_MTP_WINDOW=8192 --env LLAMA_MMVQ_BATCH_INVARIANT=1 --env LLAMA_MMVQ_UNPACK_ONCE=0 `
      --out "results/prof/t6_200k_${s}_${m}.json" --verbosity 2 *>> results/prof/t6_run.log
    "done $s $m" | Out-File -Append results/prof/t6_run.log
  }
}
"ALL DONE" | Out-File -Append results/prof/t6_run.log
