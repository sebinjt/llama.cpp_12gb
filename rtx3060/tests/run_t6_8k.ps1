# T6: 8K mode comparison over 5 task prompts, both matmul routes (fresh prefill each run).
Set-Location C:\Users\sebinj\qwen_inference
foreach ($q in 1..5) {
  foreach ($m in @("mmvq","mmq")) {
    $env:GGML_CUDA_PREFER_MMQ = if ($m -eq "mmq") { "1" } else { "0" }
    $port = 10070 + $q * 2 + [int]($m -eq "mmq")
    python tests/t2_dflash.py --mode dflash --prompt-file "results/prof/t6_8k_q$q.txt" `
      --ctx 16384 --drafts 7 --n 512 --draft-ubatch 16 --ub 128 --port $port `
      --env LLAMA_FA_DECODE_Q4=1 --env LLAMA_FA_VEC_MAX_NQ=128 --env LLAMA_RS_REPLAY=1 `
      --env LLAMA_MTP_WINDOW=8192 --env LLAMA_MMVQ_BATCH_INVARIANT=1 --env LLAMA_MMVQ_UNPACK_ONCE=0 `
      --out "results/prof/t6_8k_q${q}_${m}.json" --verbosity 2 *>> results/prof/t6_8k.log
    "done q$q $m" | Out-File -Append results/prof/t6_8k.log
  }
}
"ALL DONE" | Out-File -Append results/prof/t6_8k.log
