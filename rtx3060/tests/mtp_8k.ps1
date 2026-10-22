# EXP 1 end to end: MTP d3/d4/d5 at 8K, old (unpack per column) vs new (unpack once) MMVQ kernel.
# Same 8,828-token code prefix that produced the dq4-short-q4-mtp snapshot, ctx 16384, all layers resident,
# 512 greedy tokens, full exact-output flag set. Every run is compared against no-speculation with the same flags.
Set-Location C:\Users\sebinj\qwen_inference

$base = @("--staged", "0", "--ctx", "16384",
          "--prompt-file", "results\dq4\code_short.prompt.txt",
          "--snapshot", "dq4-short-q4-mtp.bin", "--n", "512", "--reps", "1",
          "--draft-ubatch", "16",
          "--env", "LLAMA_FA_VEC_MAX_NQ=128",
          "--env", "LLAMA_FA_DECODE_Q4=1",
          "--env", "LLAMA_RS_REPLAY=1",
          "--env", "LLAMA_MTP_WINDOW=8192",
          "--env", "LLAMA_MMVQ_BATCH_INVARIANT=1")

foreach ($un in @(0, 1)) {
    python tests\dq4_e2e.py @base --env "LLAMA_MMVQ_UNPACK_ONCE=$un" `
        --tag "M8k-nomtp-u$un" --out "results\dq4\M8k-nomtp-u$un.json"
    foreach ($d in @(3, 4, 5)) {
        python tests\dq4_e2e.py @base --env "LLAMA_MMVQ_UNPACK_ONCE=$un" `
            --spec draft-mtp --draft-max $d `
            --tag "M8k-d$d-u$un" --out "results\dq4\M8k-d$d-u$un.json"
    }
}
