# Where does MTP exactness break at 8K? Same setup, full exact flags minus LLAMA_MMVQ_BATCH_INVARIANT.
# d3 verifies 4 columns (1-4 warp path, unchanged); d4 verifies 5 columns (2-warp path).
Set-Location C:\Users\sebinj\qwen_inference

$base = @("--staged", "0", "--ctx", "16384",
          "--prompt-file", "results\dq4\code_short.prompt.txt",
          "--snapshot", "dq4-short-q4-mtp.bin", "--n", "512", "--reps", "1",
          "--draft-ubatch", "16",
          "--env", "LLAMA_FA_VEC_MAX_NQ=128",
          "--env", "LLAMA_FA_DECODE_Q4=1",
          "--env", "LLAMA_RS_REPLAY=1",
          "--env", "LLAMA_MTP_WINDOW=8192",
          "--env", "LLAMA_MMVQ_UNPACK_ONCE=0")

foreach ($d in @(3, 4)) {
    python tests\dq4_e2e.py @base --spec draft-mtp --draft-max $d `
        --tag "M8k-d$d-nobinv" --out "results\dq4\M8k-d$d-nobinv.json"
}
