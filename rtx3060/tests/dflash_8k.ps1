# EXP 2: DFlash vs MTP at 8K. Same 8,828-token code prefix, ctx 16384, all layers resident, 512 greedy tokens,
# full exact-output flag set. DFlash needs target features, so every run here prefills fresh (no snapshot restore)
# and the baseline is a fresh no-speculation decode with the same flags.
Set-Location C:\Users\sebinj\qwen_inference

$common = @("--staged", "0", "--ctx", "16384",
            "--prompt-file", "results\dq4\code_short.prompt.txt",
            "--snapshot", "dq4-short-q4-mtp.bin", "--no-restore",
            "--n", "512", "--repeat", "1", "--vram", "--verbosity", "3",
            "--draft-ubatch", "16",
            "--env", "LLAMA_FA_VEC_MAX_NQ=128",
            "--env", "LLAMA_FA_DECODE_Q4=1",
            "--env", "LLAMA_RS_REPLAY=1",
            "--env", "LLAMA_MTP_WINDOW=8192",
            "--env", "LLAMA_MMVQ_BATCH_INVARIANT=1",
            "--env", "LLAMA_MMVQ_UNPACK_ONCE=1")

python _cfg.py @common --tag "D8k-nomtp" --result "results\dq4\D8k-nomtp.json"
python _cfg.py @common --spec draft-mtp --draft-max 4 --tag "D8k-mtp4" --result "results\dq4\D8k-mtp4.json"
foreach ($d in @(3, 4, 5, 6, 7)) {
    python _cfg.py @common --spec draft-dflash --draft-max $d `
        --tag "D8k-dflash$d" --result "results\dq4\D8k-dflash$d.json"
}
