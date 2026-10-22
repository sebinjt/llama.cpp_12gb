# Does the DFlash drafter's memory grow with context? Same 8,828-token prompt, ctx 16384 vs 32768.
# The drafter KV is declared sliding-window 2048, so the 32K run shows whether that window is honoured
# in the allocation. Also: --draft-max 12, to record the block-size clamp (>8 verify columns is unreachable).
Set-Location C:\Users\sebinj\qwen_inference

$flags = @("--staged", "0", "--prompt-file", "results\dq4\code_short.prompt.txt",
           "--snapshot", "dq4-short-q4-mtp.bin", "--no-restore",
           "--n", "512", "--repeat", "1", "--vram", "--verbosity", "3", "--draft-ubatch", "16",
           "--env", "LLAMA_FA_VEC_MAX_NQ=128",
           "--env", "LLAMA_FA_DECODE_Q4=1",
           "--env", "LLAMA_RS_REPLAY=1",
           "--env", "LLAMA_MTP_WINDOW=8192",
           "--env", "LLAMA_MMVQ_BATCH_INVARIANT=1",
           "--env", "LLAMA_MMVQ_UNPACK_ONCE=1")

python _cfg.py @flags --ctx 32768 --tag "D32k-nomtp"   --result "results\dq4\D32k-nomtp.json"
python _cfg.py @flags --ctx 32768 --spec draft-dflash --draft-max 4 --tag "D32k-dflash4" --result "results\dq4\D32k-dflash4.json"
python _cfg.py @flags --ctx 32768 --spec draft-mtp    --draft-max 4 --tag "D32k-mtp4"    --result "results\dq4\D32k-mtp4.json"
python _cfg.py @flags --ctx 16384 --spec draft-dflash --draft-max 12 --tag "D8k-dflash12" --result "results\dq4\D8k-dflash12.json"
