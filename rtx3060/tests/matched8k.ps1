# seq 43 item 1: matched 8K baseline. One harness (_cfg.py, no n_probs request), fresh prefill every run,
# 8,828-token code prefix, ctx 16384, all layers resident, 512 greedy tokens, exact-output env, unpack-once off.
# Four configurations x three rounds with the order alternating (forward, reverse, forward).
Set-Location C:\Users\sebinj\qwen_inference

$common = @("--staged", "0", "--ctx", "16384",
            "--prompt-file", "results\dq4\code_short.prompt.txt",
            "--no-restore", "--n", "512", "--repeat", "1", "--vram", "--verbosity", "2",
            "--draft-ubatch", "16",
            "--env", "LLAMA_FA_VEC_MAX_NQ=128",
            "--env", "LLAMA_FA_DECODE_Q4=1",
            "--env", "LLAMA_RS_REPLAY=1",
            "--env", "LLAMA_MTP_WINDOW=8192",
            "--env", "LLAMA_MMVQ_BATCH_INVARIANT=1",
            "--env", "LLAMA_MMVQ_UNPACK_ONCE=0")

function Cfgs($round) {
    $list = @(
        @{ name = "nospec";  extra = @() },
        @{ name = "mtp3";    extra = @("--spec", "draft-mtp", "--draft-max", "3") },
        @{ name = "dflash3"; extra = @("--spec", "draft-dflash", "--draft-max", "3") },
        @{ name = "dflash4"; extra = @("--spec", "draft-dflash", "--draft-max", "4") }
    )
    if ($round -eq 2) { [array]::Reverse($list) }
    return $list
}

foreach ($round in @(1, 2, 3)) {
    foreach ($cfg in Cfgs $round) {
        $tag = "M8k2-R$round-$($cfg.name)"
        python _cfg.py @common @($cfg.extra) --tag $tag --result "results\dq4\$tag.json"
    }
}
