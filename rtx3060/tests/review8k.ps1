Set-Location C:\Users\sebinj\qwen_inference
$common = @("--staged", "0", "--ctx", "16384",
    "--prompt-file", "results\dq4\code_short.prompt.txt", "--no-restore",
    "--n", "512", "--vram", "--verbosity", "3", "--draft-ubatch", "16",
    "--env", "LLAMA_FA_VEC_MAX_NQ=128", "--env", "LLAMA_FA_DECODE_Q4=1",
    "--env", "LLAMA_MTP_WINDOW=8192", "--env", "LLAMA_MMVQ_BATCH_INVARIANT=1",
    "--env", "LLAMA_MMVQ_UNPACK_ONCE=0")
$cases = @(
    @{tag="mtp3-base"; spec="draft-mtp"; d=3; replay=1; extra=@()},
    @{tag="mtp3-min"; spec="draft-mtp"; d=3; replay=1; extra=@("--samplers","temperature")},
    @{tag="mtp3-gpu"; spec="draft-mtp"; d=3; replay=1; extra=@("--samplers","temperature","--backend-sampling")},
    @{tag="mtp3-native"; spec="draft-mtp"; d=3; replay=0; extra=@("--samplers","temperature","--backend-sampling")},
    @{tag="df4-base"; spec="draft-dflash"; d=4; replay=1; extra=@()},
    @{tag="df4-gpu"; spec="draft-dflash"; d=4; replay=1; extra=@("--samplers","temperature","--backend-sampling")},
    @{tag="df4-native"; spec="draft-dflash"; d=4; replay=0; extra=@("--samplers","temperature","--backend-sampling")}
)
foreach ($case in $cases) {
    $tag = "R8k-$($case.tag)"
    python _cfg8k_review.py @common --spec $case.spec --draft-max $case.d --env "LLAMA_RS_REPLAY=$($case.replay)" @($case.extra) --tag $tag --result "results\dq4\$tag.json"
    if ($LASTEXITCODE -ne 0) { throw "Run $tag failed ($LASTEXITCODE)" }
    python -c "import json; p='results/dq4/$tag.json'; r=json.load(open(p))['response']; b=json.load(open('results/dq4/D8k-nomtp.json'))['response']; t=r['tokens']; u=b['tokens']; print('TOKEN_CHECK', len(t), t==u, 'first_difference',next((i for i,(a,b) in enumerate(zip(t,u)) if a!=b),None))"
}
