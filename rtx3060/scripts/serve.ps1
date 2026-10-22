# Serve Qwen3.8-27B on a 12 GiB RTX 3060 with host-staged KV + MTP speculative decoding.
#
# Default = the fastest *stable* configuration measured on this card:
#   96K context, q4_0 KV, all 16 KV layers staged to pinned host RAM, MTP n_max=2
#   -> 10.55 t/s decode at 96K.
#
#   .\serve.ps1                                    # 127.0.0.1:8080
#   .\serve.ps1 -Port 9931 -HostAddress 0.0.0.0
#   .\serve.ps1 -Watchdog                          # auto-restart if the server exits
#   .\serve.ps1 -Ctx 131072 -StagedLayers 8 -Mtp 0 # 128K window, no spec decoding
#
# Staging depth is limited by the MTP draft context, which is created AFTER the host
# KV is reserved and costs ~495 MiB on the device: 108 MiB draft KV + 387 MiB compute
# buffer. Staging 10 or fewer layers leaves under 161 MiB free and the server dies at
# load with "failed to create MTP context". Staging 14 is the floor (593 MiB free).
# -DraftUBatch 8 trims that buffer from 428 to 387 MiB, which is what makes 14 viable;
# with the default 128 it needs 15-16 layers. A request that replaces the cached prompt
# also allocates a ~150 MiB context checkpoint, so do not run this at the edge.
# Measured at 96K after the flash-attention scratch fix: 12 staged layers give 11.28 t/s
# (88.7 ms/token). 10 staged gives 11.52 t/s but leaves only 35 MiB free during a 96K
# prefill, so 12 is the default. More staged layers is slower: 14 gives 10.79 t/s.
#
# Draft KV stays q4_0. flash attention used to reserve a context-sized f16 copy of K and V
# in the compute buffer of every attention layer, 387 MiB at 96K. Those copies now come
# from the CUDA pool for the duration of the op, so quantised KV no longer costs VRAM and
# the target compute buffer fell from 433 to 61 MiB.
param(
    [int]    $Port          = 8080,
    [string] $HostAddress   = '127.0.0.1',
    [int]    $Ctx           = 98304,
    [int]    $StagedLayers  = 12,
    [int]    $SlotDepth     = 2,
    # one draft token: n_max=1 measured faster than 2 or 3 at both 32K (24.44 vs 23.83 t/s)
    # and 200K (5.74 vs 4.80), because each extra draft pass costs a full attention sweep
    [int]    $Mtp           = 1,
    [string] $DraftTypeK    = 'q4_0',
    [string] $DraftTypeV    = 'q4_0',
    [int]    $DraftUBatch    = 8,
    [string] $Model         = 'Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf',
    [string] $TypeK         = 'q4_0',
    [string] $TypeV         = 'q4_0',
    [int]    $Batch         = 512,
    [int]    $UBatch        = 128,
    [int]    $Parallel      = 1,
    [string] $Reasoning     = 'off',
    [switch] $NoSlotCache,
    [switch] $Watchdog,
    [switch] $NoGraphs,
    [switch] $VecQuantKv,
    # e.g. -OverrideTensor 'output=CPU' keeps the 644 MiB output layer in host RAM for
    # ~0.9 ms of decode; nothing else frees that much VRAM (report 14.3).
    [string[]] $OverrideTensor = @()
)

$ErrorActionPreference = 'Stop'
# Tensor overrides use regex_search. Keep the output-head shorthand from matching attn_output.
$OverrideTensor = @(foreach ($ot in $OverrideTensor) {
    if ($ot.StartsWith('output=')) { '^output\.weight$=' + $ot.Substring(7) }
    else { $ot }
})
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exe  = Join-Path $root 'build-cuda\bin\llama-server.exe'
$gguf = Join-Path $root $Model
if (-not (Test-Path $exe))  { throw "llama-server not built at $exe (run scripts\build-cuda.bat)" }
if (-not (Test-Path $gguf)) { throw "model not found: $gguf" }

# Pre-flight VRAM estimate: fixed footprint + device-resident KV + staging slots
# (+ one extra layer of draft KV when MTP is on).
$kvPerLayer = 144.0 * $Ctx / 131072.0
# 9950 MiB is the full non-MTP footprint (9,948.60). An offloaded tensor moves out of it:
# output.weight is 644.1 MiB, token_embd.weight is 388.4 MiB and already host-mapped.
$offMib = 0
foreach ($ot in $OverrideTensor) {
    if (($ot -split '=', 2)[0] -eq '^output\.weight$') { $offMib += 644.1 }
    if ($ot -match '^\s*token_embd') { $offMib += 0 }   # already in CPU_Mapped
}
$needMib = [int](9950 - $offMib + (16 - $StagedLayers) * $kvPerLayer + $SlotDepth * $kvPerLayer + $(if ($Mtp -gt 0) { $kvPerLayer + 60 } else { 0 }))
try {
    $freeMib = [int]((& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null) | Select-Object -First 1)
} catch { $freeMib = 0 }
if ($freeMib -gt 0 -and $freeMib -lt $needMib) {
    Write-Warning "Only $freeMib MiB VRAM free; this config needs about $needMib MiB. Close GPU apps or lower -Ctx."
}

if ($StagedLayers -gt 0) { $env:LLAMA_KV_STAGED_LAYERS = "$StagedLayers" }
else                     { Remove-Item Env:\LLAMA_KV_STAGED_LAYERS -ErrorAction SilentlyContinue }
$env:LLAMA_KV_STAGE_SLOTS = "$SlotDepth"
# quantized KV attention without the full-context f16 scratch, so long context fits at deeper staging
if ($VecQuantKv) { $env:LLAMA_FA_VEC_QUANT_KV = '1' }
else { Remove-Item Env:\LLAMA_FA_VEC_QUANT_KV -ErrorAction SilentlyContinue }
foreach ($v in 'LLAMA_KV_STAGE_NODMA', 'LLAMA_KV_STAGE_TRACE', 'LLAMA_KV_STAGE_JOIN', 'GGML_CUDA_DISABLE_GRAPHS') {
    Remove-Item "Env:\$v" -ErrorAction SilentlyContinue
}
if ($NoGraphs) { $env:GGML_CUDA_DISABLE_GRAPHS = '1' }

$cliArgs = @(
    '-m', $gguf
    '-c', "$Ctx"
    '-ngl', '99'
    '-fa', 'on'
    '-ctk', $TypeK
    '-ctv', $TypeV
    '-b', "$Batch"
    '-ub', "$UBatch"
    '-np', "$Parallel"
    '--fit', 'off'              # never silently trade VRAM for speed
    '--no-context-shift'        # KV row-shift is untested with staged slots
    '--cache-ram', '4096'       # needed for multi-turn prefix reuse
    '--jinja'
    '--reasoning', $Reasoning
    '--host', $HostAddress
    '--port', "$Port"
    '--metrics'
    '--log-verbosity', '4'      # startup banners: staged layer count, KV size, MTP stats
)
if ($Mtp -gt 0) {
    $cliArgs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$Mtp",
                  '--spec-draft-type-k', $DraftTypeK, '--spec-draft-type-v', $DraftTypeV,
                  '--spec-draft-n-ubatch', "$DraftUBatch")
}
foreach ($ot in $OverrideTensor) { $cliArgs += @('-ot', $ot) }
if ($Parallel -ne 1) {
    Write-Warning "staged KV supports one sequence; -Parallel $Parallel will serialize requests."
}
# KV snapshot cache on disk: restart-resume and fast repeat prompts.
if (-not $NoSlotCache) {
    $slotDir = (Join-Path $PSScriptRoot '..\results\.kv_cache') + [IO.Path]::DirectorySeparatorChar
    New-Item -ItemType Directory -Force -Path $slotDir | Out-Null
    $cliArgs += @('--slot-save-path', $slotDir)
}

Write-Host "llama-server: ctx=$Ctx  K/V=$TypeK/$TypeV  staged=${StagedLayers}/16  slots=$SlotDepth  mtp=$Mtp  est. ${needMib} MiB" -ForegroundColor Cyan
Write-Host "  $exe $($cliArgs -join ' ')" -ForegroundColor DarkGray

if ($Watchdog) {
    while ($true) {
        & $exe @cliArgs
        $code = $LASTEXITCODE
        Write-Warning "llama-server exited with code $code; restarting in 3s (Ctrl+C to stop)"
        Start-Sleep -Seconds 3
    }
}
& $exe @cliArgs
