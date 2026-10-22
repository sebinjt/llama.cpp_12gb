<#
  tests\serve_prod.ps1 - production launcher for the configs that were actually MEASURED in report.md.
  Nothing here is guessed: every flag/env below is cited to the section that measured it.

  Profiles
  --------
  p8k    Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp (IQ3_XXS) + DFlash drafter Q2_K_S, ctx 16384, KV resident,
         GGML_CUDA_PREFER_MMQ=1, --spec-draft-n-max 7                       report 33 "Recommended exact
         configs" 8K row: 54.19 t/s all-MMQ + DFlash d7, exact. MMQ at 8K is +53% (report 41.1) and
         PREFER_MMQ=1 alone is exact after the sec-33 fix (no DISABLE_FUSION needed).

  plong  same model/drafter, ctx 204800, LLAMA_KV_GPU_BUDGET_MIB=1000, PREFER_MMQ=1, -n-max 7.
         report 33 200K row (21.06 t/s, KV staged) + report 42 (MMQ is the right mode everywhere:
         +4.7% mean over 5 task prompts at 200K, equal acceptance; exact). -Ctx overrides -c without
         touching anything else, so the "same speed at 4K" check runs both contexts on one set of flags
         (report 44/46 measured the -c dependence this checks for).

  s64k   Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp (IQ3_S) on D: + DFlash drafter, ctx 65792, KV budget 300 MiB
         (14/16 attention layers staged - the budget is a CEILING, so 300 stages the max; report 55.7 a),
         slots 48 MiB, prefetch on. Decode row: report 55.7 c.
         STREAMED SET CHANGED to S=797 MiB: 55.7's S=612 set (16 attention ffn_down) fits only a snapshot
         RESTORE - a real request cannot allocate its prefill compute buffer (measured: first request dies
         with "CUDA error: out of memory" at once, harness and launcher alike, because every 55.7 row was
         restore-based). Measured floor for a real 12,265-token prefill: S=736 fails, S=797 works with 629 MiB
         free at peak; S=797 then decodes 23.65 t/s at 64K context (restore, text 4f3bce79f1efe976) and
         27.4 t/s at 12K context, against 25.35 t/s for the restore-only S=612 set. The set is the 16
         attention ffn_down plus the GDN ffn_down of 0,4,8,12,16,20.
         -ub 32 is required, not cosmetic: with -ub 128 the same S=797 set dies in the first prefill
         ("CUDA error: the resource allocation failed", peak 11711 MiB), which is why the profile carries
         its own -b/-ub instead of report 33's 512/128.
         GGML_CUDA_DISABLE_GRAPHS=1 matches the measured rows; graphs on measured 25.18 (55.7 c).

  Common to every profile (report 33 "Common environment"): LLAMA_FA_DECODE_Q4=1, LLAMA_FA_VEC_MAX_NQ=128,
  LLAMA_RS_REPLAY=1, LLAMA_MTP_WINDOW=8192, LLAMA_MMVQ_BATCH_INVARIANT=1, LLAMA_MMVQ_UNPACK_ONCE=0, and the
  server flags -fa on -ctk q4_0 -ctv q4_0 -b 512 -ub 128 -np 1 --fit off --no-context-shift --jinja
  -ot "^output\.weight$=CUDA0". LLAMA_STEP_TIMING=1 is diagnostics only (report 27/55 STEPTIME lines, no
  effect on tokens or speed).

  Usage
  -----
    powershell -ExecutionPolicy Bypass -File tests\serve_prod.ps1 -Profile p8k -Port 8091
    powershell -ExecutionPolicy Bypass -File tests\serve_prod.ps1 -Profile plong -Ctx 32768 -Port 8092
    powershell -ExecutionPolicy Bypass -File tests\serve_prod.ps1 -Profile s64k -DryRun   # print, run nothing

  Parameters
  ----------
    -Profile p8k|plong|s64k   the measured config above (required)
    -Port N                   listening port (default 8080)
    -Ctx N                    overrides -c only, everything else stays (scenario 1: 32768 vs 204800)
    -Ctk / -Ctv TYPE          KV cache types, default q4_0/q4_0 = the production path, command line unchanged.
                              A q8* type on either side automatically removes LLAMA_FA_VEC_QUANT_KV (that route
                              only knows q4_0) and the launcher prints which route it chose (report 60.1/60.6 C.4).
    -NoSpec                   drop the drafter (spec vs no-spec exactness A/B)
    -Blocks "L1,L2,..."       s64k only: overrides the streamed ffn_down layer set (and its -ot pattern)
    -Batch N / -Ub N          override -b / -ub (s64k carries its own 512/32; 128 OOMs its prefill, report 58.1)
    -Env "NAME=VALUE"         extra environment variable, merged last (e.g. LLAMA_ARG_THINK_BUDGET=0)
    -ExtraArgs '--flag' 'v'   extra raw server arguments, passed as separate tokens
    -DryRun                   print profile, env, route and the exact command line, launch nothing

  The GPU lock (comm\GPU_LOCK_RULES.md) is the caller's job: acquire before launching, hold until this
  process has exited, release only your own.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('p8k', 'plong', 's64k')]
    [string]$Profile,

    [int]$Port = 8080,

    # 0 = use the profile's own context; otherwise overrides -c only (scenario 1: 32768 vs 204800)
    [int]$Ctx = 0,

    # disable speculative decoding (spec vs no-spec exactness A/B; scenario 3)
    [switch]$NoSpec,

    # print the exact command line and environment, launch nothing
    [switch]$DryRun,

    # override the streamed block list (s64k only): the measured S=612 leaves too little VRAM for a real
    # prefill (58.1), so the profile runs S=797; this switch rewrites both the env and the -ot override.
    [string]$Blocks = '',

    # KV cache types. Default q4_0/q4_0 = the measured production path, unchanged.
    [string]$Ctk = 'q4_0',
    [string]$Ctv = 'q4_0',

    # batch / ubatch. 512/128 is report 33's measured server config; s64k's 25.35 t/s row was measured at
    # -ub 32 (report 55.7), and the ubatch also sets the prefill compute buffer that decides whether a real
    # request fits next to the drafter on a 12 GiB card.
    [int]$Batch = 0,
    [int]$Ub = 0,

    # extra environment variables as NAME=VALUE (e.g. LLAMA_ARG_CTX_CHECKPOINTS=0), merged last
    [string[]]$Env = @(),

    # extra raw arguments appended last (pass them as separate tokens: -ExtraArgs '--flag' 'value')
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'

# repository root is two levels above this script (rtx3060/tests -> repo root)
$Root     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Exe      = Join-Path $Root 'build-cuda\bin\llama-server.exe'
$Drafter  = Join-Path $Root 'Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf'
$Iq3xxs   = Join-Path $Root 'Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf'
$Iq3s     = Join-Path $Root 'Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'
$Slots    = Join-Path $PSScriptRoot '..\results\.kv_cache'

# S=797 MiB: the 16 attention layers of IQ3_S (every 4th, report 55.7 c/d) plus the GDN ffn_down of
# 0,4,8,12,16,20, because the measured S=612 set only survives a snapshot restore (report 58.1)
$S64kBlocks = '2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,0,4,8,12,16,20'

$profiles = @{
    p8k = @{
        model    = $Iq3xxs
        ctx      = 16384
        draft_n  = 7
        budget   = $null
        blocks   = $null
        graphs   = $true
        source   = 'report 33 (recommended exact configs, 8K row 54.19 t/s; exactness sec 33/41.1)'
    }
    plong = @{
        model    = $Iq3xxs
        ctx      = 204800
        draft_n  = 7
        budget   = 1000
        blocks   = $null
        graphs   = $true
        source   = 'report 33 (recommended 200K row, budget 1000) + report 42 (MMQ +4.7%, exact)'
    }
    s64k = @{
        model    = $Iq3s
        ctx      = 65792
        draft_n  = 7
        budget   = 300
        blocks   = $S64kBlocks
        graphs   = $false
        batch    = 512
        ub       = 32
        source   = 'report 55.7 c decode row + 58.1 re-size (budget 300 = 14/16 staged, S=797 not 612, -ub 32; 23.65 t/s at 64K context, sha 4f3bce79f1efe976)'
    }
}

$p = $profiles[$Profile]
if (-not $p.ContainsKey('batch')) { $p.batch = 512; $p.ub = 128 }   # report 33 server config
if ($Batch -gt 0) { $p = $p.Clone(); $p.batch = $Batch }
if ($Ub -gt 0)    { $p = $p.Clone(); $p.ub = $Ub }
if ($Blocks) {
    if ($null -eq $p.blocks) { throw "-Blocks only applies to a profile with weight streaming (s64k)" }
    $p = $p.Clone()
    $p.blocks = $Blocks
}
$useCtx = if ($Ctx -gt 0) { $Ctx } else { $p.ctx }

# ---- environment (set in this process so the child inherits it exactly) -----------------------------
$envVars = [ordered]@{
    LLAMA_FA_DECODE_Q4        = '1'                  # report 33 common env
    LLAMA_FA_VEC_MAX_NQ       = '128'                # report 33 common env
    LLAMA_RS_REPLAY           = '1'                  # report 33 common env
    LLAMA_MTP_WINDOW          = '8192'               # report 33 common env
    LLAMA_MMVQ_BATCH_INVARIANT = '1'                 # report 33 common env
    LLAMA_MMVQ_UNPACK_ONCE    = '0'                  # report 33 common env
    LLAMA_STEP_TIMING         = '1'                  # diagnostics only (report 27/55 STEPTIME)
    GGML_CUDA_PREFER_MMQ      = '1'                  # report 42: right mode at every context, exact
}
if ($null -ne $p.budget) {
    $envVars['LLAMA_KV_GPU_BUDGET_MIB'] = "$($p.budget)"   # 55.7 a: ceiling, small = staged
}
if ($null -ne $p.blocks) {
    $envVars['LLAMA_W_STAGE_ENABLE']    = '1'        # report 55.2 M2 weight streaming, opt-in
    $envVars['LLAMA_W_STAGE_BLOCKS']    = $p.blocks  # spread set, S=797 MiB (58.1)
    $envVars['LLAMA_W_STAGE_SLOT_MIB']  = '48'       # report 55.2 (2 x 48 MiB slots)
    $envVars['LLAMA_W_STAGE_PREFETCH']  = '1'        # report 55.2: +25 of the 51 ms DMA hidden
}
foreach ($e in $Env) {
    if ($e -notmatch '^([^=]+)=(.+)$') { throw "bad -Env '$e' (want NAME=VALUE)" }
    $envVars[$Matches[1]] = $Matches[2]
}
# KV-quant route (report 60.1/60.6): LLAMA_FA_VEC_QUANT_KV only knows q4_0, so any q8_x KV is dragged through the
# generic vector kernel while it is set. Drop it for q8 (the MMA f16 route, measured +18% decode / +42% PP at 8K);
# leave it untouched for q4_0/q4_0, which is the measured production path.
$kvRoute = 'vec-q4 (LLAMA_FA_VEC_QUANT_KV set)'
if (($Ctk -like 'q8*') -or ($Ctv -like 'q8*')) {
    $envVars.Remove('LLAMA_FA_VEC_QUANT_KV') | Out-Null
    $kvRoute = 'mma-f16 (LLAMA_FA_VEC_QUANT_KV not set)'
}
if (-not $p.graphs) {
    $envVars['GGML_CUDA_DISABLE_GRAPHS'] = '1'       # 55.7 c: the quoted 25.35 row was graphs off
}
if ($NoSpec) {
    $envVars.Remove('LLAMA_MTP_WINDOW') | Out-Null   # drafting off: not read, kept out of the picture
}

# ---- command line ----------------------------------------------------------------------------------
$ot = '^output\.weight$=CUDA0'                       # report 33 common command
if ($null -ne $p.blocks) {
    $pat = ($p.blocks -split ',') -join '|'
    $ot += ",^blk\.($pat)\.ffn_down\.weight=CPU"     # 55.7: the streamed tensors are sourced from CPU
}

$argv = @(
    '-m', $p.model
    '-c', "$useCtx"
    '-ngl', '99'
    '-fa', 'on'
    '-ctk', $Ctk
    '-ctv', $Ctv
    '-b', "$($p.batch)"
    '-ub', "$($p.ub)"
    '-np', '1'
    '--fit', 'off'
    '--no-context-shift'
    '--jinja'
    '--port', "$Port"
    '--slot-save-path', $Slots
    '-ot', $ot
)
if (-not $NoSpec) {
    $argv += @(
        '--spec-type', 'draft-dflash'
        '--spec-draft-model', $Drafter
        '--spec-draft-ngl', '99'
        '--spec-draft-type-k', 'q4_0'
        '--spec-draft-type-v', 'q4_0'
        '--spec-draft-n-max', "$($p.draft_n)"
        '--spec-draft-n-ubatch', '16'
    )
}
if ($ExtraArgs.Count -gt 0) { $argv += $ExtraArgs }

# ---- report -------------------------------------------------------------------------------------------
Write-Host "profile : $Profile   ($($p.source))"
Write-Host "kv      : -ctk $Ctk -ctv $Ctv  route: $kvRoute"
Write-Host "batch   : -b $($p.batch) -ub $($p.ub)$(if ($Batch -or $Ub) { '  (overridden)' })"
Write-Host "ctx     : $useCtx$(if ($Ctx -gt 0) { "  (overridden from $($p.ctx))" })"
Write-Host "spec    : $(if ($NoSpec) { 'OFF (no-spec A/B)' } else { "DFlash d7 (n-max $($p.draft_n))" })"
Write-Host "env     :"
foreach ($k in $envVars.Keys) { Write-Host ("  {0}={1}" -f $k, $envVars[$k]) }
Write-Host "command :"
Write-Host ("  `"$Exe`" " + ($argv -join ' '))

if ($DryRun) { return }

foreach ($k in $envVars.Keys) { Set-Item -Path ("Env:" + $k) -Value $envVars[$k] }

Write-Host "starting server on port $Port ..."
& $Exe @argv
