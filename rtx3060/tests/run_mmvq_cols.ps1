# Microbench: MMVQ per-tensor time vs column count, old vs new (unpack once) kernel.
# Both runs use the same binary; only LLAMA_MMVQ_UNPACK_ONCE differs.
Set-Location C:\Users\sebinj\qwen_inference
$env:PATH = "C:\Users\sebinj\qwen_inference\..\build-cuda\bin;" + $env:PATH
$out = "results\mmvq_cols"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:LLAMA_MMVQ_UNPACK_ONCE = "0"
& tests\bin\mmvq_cols_bench.exe 50 "$out\old2" | Tee-Object -FilePath "$out\old2.txt"

$env:LLAMA_MMVQ_UNPACK_ONCE = "1"
& tests\bin\mmvq_cols_bench.exe 50 "$out\new2" | Tee-Object -FilePath "$out\new2.txt"
