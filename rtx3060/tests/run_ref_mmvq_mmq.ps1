# Reference: per-tensor MMVQ and MMQ times for the 4 weight types used by the small-batch study.
Set-Location C:\Users\sebinj\qwen_inference
$env:PATH = "C:\Users\sebinj\qwen_inference\..\build-cuda\bin;" + $env:PATH
$out = "results\ref"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# MMVQ (default routing)
Remove-Item Env:GGML_CUDA_PREFER_MMQ -ErrorAction SilentlyContinue
& tests\bin\mmvq_cols_bench4.exe 20 | Tee-Object -FilePath "$out\mmvq.txt"
# MMQ (forced)
$env:GGML_CUDA_PREFER_MMQ = "1"
& tests\bin\mmvq_cols_bench4.exe 20 | Tee-Object -FilePath "$out\mmq.txt"
