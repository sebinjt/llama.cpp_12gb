# Item 2: 64K / 128K prefill at ub 2048 with the output layer on the CPU (-ot output=CPU), to test whether
# moving the 2048 x 248320 x 4 B logits reserve off the device removes the long-context OOM.
# ub 512 with the same flags is the reference; output on CUDA0 is the old behaviour.
Set-Location C:\Users\sebinj\qwen_inference

$flags = @("--staged", "0", "--no-restore", "--n", "1", "--repeat", "1", "--verbosity", "2",
           "--env", "LLAMA_FA_VEC_MAX_NQ=128")

function Run($ctx, $ub, $outLayer, $prompt, $tag) {
    python _cfg.py @flags --ctx $ctx --b $ub --ub $ub --output $outLayer --prompt-file $prompt `
        --tag $tag --result "results\dq4\$tag.json"
}

# 64K prompt is 67,966 tokens; 128K prompt is 130,383 tokens
Run 69632  2048 "CPU"   "prompt_64k.txt"  "P64k-ub2048-outCPU"
Run 131072 2048 "CPU"   "prompt_128k.txt" "P128k-ub2048-outCPU"
Run 69632  512  "CPU"   "prompt_64k.txt"  "P64k-ub512-outCPU"
Run 131072 512  "CPU"   "prompt_128k.txt" "P128k-ub512-outCPU"
Run 69632  2048 "CUDA0" "prompt_64k.txt"  "P64k-ub2048-outCUDA"
Run 69632  512  "CUDA0" "prompt_64k.txt"  "P64k-ub512-outCUDA"
Run 131072 512  "CUDA0" "prompt_128k.txt" "P128k-ub512-outCUDA"
