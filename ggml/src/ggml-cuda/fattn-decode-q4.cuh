#pragma once

#include "common.cuh"

// Decode attention for q4_0 K / q4_0 V, head size 256, GQA 6, 1-8 query tokens, Ampere mma.sync (opt-in:
// LLAMA_FA_DECODE_Q4=1). CUDA port of the paged-decode design in amd_ref (fattn-paged-decode-kernel.cuh), contiguous
// K/V only.

// true when the env flag is set and dst has a shape the kernel takes
bool ggml_cuda_fattn_decode_q4_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_decode_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
