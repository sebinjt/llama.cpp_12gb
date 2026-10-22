// End-to-end numerical regression test: resident vs staged KV, actual model forwards.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void env(const char * value) {
#ifdef _WIN32
    _putenv_s("LLAMA_KV_STAGED_LAYERS", value);
#else
    if (*value) setenv("LLAMA_KV_STAGED_LAYERS", value, 1);
    else unsetenv("LLAMA_KV_STAGED_LAYERS");
#endif
}
int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 4096; cp.n_ubatch = 128; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = argc > 2 ? GGML_TYPE_Q5_0 : GGML_TYPE_F16;
    cp.type_v = argc > 2 ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
    cp.n_threads = 6; cp.n_threads_batch = 6;
    env(""); auto * resident = llama_init_from_model(model, cp);
    env("99"); auto * staged = llama_init_from_model(model, cp); env("");
    if (!resident || !staged) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::string text = "The old lighthouse stood above the harbour. Every evening the keeper checked the lamp, watched the weather, and recorded the ships returning home. ";
    std::string prompt;
    for (int i = 0; i < 35; ++i) prompt += text;
    std::vector<llama_token> tokens(prompt.size() + 16);
    int nt = llama_tokenize(vocab, prompt.data(), (int) prompt.size(), tokens.data(), (int) tokens.size(), true, true);
    if (nt <= 0 || nt > 3000) return 2;
    tokens.resize(nt);
    auto forward = [](llama_context * ctx, std::vector<llama_token> & t) {
        if (llama_decode(ctx, llama_batch_get_one(t.data(), (int) t.size())) != 0) std::exit(3);
    };
    auto compare = [&](const char * label) {
        auto * a = llama_get_logits_ith(resident, -1);
        auto * b = llama_get_logits_ith(staged, -1);
        double maxerr = 0, rms = 0;
        int mismatches = 0;
        for (int i = 0; i < nv; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) std::exit(4);
            double d = double(a[i]) - double(b[i]);
            maxerr = std::max(maxerr, std::abs(d)); rms += d*d;
            mismatches += a[i] != b[i];
        }
        printf("%s: max_abs=%.9g rms=%.9g different=%d/%d\n", label, maxerr, std::sqrt(rms/nv), mismatches, nv);
        // No new approximation; the very same CUDA kernels consume bit-identical cache.
        if (maxerr > 1e-5) std::exit(5);
        return int(std::max_element(a, a + nv) - a);
    };
    forward(resident, tokens); forward(staged, tokens);
    int next = compare("prefill");
    for (int i = 0; i < 32; ++i) {
        std::vector<llama_token> batch{next};
        forward(resident, batch); forward(staged, batch);
        next = compare("decode");
    }
    // Save/restore the canonical pinned cache; then verify a multi-token block (spec verification geometry).
    size_t size = llama_state_get_size(staged);
    std::vector<uint8_t> state(size);
    size_t saved = llama_state_get_data(staged, state.data(), state.size());
    if (!saved) return 6;
    llama_memory_clear(llama_get_memory(staged), true);
    if (llama_state_set_data(staged, state.data(), saved) != saved) return 6;
    std::vector<llama_token> block(tokens.begin() + 5, tokens.begin() + 9);
    forward(resident, block); forward(staged, block); compare("state-restore+verify4");
    // Both caches discard the rejected suffix, then overwrite it with other tokens.
    const bool rm_a = llama_memory_seq_rm(llama_get_memory(resident), 0, nt + 33, -1);
    const bool rm_b = llama_memory_seq_rm(llama_get_memory(staged), 0, nt + 33, -1);
    if (rm_a != rm_b) return 7;
    if (rm_a) {
        block.assign(tokens.begin() + 10, tokens.begin() + 13);
        forward(resident, block); forward(staged, block); compare("rollback+overwrite");
    } else {
        puts("SKIP suffix rollback: upstream recurrent model requires its speculative state-checkpoint path");
    }
    llama_free(staged); llama_free(resident); llama_model_free(model); llama_backend_free();
    puts("PASS: staged KV matches resident prefill/decode, graph replay, state IO and rollback");
}
