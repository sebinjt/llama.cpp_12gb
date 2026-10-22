// rs_replay_check: replay rollback (LLAMA_RS_REPLAY=1) vs snapshot-plane rollback, bit for bit.
// For each mode, after a prompt: decode a verify-shaped batch of n_max+1 tokens, roll back to `keep` tokens
// (accept-all, partial, reject-all), then compare (1) the full sequence state blob, (2) the logits of the next
// decoded token, and (3) save -> clear -> restore -> decode, which must give the same logits.
// Usage: rs_replay_check <model> <prompt file> [n_prompt=2048] [n_max=4]
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static uint64_t fnv(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static bool decode(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, bool logits_last) {
    llama_batch b = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = pos0 + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = logits_last && i + 1 == toks.size();
    }
    b.n_tokens = (int) toks.size();
    const int r = llama_decode(ctx, b);
    llama_batch_free(b);
    return r == 0;
}

struct result { uint64_t state = 0, logits = 0, logits_restored = 0; size_t state_size = 0; bool ok = false; };

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model prompt [n_prompt] [n_max]\n", argv[0]); return 1; }
    const int n_prompt = argc > 3 ? atoi(argv[3]) : 2048;
    const int n_max    = argc > 4 ? atoi(argv[4]) : 4;
    _putenv_s("LLAMA_KV_STAGED_LAYERS", "0");

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream pf(argv[2], std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(pf)), {});
    std::vector<llama_token> toks(text.size() + 16);
    const int nt = llama_tokenize(vocab, text.data(), (int) text.size(), toks.data(), (int) toks.size(), true, true);
    if (nt < n_prompt + n_max + 3) { fprintf(stderr, "prompt too short\n"); return 1; }
    toks.resize(nt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const std::vector<llama_token> prompt(toks.begin(), toks.begin() + n_prompt);
    const std::vector<llama_token> verify(toks.begin() + n_prompt, toks.begin() + n_prompt + n_max + 1);
    const int keeps[] = { n_max + 1, 3, 1 }; // accept all, partial, reject all (only the first token kept)

    result res[3][3];
    for (int mode = 0; mode < 3; ++mode) {
        // mode 2: reference, decode only the kept tokens (no rollback), snapshot-plane memory
        _putenv_s("LLAMA_RS_REPLAY", mode == 1 ? "1" : "0");
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 128; cp.n_seq_max = 1; cp.n_rs_seq = n_max;
        cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }
        for (int s = 0; s < 3; ++s) {
            result & r = res[mode][s];
            const int keep = keeps[s];
            llama_memory_clear(llama_get_memory(ctx), true);
            bool ok = true;
            for (int p = 0; p < n_prompt && ok; p += 512) {
                const int n = std::min(512, n_prompt - p);
                ok = decode(ctx, std::vector<llama_token>(prompt.begin() + p, prompt.begin() + p + n), p, false);
            }
            if (mode == 2) {
                ok = ok && decode(ctx, std::vector<llama_token>(verify.begin(), verify.begin() + keep), n_prompt, true);
            } else {
                ok = ok && decode(ctx, verify, n_prompt, true);
            }
            // a speculative decoder reads the verify logits before it rolls back; do the same (this synchronizes)
            if (!getenv("RS_CHECK_NOSYNC")) {
                llama_synchronize(ctx);
            }
            if (ok && mode != 2 && keep < n_max + 1) {
                ok = llama_memory_seq_rm(llama_get_memory(ctx), 0, n_prompt + keep, -1);
                if (!ok) fprintf(stderr, "mode %d keep %d: seq_rm refused\n", mode, keep);
            }
            // state blob
            std::vector<uint8_t> st;
            if (ok) {
                st.resize(llama_state_seq_get_size(ctx, 0));
                ok = llama_state_seq_get_data(ctx, st.data(), st.size(), 0) == st.size();
                r.state = fnv(st.data(), st.size());
                r.state_size = st.size();
            }
            // next token: the token after the kept ones (teacher-forced)
            const llama_token next = toks[n_prompt + keep];
            if (ok) {
                ok = decode(ctx, { next }, n_prompt + keep, true);
                r.logits = ok ? fnv(llama_get_logits_ith(ctx, -1), sizeof(float) * n_vocab) : 0;
            }
            // save -> clear -> restore -> same next token
            if (ok) {
                llama_memory_clear(llama_get_memory(ctx), true);
                ok = llama_state_seq_set_data(ctx, st.data(), st.size(), 0) == st.size();
                ok = ok && decode(ctx, { next }, n_prompt + keep, true);
                r.logits_restored = ok ? fnv(llama_get_logits_ith(ctx, -1), sizeof(float) * n_vocab) : 0;
            }
            r.ok = ok;
            printf("mode=%s keep=%d/%d ok=%d state=%016llx (%zu B) next_logits=%016llx restored_logits=%016llx\n",
                   mode == 1 ? "replay   " : mode == 2 ? "reference" : "snapshot ", keep, n_max + 1, ok, (unsigned long long) r.state, r.state_size,
                   (unsigned long long) r.logits, (unsigned long long) r.logits_restored);
            fflush(stdout);
        }
        llama_free(ctx);
    }
    int bad = 0;
    for (int s = 0; s < 3; ++s) {
        const bool eq = res[0][s].ok && res[1][s].ok && res[0][s].state == res[1][s].state &&
                        res[0][s].logits == res[1][s].logits && res[1][s].logits == res[1][s].logits_restored &&
                        res[0][s].logits == res[0][s].logits_restored;
        printf("keep=%d: replay vs snapshot %s; replay vs reference: state %s logits %s; snapshot vs reference: state %s logits %s\n",
               keeps[s], eq ? "IDENTICAL (state, next logits, restored logits)" : "DIFFERENT",
               res[1][s].state == res[2][s].state ? "==" : "!=", res[1][s].logits == res[2][s].logits ? "==" : "!=",
               res[0][s].state == res[2][s].state ? "==" : "!=", res[0][s].logits == res[2][s].logits ? "==" : "!=");
        bad += !eq;
    }
    llama_model_free(model);
    return bad ? 2 : 0;
}
