// Build the MTP cache from saved target hidden rows; emit the server draft appendix.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) try {
    std::string model, prompt, hidden, output;
    int ctx = 204800, ub = 1024;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("missing option value");
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--model") model = v;
        else if (k == "--prompt") prompt = v;
        else if (k == "--hidden") hidden = v;
        else if (k == "--out") output = v;
        else if (k == "--ctx") ctx = std::stoi(v);
        else if (k == "--ubatch") ub = std::stoi(v);
        else throw std::runtime_error("unknown option " + k);
    }
    if (model.empty() || prompt.empty() || hidden.empty() || output.empty() || ctx < 2 || ub < 16 || ub > 2048 ||
            std::filesystem::exists(output) || std::filesystem::exists(output + ".partial")) {
        throw std::runtime_error("require --model --prompt --hidden and new --out; valid ctx/ubatch");
    }
    std::ifstream hf(hidden, std::ios::binary);
    char magic[8]; uint64_t rows = 0; uint32_t width = 0, dtype = 0;
    hf.read(magic, 8); hf.read((char *) &rows, 8); hf.read((char *) &width, 4); hf.read((char *) &dtype, 4);
    if (!hf || std::memcmp(magic, "QNEXTH01", 8) || dtype || !rows || rows >= (uint64_t) ctx || width != 5120 ||
            std::filesystem::file_size(hidden) != 24 + rows * width * sizeof(float)) {
        throw std::runtime_error("invalid hidden history");
    }
    _putenv_s("LLAMA_PREFILL_LAYER_RANGE", "");
    _putenv_s("LLAMA_KV_STAGED_LAYERS", "0");
    _putenv_s("LLAMA_FA_VEC_QUANT_KV", "");
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
    ggml_backend_load_all(); llama_backend_init();
    llama_model_tensor_buft_override overrides[] = {
        { "^blk\\.([0-9]|[1-5][0-9]|6[0-3])\\.", ggml_backend_cpu_buffer_type() }, { nullptr, nullptr }
    };
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99; mp.load_mtp = true; mp.tensor_buft_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> m(llama_model_load_from_file(model.c_str(), mp), llama_model_free);
    if (!m || llama_model_n_layer_nextn(m.get()) != 1) throw std::runtime_error("need a single-head MTP model");
    std::ifstream pf(prompt, std::ios::binary);
    if (!pf) throw std::runtime_error("cannot read prompt");
    std::string text((std::istreambuf_iterator<char>(pf)), {});
    auto * vocab = llama_model_get_vocab(m.get());
    int nt = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(std::abs(nt));
    nt = llama_tokenize(vocab, text.data(), (int) text.size(), tokens.data(), (int) tokens.size(), true, true);
    if (nt < 0 || (uint64_t) nt < rows) throw std::runtime_error("prompt too short for hidden history");
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = ctx; cp.n_batch = cp.n_ubatch = ub; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.n_outputs_max = 1; cp.n_threads = cp.n_threads_batch = 6;
    cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> c(llama_init_from_model(m.get(), cp), llama_free);
    if (!c) throw std::runtime_error("MTP context failed");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> b(llama_batch_ext_init(c.get()), llama_batch_ext_free);
    std::vector<float> carry(width, 0), h((size_t) ub * width);
    auto start = std::chrono::steady_clock::now();
    for (uint64_t p = 0; p < rows; p += ub) {
        const int n = (int) std::min<uint64_t>(ub, rows - p);
        hf.read((char *) h.data(), (size_t) n * width * sizeof(float));
        if (!hf) throw std::runtime_error("truncated hidden history");
        llama_batch_ext_clear(b.get());
        for (int j = 0; j < n; ++j) {
            const int idx = llama_batch_ext_add_token(b.get(), 0, tokens[p + j]);
            const llama_pos pos = (llama_pos) (p + j);
            const float * prev = j ? h.data() + (size_t) (j - 1) * width : carry.data();
            if (idx < 0 || !llama_batch_ext_set_pos(b.get(), idx, &pos) ||
                    !llama_batch_ext_set_output_logits(b.get(), idx, false) ||
                    !llama_batch_ext_set_embd_state(b.get(), idx, {prev, 1, width})) {
                throw std::runtime_error("invalid MTP batch");
            }
        }
        if (llama_process(c.get(), LLAMA_PROCESS_TYPE_DECODE, b.get())) throw std::runtime_error("MTP decode failed");
        std::copy_n(h.data() + (size_t) (n - 1) * width, width, carry.data());
        if (p % (ub * 16) == 0 || p + n == rows) {
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("MTP_PROGRESS tokens=%llu/%llu seconds=%.3f tps=%.1f\n", (unsigned long long) (p+n), (unsigned long long) rows, sec, (p+n)/sec);
            fflush(stdout);
        }
    }
    std::vector<uint8_t> kv(llama_state_seq_get_size_ext(c.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    if (llama_state_seq_get_data_ext(c.get(), kv.data(), kv.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != kv.size()) {
        throw std::runtime_error("draft state save failed");
    }
    std::ofstream out(output + ".partial", std::ios::binary);
    const uint32_t hdr[] = {0x54464453, 1};
    const uint32_t shdr[] = {0x3150544d, 1, width, 0};
    const uint64_t nk = kv.size(), ns = sizeof(shdr) + width * sizeof(float);
    out.write((const char *) hdr, sizeof(hdr));
    out.write((const char *) &nk, 8); out.write((const char *) kv.data(), kv.size());
    out.write((const char *) &ns, 8); out.write((const char *) shdr, sizeof(shdr));
    out.write((const char *) carry.data(), width * sizeof(float)); out.flush();
    if (!out) throw std::runtime_error("appendix write failed");
    out.close(); std::filesystem::rename(output + ".partial", output);
    printf("MTP_SAVED rows=%llu bytes=%llu\n", (unsigned long long) rows, (unsigned long long) std::filesystem::file_size(output));
    return 0;
} catch (const std::exception & e) {
    fprintf(stderr, "ERROR %s\n", e.what()); return 1;
}
