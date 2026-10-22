// Initialize the windowed DFlash cache from saved target-layer input features.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <algorithm>
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
    std::string target, draft, features, output;
    int ctx = 204800;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("missing option value");
        std::string k = argv[i], v = argv[i+1];
        if (k == "--model") target = v;
        else if (k == "--draft") draft = v;
        else if (k == "--features") features = v;
        else if (k == "--out") output = v;
        else if (k == "--ctx") ctx = std::stoi(v);
        else throw std::runtime_error("unknown option " + k);
    }
    if (target.empty() || draft.empty() || features.empty() || output.empty() || ctx < 256 ||
            std::filesystem::exists(output) || std::filesystem::exists(output + ".partial")) throw std::runtime_error("invalid arguments");
    _putenv_s("LLAMA_KV_STAGED_LAYERS", ""); _putenv_s("LLAMA_FA_VEC_QUANT_KV", "");
    _putenv_s("LLAMA_PREFILL_LAYER_RANGE", ""); _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
    ggml_backend_load_all(); llama_backend_init();
    using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
    using ctx_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    model_ptr mt(llama_model_load_from_file(target.c_str(), mp), llama_model_free);
    if (!mt) throw std::runtime_error("target model failed");
    auto cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = cp.n_ubatch = 128;
    cp.n_seq_max = 1; cp.n_rs_seq = 0; cp.n_outputs_max = 1;
    cp.n_threads = cp.n_threads_batch = 6; cp.type_k = cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    ctx_ptr ct(llama_init_from_model(mt.get(), cp), llama_free);
    if (!ct) throw std::runtime_error("target placeholder context failed");
    mp.n_gpu_layers = 99;
    model_ptr md(llama_model_load_from_file(draft.c_str(), mp), llama_model_free);
    if (!md) throw std::runtime_error("draft model failed");
    cp.ctx_other = ct.get(); cp.n_ctx = ctx; cp.n_batch = cp.n_ubatch = 512; cp.n_outputs_max = 8;
    ctx_ptr cd(llama_init_from_model(md.get(), cp), llama_free);
    if (!cd) throw std::runtime_error("draft context failed");
    const auto * layers = llama_model_target_layer_ids(md.get());
    const uint32_t nl = llama_model_target_layer_ids_n(md.get()), width = llama_model_n_embd(mt.get());
    if (nl != 5 || width != 5120) throw std::runtime_error("unexpected DFlash feature layout");
    uint64_t rows = 0, begin = 0;
    std::vector<float> data;
    for (uint32_t k = 0; k < nl; ++k) {
        std::ifstream f(features + ".layer" + std::to_string(layers[k]) + ".f32", std::ios::binary);
        char magic[8]; uint64_t n = 0, pos = 0; uint32_t w = 0, il = 0;
        f.read(magic, 8); f.read((char *) &n, 8); f.read((char *) &pos, 8); f.read((char *) &w, 4); f.read((char *) &il, 4);
        if (!f || std::memcmp(magic, "QDFEAT01", 8) || w != width || il != (uint32_t) layers[k] || !n || n > 2048 || pos+n >= (uint64_t) ctx ||
                (k && (n != rows || pos != begin))) throw std::runtime_error("invalid feature header");
        if (!k) { rows = n; begin = pos; data.resize(rows * nl * width); }
        for (uint64_t i = 0; i < rows; ++i) f.read((char *) (data.data() + (i*nl+k)*width), width*sizeof(float));
        if (!f || f.peek() != std::char_traits<char>::eof()) throw std::runtime_error("invalid feature payload");
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> b(llama_batch_ext_init(cd.get()), llama_batch_ext_free);
    for (uint64_t p = 0; p < rows; p += 512) {
        const int n = (int) std::min<uint64_t>(512, rows-p);
        llama_batch_ext_clear(b.get());
        for (int i = 0; i < n; ++i) {
            const int idx = llama_batch_ext_add_embd(b.get(), 0, {data.data() + (p+i)*nl*width, 1, (size_t) nl*width});
            const llama_pos pos = (llama_pos) (begin+p+i), positions[] = {pos,pos,pos,0};
            if (idx < 0 || !llama_batch_ext_set_pos(b.get(), idx, positions) || !llama_batch_ext_set_output_logits(b.get(), idx, false)) throw std::runtime_error("invalid injection batch");
        }
        if (llama_process(cd.get(), LLAMA_PROCESS_TYPE_DECODE, b.get())) throw std::runtime_error("feature injection failed");
    }
    std::vector<uint8_t> kv(llama_state_seq_get_size_ext(cd.get(), 0, LLAMA_STATE_SEQ_FLAGS_NONE));
    if (llama_state_seq_get_data_ext(cd.get(), kv.data(), kv.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) != kv.size()) throw std::runtime_error("draft export failed");
    const uint32_t hdr[] = {0x54464453,1}; const uint64_t nk = kv.size(), ns = 0;
    std::ofstream out(output + ".partial", std::ios::binary);
    out.write((const char *) hdr, sizeof(hdr)); out.write((const char *) &nk, 8); out.write((const char *) kv.data(), kv.size()); out.write((const char *) &ns, 8);
    out.flush(); if (!out) throw std::runtime_error("appendix write failed");
    out.close(); std::filesystem::rename(output + ".partial", output);
    printf("DFLASH_SAVED rows=%llu begin=%llu bytes=%llu\n", (unsigned long long) rows, (unsigned long long) begin, (unsigned long long) std::filesystem::file_size(output));
    return 0;
} catch (const std::exception & e) { fprintf(stderr, "ERROR %s\n", e.what()); return 1; }
