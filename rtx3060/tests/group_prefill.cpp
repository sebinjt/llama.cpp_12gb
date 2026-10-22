// Offline layer-group prefill with ordinary llama.cpp sequence-state output.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

static void env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) unsetenv(name); else setenv(name, value.c_str(), 1);
#endif
}

struct options {
    std::string model, prompt, out, state, logits, nextn, trace, dflash, mode = "split";
    int ctx = 204800, ubatch = 512, split = 32, limit = 0, staged = 0, generate = 16;
};

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) throw std::runtime_error("each option requires a value");
        std::string k = argv[i], v = argv[i + 1];
        if      (k == "--model") o.model = v;
        else if (k == "--prompt") o.prompt = v;
        else if (k == "--out") o.out = v;
        else if (k == "--state") o.state = v;
        else if (k == "--logits") o.logits = v;
        else if (k == "--nextn-out") o.nextn = v;
        else if (k == "--trace-dir") o.trace = v;
        else if (k == "--dflash-features") o.dflash = v;
        else if (k == "--mode") o.mode = v;
        else if (k == "--ctx") o.ctx = std::stoi(v);
        else if (k == "--ubatch") o.ubatch = std::stoi(v);
        else if (k == "--split") o.split = std::stoi(v);
        else if (k == "--tokens") o.limit = std::stoi(v);
        else if (k == "--staged") o.staged = std::stoi(v);
        else if (k == "--generate") o.generate = std::stoi(v);
        else throw std::runtime_error("unknown option: " + k);
    }
    if (o.model.empty() || o.prompt.empty() || o.ctx < 2 || o.ctx > 262144 ||
        o.ubatch < 16 || o.ubatch > 2048 || o.limit < 0 || o.staged < 0 || o.staged > 16 ||
        o.generate < 1 || o.generate > 1024 || (o.mode != "split" && o.mode != "reference" && o.mode != "restore")) {
        throw std::runtime_error("invalid options; require --model GGUF --prompt TXT [--out SNAPSHOT]");
    }
    if (o.mode != "restore" && (o.out.empty() || std::filesystem::exists(o.out) || std::filesystem::exists(o.out + ".partial"))) {
        throw std::runtime_error("choose a new --out snapshot path");
    }
    if (o.mode == "restore" && o.state.empty()) throw std::runtime_error("restore requires --state");
    if (!o.nextn.empty() && (o.mode == "restore" || std::filesystem::exists(o.nextn) || std::filesystem::exists(o.nextn + ".partial"))) {
        throw std::runtime_error("choose a new --nextn-out path in a prefill mode");
    }
    return o;
}

static model_ptr load_model(const options & o, int begin, int end) {
    std::string pattern;
    for (int il = 0; il < 64; ++il) {
        if (il >= begin && il < end) continue;
        if (!pattern.empty()) pattern += "|";
        pattern += std::to_string(il);
    }
    pattern = "blk\\.(" + pattern + ")\\..*";
    std::vector<llama_model_tensor_buft_override> overrides;
    if (begin != 0 || end != 64) {
        overrides.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        if (end != 64) overrides.push_back({"^output\\.weight$", ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});
    auto p = llama_model_default_params();
    p.n_gpu_layers = 99;
    p.load_mtp = false;
    p.tensor_buft_overrides = overrides.data();
    model_ptr m(llama_model_load_from_file(o.model.c_str(), p), llama_model_free);
    if (!m) throw std::runtime_error("model load failed");
    char arch[64] = {};
    llama_model_meta_val_str(m.get(), "general.architecture", arch, sizeof(arch));
    if (std::string(arch) != "qwen35" || llama_model_n_layer(m.get()) != 64) {
        throw std::runtime_error("this tool requires the 64-layer dense Qwen35 model");
    }
    return m;
}

static bool trace_tensor(ggml_tensor * t, bool ask, void * opaque) {
    const std::string name(t->name);
    const bool selected = (name.size() >= 2 && name.compare(name.size() - 2, 2, "-3") == 0) || name == "attn_inp_kq_mask";
    if (!selected || ggml_nelements(t) >= 1000000) return false;
    if (ask) return true;
    const auto & dir = *static_cast<const std::string *>(opaque);
    std::filesystem::create_directories(dir);
    std::ofstream f(std::filesystem::path(dir) / (name + ".bin"), std::ios::binary);
    const int32_t type = t->type;
    f.write((const char *) &type, sizeof(type));
    f.write((const char *) t->ne, sizeof(t->ne));
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    f.write((const char *) bytes.data(), bytes.size());
    if (name == "attn_output-3") {
        std::ofstream meta(std::filesystem::path(dir) / "projection.txt");
        meta << "op " << ggml_op_name(t->op) << "\n";
        for (int i = 0; i < 2; ++i) {
            auto * src = t->src[i];
            meta << i << " " << src->name << " " << ggml_type_name(src->type) << " " << ggml_backend_buffer_name(src->buffer);
            for (int k = 0; k < 4; ++k) meta << " " << src->ne[k] << ":" << src->nb[k];
            meta << "\n";
            std::vector<uint8_t> raw(ggml_nbytes(src));
            ggml_backend_tensor_get(src, raw.data(), 0, raw.size());
            std::ofstream data(std::filesystem::path(dir) / ("projection-src" + std::to_string(i) + ".bin"), std::ios::binary);
            data.write((const char *) raw.data(), raw.size());
        }
        for (int i = 0; i < GGML_MAX_OP_PARAMS; ++i) meta << (int) ((const uint8_t *) t->op_params)[i] << " ";
    }
    return true;
}

static context_ptr make_context(const options & o, llama_model * m, bool boundary) {
    auto p = llama_context_default_params();
    p.n_ctx = o.ctx;
    p.n_batch = o.ubatch;
    p.n_ubatch = o.ubatch;
    p.n_seq_max = 1;
    p.n_rs_seq = 0;
    p.n_outputs_max = boundary ? o.ubatch : 1;
    p.n_threads = 6;
    p.n_threads_batch = 6;
    p.type_k = GGML_TYPE_Q4_0;
    p.type_v = GGML_TYPE_Q4_0;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    p.pooling_type = LLAMA_POOLING_TYPE_NONE;
    p.embeddings = boundary;
    if (!o.trace.empty()) {
        p.cb_eval = trace_tensor;
        p.cb_eval_user_data = (void *) &o.trace;
    }
    context_ptr c(llama_init_from_model(m, p), llama_free);
    if (!c) throw std::runtime_error("context allocation failed");
    if (!boundary && !o.nextn.empty()) llama_set_embeddings_nextn(c.get(), true, false);
    return c;
}

static std::vector<llama_token> tokenize(llama_model * m, const options & o) {
    std::ifstream f(o.prompt, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open prompt");
    std::string text((std::istreambuf_iterator<char>(f)), {});
    const auto * vocab = llama_model_get_vocab(m);
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens((size_t) std::abs(n));
    n = llama_tokenize(vocab, text.data(), (int) text.size(), tokens.data(), (int) tokens.size(), true, true);
    if (n < 2) throw std::runtime_error("prompt must contain at least two tokens");
    tokens.resize(n);
    return tokens;
}

static void prefill(llama_context * c, const options & o, const std::vector<llama_token> & tokens,
                    int count, int d, std::vector<float> * hidden, bool input_hidden, const char * label) {
    auto start = clock_type::now();
    double last_report = -60;
    std::ofstream nextn;
    if (!o.nextn.empty() && (input_hidden || !hidden)) {
        nextn.open(o.nextn + ".partial", std::ios::binary);
        const uint64_t rows = count;
        const uint32_t width = d, type = 0;
        nextn.write("QNEXTH01", 8);
        nextn.write((const char *) &rows, sizeof(rows));
        nextn.write((const char *) &width, sizeof(width));
        nextn.write((const char *) &type, sizeof(type));
        if (!nextn) throw std::runtime_error("cannot create target hidden-state file");
    }
    llama_batch b = llama_batch_init(o.ubatch, input_hidden ? d : 0, 1);
    if (input_hidden) b.token = (llama_token *) malloc((size_t) o.ubatch*sizeof(llama_token));
    for (int pos = 0; pos < count; pos += o.ubatch) {
        b.n_tokens = std::min(o.ubatch, count - pos);
        const int feature_begin = std::max(0, count - 2048);
        std::vector<int> feature_layers;
        if (!o.dflash.empty() && pos + b.n_tokens > feature_begin) {
            for (int il : {6, 20, 34, 48, 62}) {
                if (o.mode == "split" && ((il < o.split) == input_hidden)) continue;
                feature_layers.push_back(il);
                llama_set_embeddings_layer_inp(c, il, true);
            }
        }
        if (input_hidden) {
            std::copy_n(hidden->data() + (size_t) pos*d, (size_t) b.n_tokens*d, b.embd);
        }
        for (int i = 0; i < b.n_tokens; ++i) {
            b.token[i] = tokens[pos + i];
            b.pos[i] = pos + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (hidden && !input_hidden) || pos + i == count - 1;
        }
        if (llama_decode(c, b) != 0) {
            llama_batch_free(b);
            throw std::runtime_error(std::string(label) + " decode failed at " + std::to_string(pos));
        }
        llama_synchronize(c);
        for (int il : feature_layers) {
            const std::string path = o.dflash + ".layer" + std::to_string(il) + ".f32";
            const bool first = pos <= feature_begin;
            std::ofstream f(path, std::ios::binary | (first ? std::ios::trunc : std::ios::app));
            if (first) {
                const uint64_t n = count - feature_begin, begin = feature_begin;
                const uint32_t width = d, layer = il;
                f.write("QDFEAT01", 8); f.write((const char *) &n, 8); f.write((const char *) &begin, 8);
                f.write((const char *) &width, 4); f.write((const char *) &layer, 4);
            }
            const float * values = llama_get_embeddings_layer_inp(c, il);
            if (!values) throw std::runtime_error("missing DFlash target features");
            const int skip = std::max(0, feature_begin - pos);
            f.write((const char *) (values + (size_t) skip*d), (size_t) (b.n_tokens - skip)*d*sizeof(float));
            if (!f) throw std::runtime_error("DFlash target feature write failed");
        }
        if (nextn.is_open()) {
            const float * values = llama_get_embeddings_nextn(c);
            if (!values) throw std::runtime_error("missing target nextn embeddings");
            nextn.write((const char *) values, (size_t) b.n_tokens*d*sizeof(float));
            if (!nextn) throw std::runtime_error("target hidden-state write failed");
        }
        if (hidden && !input_hidden) {
            const float * out = llama_get_embeddings(c);
            if (!out) throw std::runtime_error("missing layer boundary embeddings");
            std::copy_n(out, (size_t) b.n_tokens*d, hidden->data() + (size_t) pos*d);
        }
        const double elapsed = seconds(start);
        if (elapsed - last_report >= 15 || pos + b.n_tokens == count) {
            printf("PROGRESS %s %d/%d %.1f s %.1f tok/s\n", label, pos + b.n_tokens, count,
                   elapsed, (pos + b.n_tokens)/std::max(0.001, elapsed));
            last_report = elapsed;
        }
    }
    llama_batch_free(b);
    if (nextn.is_open()) {
        nextn.close();
        if (!nextn) throw std::runtime_error("target hidden-state close failed");
        std::filesystem::rename(o.nextn + ".partial", o.nextn);
        printf("HIDDEN_SAVED %s rows=%d width=%d bytes=%zu\n", o.nextn.c_str(), count, d, (size_t) count*d*sizeof(float) + 24);
    }
}

static void save(llama_context * c, const options & o, const std::vector<llama_token> & tokens, int count) {
    auto start = clock_type::now();
    const std::string temp = o.out + ".partial";
    const size_t n = llama_state_seq_save_file(c, temp.c_str(), 0, tokens.data(), count);
    if (!n) throw std::runtime_error("snapshot save failed");
    std::filesystem::rename(temp, o.out);
    printf("SAVED %s tokens=%d bytes=%zu seconds=%.3f\n", o.out.c_str(), count, n, seconds(start));
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const auto o = parse(argc, argv);
        env("LLAMA_KV_STAGED_LAYERS", "");
        env("LLAMA_KV_GPU_BUDGET_MIB", "");
        env("LLAMA_PREFILL_LAYER_RANGE", "");
        env("LLAMA_FA_VEC_QUANT_KV", "");
        env("GGML_CUDA_DISABLE_GRAPHS", "1");
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
        }, nullptr);
        printf("BACKENDS loading\n");
        ggml_backend_load_all();
        printf("BACKENDS devices=%zu\n", ggml_backend_dev_count());
        llama_backend_init();
        auto start = clock_type::now();
        if (o.mode == "restore") {
            env("LLAMA_KV_STAGED_LAYERS", std::to_string(o.staged));
            env("LLAMA_FA_VEC_QUANT_KV", "1");
            auto m = load_model(o, 0, 64);
            auto c = make_context(o, m.get(), false);
            auto tokens = tokenize(m.get(), o);
            if (o.limit) {
                if (o.limit >= (int) tokens.size()) throw std::runtime_error("--tokens exceeds prompt length");
                tokens.resize(o.limit + 1);
            }
            std::vector<llama_token> saved(o.ctx);
            size_t count = 0;
            if (!llama_state_seq_load_file(c.get(), o.state.c_str(), 0, saved.data(), saved.size(), &count)) {
                throw std::runtime_error("restore failed");
            }
            if (count >= tokens.size() || !std::equal(saved.begin(), saved.begin() + count, tokens.begin())) {
                throw std::runtime_error("snapshot must be a strict token prefix of the prompt");
            }
            printf("RESTORED tokens=%zu seconds=%.3f\n", count, seconds(start));
            std::vector<llama_token> tail(tokens.begin() + count, tokens.end());
            if (llama_decode(c.get(), llama_batch_get_one(tail.data(), (int) tail.size())) != 0) {
                throw std::runtime_error("suffix decode failed");
            }
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m.get()));
            for (int i = 0; i < o.generate; ++i) {
                const float * logits = llama_get_logits_ith(c.get(), -1);
                if (i == 0 && !o.logits.empty()) {
                    std::ofstream f(o.logits, std::ios::binary);
                    f.write((const char *) logits, (size_t) nv*sizeof(float));
                    if (!f) throw std::runtime_error("logit dump failed");
                }
                llama_token next = (llama_token) (std::max_element(logits, logits + nv) - logits);
                printf("TOKEN %d %d\n", i, next);
                if (i + 1 < o.generate && llama_decode(c.get(), llama_batch_get_one(&next, 1)) != 0) {
                    throw std::runtime_error("generation failed");
                }
            }
        } else {
            if (o.mode == "split" && (o.split < 4 || o.split > 60 || o.split % 4)) {
                throw std::runtime_error("--split must be a multiple of four between 4 and 60");
            }
            if (o.mode == "split") env("LLAMA_PREFILL_LAYER_RANGE", "0:" + std::to_string(o.split));
            auto m = load_model(o, 0, o.mode == "split" ? o.split : 64);
            const auto tokens = tokenize(m.get(), o);
            const int count = o.limit ? o.limit : (int) tokens.size() - 1;
            if (count >= o.ctx || count >= (int) tokens.size()) {
                throw std::runtime_error("prefill must fit context and leave at least one prompt token for restore");
            }
            const int d = llama_model_n_embd(m.get());
            printf("BEGIN mode=%s ctx=%d prompt_tokens=%zu cached_tokens=%d ubatch=%d split=%d\n",
                   o.mode.c_str(), o.ctx, tokens.size(), count, o.ubatch, o.split);
            auto c = make_context(o, m.get(), o.mode == "split");
            if (o.mode == "split") {
                std::vector<float> hidden((size_t) count*d);
                prefill(c.get(), o, tokens, count, d, &hidden, false, "first");
                auto transfer_start = clock_type::now();
                std::vector<uint8_t> state(llama_state_seq_get_size(c.get(), 0));
                const size_t used = llama_state_seq_get_data(c.get(), state.data(), state.size(), 0);
                if (used != state.size()) throw std::runtime_error("partial state export failed");
                c.reset();
                m.reset();
                printf("SWAP exported_bytes=%zu seconds=%.3f boundary_bytes=%zu\n", used,
                       seconds(transfer_start), hidden.size()*sizeof(float));
                env("LLAMA_PREFILL_LAYER_RANGE", std::to_string(o.split) + ":64");
                m = load_model(o, o.split, 64);
                c = make_context(o, m.get(), false);
                if (llama_state_seq_set_data(c.get(), state.data(), state.size(), 0) != state.size()) {
                    throw std::runtime_error("partial state import failed");
                }
                std::vector<uint8_t>().swap(state);
                // Rebuild position metadata with the same sequence; completed layer tensors stay intact.
                llama_memory_clear(llama_get_memory(c.get()), false);
                printf("SWAP ready seconds=%.3f\n", seconds(transfer_start));
                prefill(c.get(), o, tokens, count, d, &hidden, true, "second");
            } else {
                prefill(c.get(), o, tokens, count, d, nullptr, false, "reference");
            }
            printf("PREFILL_DONE tokens=%d seconds=%.3f effective_tok_s=%.3f\n", count, seconds(start), count/seconds(start));
            save(c.get(), o, tokens, count);
            printf("COMPLETE seconds=%.3f\n", seconds(start));
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "ERROR %s\n", e.what());
        return 1;
    }
}
