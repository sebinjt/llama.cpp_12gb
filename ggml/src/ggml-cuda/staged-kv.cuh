// Opt-in, layer-granular host KV with depth-N device staging slots.
// Included by ggml-cuda.cu after the ordinary CUDA buffer-type implementation.
//
// Pipeline (stream mode; staged decode never reaches CUDA-graph replay because
// KV op shapes grow every token):
//   - canonical KV lives in pinned host RAM (cudaHostAlloc Mapped|Portable)
//   - N slots x per-layer capacity, two non-blocking copy streams, prefetching
//     is prefix-sized (only the used rows, not the allocated capacity)
//   - prefetch(i) into slot i%N waits ONLY on that slot's persistent "consumed"
//     event (recorded after the flash-attention that last read the slot), so the
//     copy engine runs continuously ACROSS token boundaries; there is no
//     per-token fork/join that would quiesce and restart the pipeline
//   - each layer's K and V are one contiguous copy each, on two copy streams
//   - flash attention / SET_ROWS wait on both copy-done events of the slot
//   - GGML_KV_STAGE_PROF=1 prints per-layer copy and attention device times

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

static const char * ggml_cuda_kv_host_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_cuda_buffer_type_context *>(buft->context)->name.c_str();
}
static ggml_backend_buffer_t ggml_cuda_kv_host_alloc(ggml_backend_buffer_type_t buft, size_t n);
static bool ggml_cuda_is_kv_host(ggml_backend_buffer_type_t buft) {
    // Do not use get_name identity: MSVC /OPT:ICF can fold its body with the
    // ordinary CUDA name getter, making every device buffer appear host-backed.
    return buft && buft->iface.alloc_buffer == ggml_cuda_kv_host_alloc;
}
struct ggml_cuda_kv_host_buffer {
    int device;
    void * host;
    void * base;
};
static void ggml_cuda_kv_host_free(ggml_backend_buffer_t b) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(p->host));
    delete p;
}
static void * ggml_cuda_kv_host_base(ggml_backend_buffer_t b) {
    return static_cast<ggml_cuda_kv_host_buffer *>(b->context)->base;
}
static char * ggml_cuda_kv_host_ptr(ggml_backend_buffer_t b, const void * addr) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    return (char *) p->host + ((const char *) addr - (const char *) p->base);
}
// Everything that changes the canonical pinned prefix (restore, clear, plain set)
// bumps this, invalidating any slot that was prefetched for the next token.
static uint64_t ggml_cuda_kv_host_generation = 1;

static void ggml_cuda_kv_host_set(ggml_backend_buffer_t b, ggml_tensor * t, const void * src, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(ggml_cuda_kv_host_ptr(b, t->data) + off, src, n);
    ggml_cuda_kv_host_generation++;
}
static void ggml_cuda_kv_host_get(ggml_backend_buffer_t b, const ggml_tensor * t, void * dst, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(dst, ggml_cuda_kv_host_ptr(b, t->data) + off, n);
}
static void ggml_cuda_kv_host_memset(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t value, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memset(ggml_cuda_kv_host_ptr(b, t->data) + off, value, n);
}
// Async backend IO is allowed to complete synchronously. State snapshots/restore
// must never issue a device-only memcpy against a mapped canonical host buffer.
static bool ggml_cuda_kv_host_io(ggml_backend_cuda_context & ctx, ggml_backend_buffer_t b,
                                const ggml_tensor * t, void * data, size_t offset,
                                size_t size, size_t count, size_t tensor_stride,
                                size_t data_stride, bool write) {
    if (!ggml_cuda_is_kv_host(b->buft)) return false;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    char * host = ggml_cuda_kv_host_ptr(b, t->data) + offset;
    for (size_t i = 0; i < count; ++i) {
        if (write) memcpy(host + i * tensor_stride, (const char *) data + i * data_stride, size);
        else       memcpy((char *) data + i * data_stride, host + i * tensor_stride, size);
    }
    if (write) ggml_cuda_kv_host_generation++;
    return true;
}
static void ggml_cuda_kv_host_clear(ggml_backend_buffer_t b, uint8_t value) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memset(p->host, value, b->size);
    ggml_cuda_kv_host_generation++;
}
static ggml_backend_buffer_t ggml_cuda_kv_host_alloc(ggml_backend_buffer_type_t buft, size_t n) {
    auto * type = static_cast<ggml_backend_cuda_buffer_type_context *>(buft->context);
    ggml_cuda_set_device(type->device);
    void * host = nullptr;
    cudaError_t err = cudaHostAlloc(&host, std::max<size_t>(n, 1), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("staged KV: failed to pin %.1f MiB: %s\n", n / 1048576.0, cudaGetErrorString(err));
        (void) cudaGetLastError();
        return nullptr;
    }
    void * base = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&base, host, 0));
    const ggml_backend_buffer_i iface = {
        ggml_cuda_kv_host_free, ggml_cuda_kv_host_base, nullptr,
        ggml_cuda_kv_host_memset, ggml_cuda_kv_host_set, ggml_cuda_kv_host_get,
        nullptr, nullptr, nullptr, ggml_cuda_kv_host_clear, nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, new ggml_cuda_kv_host_buffer{type->device, host, base}, n);
}
static size_t ggml_cuda_kv_host_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    return ggml_nbytes(t);
}
// Registry extension: same signature as ggml_backend_dev_buffer_type(). Not a generic
// host buffer: scheduler must keep SET_ROWS + FLASH_ATTN on the owning CUDA device.
static ggml_backend_buffer_type_t ggml_backend_cuda_kv_host_buffer_type(ggml_backend_dev_t dev) {
    static std::mutex mutex;
    static std::map<ggml_backend_dev_t, std::unique_ptr<ggml_backend_buffer_type>> types;
    std::lock_guard<std::mutex> lock(mutex);
    auto & type = types[dev];
    if (!type) {
        auto * original = static_cast<ggml_backend_cuda_buffer_type_context *>(ggml_backend_dev_buffer_type(dev)->context);
        type.reset(new ggml_backend_buffer_type{
            {ggml_cuda_kv_host_name, ggml_cuda_kv_host_alloc, nullptr, ggml_backend_cuda_buffer_type_get_alignment,
             nullptr, ggml_cuda_kv_host_alloc_size, nullptr, nullptr}, dev,
            new ggml_backend_cuda_buffer_type_context{original->device, original->name + "_KV_StagedHost"}});
    }
    return type.get();
}

struct ggml_cuda_kv_staging {
    int device;
    int n_slots = 0;
    cudaStream_t copy[2] = { nullptr, nullptr };
    cudaEvent_t join[2] = {};       // [stream]: copy-stream tail, joined at the end of an eval
    cudaEvent_t ready[2][8] = {};   // [stream][slot]: chunk copies done
    cudaEvent_t consumed[8] = {};   // [slot]: last flash attention read done (persistent across evals)
    void * k[8] = {};
    void * v[8] = {};
    size_t cap_k = 0, cap_v = 0;
    // What slot holds right now, and which canonical-host generation it came from.
    // Lets the next eval reuse a slot that was already staged at the end of the
    // previous eval instead of paying for the copy again.
    int    slot_layer[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
    uint64_t slot_gen[8] = {};

    // TEMP diagnostic: GGML_KV_STAGE_PROF=1 dumps per-layer copy/wait/attention times.
    bool prof = false;
    int  prof_evals = 0;
    std::vector<std::array<cudaEvent_t, 5>> pev;

    void ensure_prof(size_t n) {
        if (!prof) return;
        while (pev.size() < n) {
            std::array<cudaEvent_t, 5> e{};
            for (auto & ev : e) {
                CUDA_CHECK(cudaEventCreate(&ev));
            }
            pev.push_back(e);
        }
        if (!pref) {
            CUDA_CHECK(cudaEventCreate(&pref));
        }
    }
    cudaEvent_t pref = nullptr;
    void dump_prof(ggml_backend_cuda_context & ctx) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(copy[0]));
        CUDA_CHECK(cudaStreamSynchronize(copy[1]));
        double s_dma = 0, s_gap = 0, s_attn = 0;
        float span = 0;
        std::string rows;
        for (size_t i = 0; i < pev.size(); ++i) {
            float cs = 0, ce = 0, ve = 0, as = 0, ae = 0;
            CUDA_CHECK(cudaEventElapsedTime(&cs, pref, pev[i][0]));
            CUDA_CHECK(cudaEventElapsedTime(&ce, pref, pev[i][1]));
            CUDA_CHECK(cudaEventElapsedTime(&ve, pref, pev[i][2]));
            CUDA_CHECK(cudaEventElapsedTime(&as, pref, pev[i][3]));
            CUDA_CHECK(cudaEventElapsedTime(&ae, pref, pev[i][4]));
            const float copy_end = std::max(ce, ve);
            const float gap = as - copy_end;
            s_dma += copy_end - cs; s_gap += gap; s_attn += ae - as;
            span = std::max(span, ae);
            char buf[200];
            snprintf(buf, sizeof(buf), "  L%zu dma[%7.2f ->%7.2f] attn[%7.2f ->%7.2f] gap %+6.2f\n",
                     i, cs, copy_end, as, ae, gap);
            rows += buf;
        }
        GGML_LOG_INFO("staged KV profile eval %d: span %.2f ms, dma %.2f, idle-gap %.2f, attn %.2f\n",
                      prof_evals, span, s_dma, s_gap, s_attn);
        if (prof_evals % 32 == 0) {
            GGML_LOG_INFO("%s", rows.c_str());
        }
        prof_evals++;
    }

    explicit ggml_cuda_kv_staging(int dev) : device(dev) {
        prof = getenv("GGML_KV_STAGE_PROF") != nullptr;

        const char * env = getenv("LLAMA_KV_STAGE_SLOTS");
        // one slot is allowed: it lowers VRAM but serialises copy and attention
        n_slots = env && env[0] ? std::max(1, std::min(8, std::atoi(env))) : 2;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy[0], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy[1], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&join[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&join[1], cudaEventDisableTiming));
        for (int s = 0; s < 8; ++s) {
            CUDA_CHECK(cudaEventCreateWithFlags(&consumed[s], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(consumed[s], copy[0]));  // pre-signal: first evals may copy at once
        }
        for (int st = 0; st < 2; ++st) {
            for (int s = 0; s < 8; ++s) {
                CUDA_CHECK(cudaEventCreateWithFlags(&ready[st][s], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(ready[st][s], copy[st]));
            }
        }
    }
    ~ggml_cuda_kv_staging() {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int s = 0; s < 8; ++s) {
            CUDA_CHECK(cudaFree(k[s])); CUDA_CHECK(cudaFree(v[s]));
            CUDA_CHECK(cudaEventDestroy(consumed[s]));
        }
        for (int st = 0; st < 2; ++st) {
            for (int s = 0; s < 8; ++s) {
                CUDA_CHECK(cudaEventDestroy(ready[st][s]));
            }
            CUDA_CHECK(cudaEventDestroy(join[st]));
            CUDA_CHECK(cudaStreamDestroy(copy[st]));
        }
    }
    bool reserve(size_t nk, size_t nv) {
        if (nk <= cap_k && nv <= cap_v) return false;
        CUDA_CHECK(cudaDeviceSynchronize());
        cap_k = std::max(cap_k, nk); cap_v = std::max(cap_v, nv);
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        GGML_LOG_INFO("staged KV: %d slots x %.2f MiB = %.2f MiB; %.2f MiB dedicated VRAM free of %.2f MiB\n",
                      n_slots, (double) (cap_k + cap_v) / 1048576.0,
                      (double) n_slots * (cap_k + cap_v) / 1048576.0,
                      (double) free_b / 1048576.0, (double) total_b / 1048576.0);
        for (int s = 0; s < n_slots; ++s) {
            CUDA_CHECK(cudaFree(k[s])); CUDA_CHECK(cudaFree(v[s]));
            cudaError_t err = cudaMalloc(&k[s], cap_k);
            if (err == cudaSuccess) err = cudaMalloc(&v[s], cap_v);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("staged KV: staging slots do not fit in VRAM: need %.2f MiB, %.2f MiB free of %.2f MiB. "
                               "Lower LLAMA_KV_STAGE_SLOTS/context or stage more layers. "
                               "Failing instead of letting the driver spill to host memory.\n",
                               (double) n_slots * (cap_k + cap_v) / 1048576.0,
                               (double) free_b / 1048576.0, (double) total_b / 1048576.0);
                GGML_ABORT("staged KV staging slots do not fit in VRAM");
            }
            slot_layer[s] = -1;
            slot_gen[s] = 0;
        }
        GGML_LOG_INFO("staged KV: %d device layer slots = %.2f MiB, pinned host backing; prefix-only DMA\n",
                      n_slots, (double) n_slots * (cap_k + cap_v) / 1048576.0);
        return true;
    }
};
static const ggml_tensor * ggml_cuda_kv_root(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}
static bool ggml_cuda_kv_tensor(const ggml_tensor * t) {
    return t && t->buffer && ggml_cuda_is_kv_host(t->buffer->buft);
}
struct ggml_cuda_kv_plan {
    struct layer {
        ggml_tensor * attention;
        const ggml_tensor * k;
        const ggml_tensor * v;
        size_t nk, nv;
    };
    std::vector<layer> layers;
    ggml_cuda_kv_staging * staging = nullptr;
    explicit ggml_cuda_kv_plan(ggml_cgraph * graph) {
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (node->op != GGML_OP_FLASH_ATTN_EXT || !ggml_cuda_kv_tensor(node->src[1])) continue;
            GGML_ASSERT(ggml_cuda_kv_tensor(node->src[2]));
            const auto * k = ggml_cuda_kv_root(node->src[1]);
            const auto * v = ggml_cuda_kv_root(node->src[2]);
            // Flash input: [head_dim, cells, heads, sequences]. Only one sequence/stream.
            GGML_ASSERT(node->src[1]->ne[3] == 1 && k->ne[2] == 1 && v->ne[2] == 1);
            GGML_ASSERT(k->nb[1] == ggml_row_size(k->type, k->ne[0]));
            GGML_ASSERT(v->nb[1] == ggml_row_size(v->type, v->ne[0]));
            layers.push_back({node, k, v, k->nb[1] * size_t(node->src[1]->ne[1]),
                                            v->nb[1] * size_t(node->src[2]->ne[1])});
        }
    }
    // Copy layer i's used prefix into slot (i % n_slots): K on copy[0], V on
    // copy[1]. Waits ONLY on the slot's persistent consumed event, i.e. the last
    // flash attention that read this slot in ANY earlier eval, so the DMA starts
    // as soon as the previous token released the slot.
    void prefetch(size_t i) {
        const auto & l = layers[i]; const int s = i % staging->n_slots;
        const char * trace = getenv("LLAMA_KV_STAGE_TRACE");
        if (trace) { fprintf(stderr, "kv prefetch %zu %s K=%zu V=%zu\n", i, l.k->name, l.nk, l.nv); fflush(stderr); }
        const char * src_k = ggml_cuda_kv_host_ptr(l.k->buffer, l.k->data);
        const char * src_v = ggml_cuda_kv_host_ptr(l.v->buffer, l.v->data);
        for (int st = 0; st < 2; ++st) {
            CUDA_CHECK(cudaStreamWaitEvent(staging->copy[st], staging->consumed[s], 0));
        }
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pev[i][0], staging->copy[0]));
        }
        // One contiguous copy per stream: chunking was measured not to change
        // throughput (13.4 GB/s for 1 to 16 chunks), so keep the simple form.
        CUDA_CHECK(cudaMemcpyAsync(staging->k[s], src_k, l.nk, cudaMemcpyHostToDevice, staging->copy[0]));
        CUDA_CHECK(cudaMemcpyAsync(staging->v[s], src_v, l.nv, cudaMemcpyHostToDevice, staging->copy[1]));
        CUDA_CHECK(cudaEventRecord(staging->ready[0][s], staging->copy[0]));
        CUDA_CHECK(cudaEventRecord(staging->ready[1][s], staging->copy[1]));
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pev[i][1], staging->copy[0]));
            CUDA_CHECK(cudaEventRecord(staging->pev[i][2], staging->copy[1]));
        }
        staging->slot_layer[s] = (int) i;
        staging->slot_gen[s] = ggml_cuda_kv_host_generation;
    }
    void begin(ggml_backend_cuda_context & ctx) {
        if (layers.empty()) return;
        staging = ctx.kv_staging.get();
        // Prefetch the slots this eval starts with. A slot already staged for the
        // same layer at the end of the previous eval is reused: the rows written
        // since then are applied by SET_ROWS, so the copy is still valid.
        staging->ensure_prof(layers.size());
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pref, ctx.stream()));
        }
        const int n = std::min<size_t>(staging->n_slots, layers.size());
        for (size_t i = 0; i < (size_t) n; ++i) {
            const int s = (int) (i % staging->n_slots);
            if (staging->slot_layer[s] == (int) i && staging->slot_gen[s] == ggml_cuda_kv_host_generation) continue;
            prefetch(i);
        }
    }
    // Intercept writes: wait for the prefetched OLD prefix, update canonical pinned
    // storage using CUDA's existing quantizer, then update the device slot identically.
    // This avoids racing prefetch with host writes, and preserves arbitrary row indices,
    // speculative overwrite/reject, and prefix reuse without a second KV position ledger.
    bool write(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
        if (node->op != GGML_OP_SET_ROWS || !ggml_cuda_kv_tensor(node)) return false;
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & l = layers[i];
            const auto * root = ggml_cuda_kv_root(node);
            const bool is_k = root->data == l.k->data;
            if (!is_k && root->data != l.v->data) continue;
            const int s = i % staging->n_slots;
            if (getenv("LLAMA_KV_STAGE_TRACE")) fprintf(stderr, "kv write %zu %s\n", i, node->name);
            for (int st = 0; st < 2; ++st) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->ready[st][s], 0));
            }
            ggml_cuda_op_set_rows(ctx, node); // small, coalesced writes into mapped pinned RAM
            ggml_tensor staged = *node;
            staged.data = (char *) (is_k ? staging->k[s] : staging->v[s]) +
                          ((char *) node->data - (char *) root->data);
            ggml_cuda_op_set_rows(ctx, &staged);
            return true;
        }
        GGML_ABORT("staged KV write has no attention consumer in this CUDA graph");
    }
    bool attention(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & l = layers[i];
            if (node != l.attention) continue;
            if (getenv("LLAMA_KV_STAGE_TRACE")) fprintf(stderr, "kv attention %zu\n", i);
            const int s = i % staging->n_slots;
            for (int st = 0; st < 2; ++st) {
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->ready[st][s], 0));
            }
            ggml_tensor k = *node->src[1], v = *node->src[2], out = *node;
            k.data = (char *) staging->k[s] + ((char *) k.data - (char *) l.k->data);
            v.data = (char *) staging->v[s] + ((char *) v.data - (char *) l.v->data);
            out.src[1] = &k; out.src[2] = &v;
            if (staging->prof) {
                CUDA_CHECK(cudaEventRecord(staging->pev[i][3], ctx.stream()));
            }
            ggml_cuda_flash_attn_ext(ctx, &out);
            if (staging->prof) {
                CUDA_CHECK(cudaEventRecord(staging->pev[i][4], ctx.stream()));
            }
            // Release the slot for the next use (next eval, or a later layer of
            // this eval) only after ALL attention kernels have consumed it.
            CUDA_CHECK(cudaEventRecord(staging->consumed[s], ctx.stream()));
            const size_t ns = (size_t) staging->n_slots;
            if (i + ns < layers.size()) {
                prefetch(i + ns);
            } else if (i >= ns) {
                // Last user of this slot in this eval: stage the layer that will use
                // it first in the NEXT eval. This gives the copy a whole layer-gap to
                // run in; otherwise every token would stall one full prefix copy at
                // its leading staged attention.
                prefetch(i % ns);
            }
            return true;
        }
        return false;
    }
    // Opt-in A/B knob. Normally NOT joined: a slot staged for the next eval must be
    // free to keep copying past the end of this eval, which is what removes the
    // start-of-token stall.
    void end(ggml_backend_cuda_context & ctx) {
        if (staging && staging->prof) {
            staging->dump_prof(ctx);
        }
        static const bool join = getenv("LLAMA_KV_STAGE_JOIN") != nullptr;
        if (!staging || !join) return;
        CUDA_CHECK(cudaEventRecord(staging->join[0], staging->copy[0]));
        CUDA_CHECK(cudaEventRecord(staging->join[1], staging->copy[1]));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->join[0], 0));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->join[1], 0));
    }
};
