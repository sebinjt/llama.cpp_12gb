#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

// Private offline prefill tool. Normal contexts always execute the full model.
struct llama_prefill_range {
    int begin = 0;
    int end;
    bool enabled = false;

    explicit llama_prefill_range(int n_layer) : end(n_layer) {
        const char * value = std::getenv("LLAMA_PREFILL_LAYER_RANGE");
        if (!value || !*value) {
            return;
        }
        const std::string text(value);
        const auto colon = text.find(':');
        if (colon == std::string::npos) {
            throw std::runtime_error("LLAMA_PREFILL_LAYER_RANGE must be begin:end");
        }
        size_t a = 0, b = 0;
        begin = std::stoi(text.substr(0, colon), &a);
        end = std::stoi(text.substr(colon + 1), &b);
        if (a != colon || b != text.size() - colon - 1 || begin < 0 || begin >= end || end > n_layer) {
            throw std::runtime_error("invalid LLAMA_PREFILL_LAYER_RANGE");
        }
        enabled = true;
    }

    bool contains(int il) const { return il >= begin && il < end; }
};
