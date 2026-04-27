// test-qwen35-chain-capture.cpp
//
// Phase 3 acceptance test (Test 3.C) for hidden-state capture.
//
// Two modes are run in a single invocation using the same model/context:
//
//   Mode A (capture):
//     - Calls llama_set_capture_hidden(ctx, true).
//     - Decodes the prompt as a chain batch.
//     - Dumps logits for the last token to --out-logits.
//     - Reads the hidden capture buffer via llama_get_hidden_capture().
//     - Dumps the capture buffer to --out-capture.
//     - Asserts: shape is [5 * hidden_dim, n_tokens]; no NaN/Inf; not all-zero.
//
//   Mode B (regression, --no-capture):
//     - Calls llama_set_capture_hidden(ctx, false) then re-decodes same prompt.
//     - Asserts logits are BIT-EQUAL to Mode A output (same values, not just close).
//     - Skips capture dump.
//
// Both modes run inside a single process so logits can be compared in memory.
// The --out-logits file is written once (from Mode A).  If Mode B differs,
// the driver exits with code 1 and prints the first discrepant index.
//
// Build: requires -DLLAMA_BUILD_TESTS_DFLASH_DRAFT=ON (not in ctest).
//
// API assumptions (implementation agent deliverables):
//   void llama_set_capture_hidden(llama_context * ctx, bool enable)
//       -- opt-in to hidden-state capture for the target model.
//   ggml_tensor * llama_get_hidden_capture(llama_context * ctx)
//       -- returns a pointer to the capture tensor after decode.
//          tensor shape: [5 * hidden_dim, n_tokens] (F32, host-accessible).
//          Returns NULL if capture was not enabled or graph not yet run.
//
// Output binary format (--out-logits):
//   int32_t n_tokens    (= 1, last-position logit row)
//   int32_t vocab_size
//   float   logits[vocab_size]
//
// Output binary format (--out-capture):
//   int32_t feat_dim    (= 5 * hidden_dim)
//   int32_t n_tokens    (number of prompt tokens)
//   float   buf[feat_dim * n_tokens]   (row-major: row i = token i's features)

#include "llama.h"
#include "ggml.h"
#include "log.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --model PATH           (Qwen3.5-27B GGUF; required)\n"
        "  --prompt-tokens PATH   (binary int32 LE token IDs; required)\n"
        "  --out-logits PATH      (F32 binary; required)\n"
        "  --out-capture PATH     (F32 binary capture dump; required)\n"
        "  --no-capture           (skip Mode A, only run Mode B regression check)\n"
        "  --n-gpu-layers N       (default 99)\n"
        "  --n-ctx N              (default 4096)\n"
        "\n"
        "Both capture and no-capture modes run in sequence within one invocation.\n"
        "Logits from both modes are compared in memory and must be bit-equal.\n",
        prog);
}

static std::vector<int32_t> read_prompt_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open prompt-tokens file: " + path);
    }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz % sizeof(int32_t) != 0) {
        throw std::runtime_error("prompt-tokens file size not a multiple of 4: " + path);
    }
    std::vector<int32_t> tokens(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char *>(tokens.data()), sz);
    return tokens;
}

static void write_logits(const std::string & path,
                         const float * data,
                         int32_t n_tokens,
                         int32_t vocab_size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open out-logits for writing: " + path);
    }
    f.write(reinterpret_cast<const char *>(&n_tokens),   sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(&vocab_size), sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data),
            (std::streamsize)((size_t)n_tokens * vocab_size * sizeof(float)));
}

static void write_capture(const std::string & path,
                          const float * data,
                          int32_t feat_dim,
                          int32_t n_tokens) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open out-capture for writing: " + path);
    }
    f.write(reinterpret_cast<const char *>(&feat_dim),  sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(&n_tokens),  sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data),
            (std::streamsize)((size_t)feat_dim * n_tokens * sizeof(float)));
}

// Decode a prompt batch and return the logits for the last position.
// The returned vector is a copy (safe across re-use of the context).
static std::vector<float> decode_chain(llama_context * ctx,
                                       const std::vector<int32_t> & prompt,
                                       int32_t vocab_size) {
    const int32_t n_tokens = (int32_t)prompt.size();
    llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]      = (llama_token)prompt[i];
        batch.pos[i]        = (llama_pos)i;
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        // Only request logits for the last token to match Phase 1 chain mode.
        batch.logits[i]     = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed");
    }

    // Copy last-token logits before freeing batch.
    const float * row = llama_get_logits_ith(ctx, n_tokens - 1);
    std::vector<float> logits(row, row + vocab_size);

    llama_batch_free(batch);
    return logits;
}

// Clear context KV cache between the two decode runs.
static void clear_kv(llama_context * ctx) {
    llama_kv_self_clear(ctx);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt_tokens_path;
    std::string out_logits_path;
    std::string out_capture_path;
    bool        no_capture   = false;
    int32_t     n_gpu_layers = 99;
    int32_t     n_ctx        = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt-tokens" && i + 1 < argc) {
            prompt_tokens_path = argv[++i];
        } else if (arg == "--out-logits" && i + 1 < argc) {
            out_logits_path = argv[++i];
        } else if (arg == "--out-capture" && i + 1 < argc) {
            out_capture_path = argv[++i];
        } else if (arg == "--no-capture") {
            no_capture = true;
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            n_ctx = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty())         { fprintf(stderr, "--model is required\n");         return 1; }
    if (prompt_tokens_path.empty()) { fprintf(stderr, "--prompt-tokens is required\n"); return 1; }
    if (out_logits_path.empty())    { fprintf(stderr, "--out-logits is required\n");    return 1; }
    if (out_capture_path.empty() && !no_capture) {
        fprintf(stderr, "--out-capture is required (or pass --no-capture to skip Mode A)\n");
        return 1;
    }

    llama_backend_init();

    auto mparams         = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    auto cparams    = llama_context_default_params();
    cparams.n_ctx   = (uint32_t)n_ctx;
    cparams.n_batch = (uint32_t)n_ctx;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    int ret = 1;
    try {
        const auto *  vocab      = llama_model_get_vocab(model);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        const int32_t hidden_dim = llama_model_n_embd(model);
        const int32_t feat_dim   = 5 * hidden_dim;

        std::vector<int32_t> prompt = read_prompt_tokens(prompt_tokens_path);
        const int32_t n_prompt      = (int32_t)prompt.size();

        std::vector<float> logits_capture;
        std::vector<float> logits_nocapture;

        // ==================================================================
        // Mode A: capture enabled
        // ==================================================================
        if (!no_capture) {
            LOG_INF("--- Mode A: capture enabled ---\n");
            llama_set_capture_hidden(ctx, true);

            logits_capture = decode_chain(ctx, prompt, vocab_size);

            // Write logits (last token only).
            write_logits(out_logits_path, logits_capture.data(), 1, vocab_size);
            LOG_INF("Mode A: logits written to %s\n", out_logits_path.c_str());

            // Read hidden capture tensor.
            struct ggml_tensor * cap_tensor = llama_get_hidden_capture(ctx);
            if (!cap_tensor) {
                throw std::runtime_error(
                    "llama_get_hidden_capture returned NULL after capture decode; "
                    "check that llama_set_capture_hidden is wired in the graph builder");
            }

            // Validate shape: expected [feat_dim, n_tokens] or [n_tokens, feat_dim].
            // The roadmap specifies [5*hidden, n_tokens] stored as ne[0]=feat_dim, ne[1]=n_tokens.
            const int64_t ne0 = cap_tensor->ne[0];
            const int64_t ne1 = cap_tensor->ne[1];
            if (ne0 != (int64_t)feat_dim || ne1 != (int64_t)n_prompt) {
                throw std::runtime_error(
                    "hidden capture tensor shape mismatch: got [" +
                    std::to_string(ne0) + ", " + std::to_string(ne1) +
                    "], expected [" + std::to_string(feat_dim) + ", " +
                    std::to_string(n_prompt) + "]");
            }
            LOG_INF("capture shape: [%lld, %lld] — OK\n", (long long)ne0, (long long)ne1);

            // Validate: no NaN/Inf and not all-zero.
            const float * cap_data = ggml_get_data_f32(cap_tensor);
            const size_t cap_n     = (size_t)feat_dim * n_prompt;
            bool any_nonzero = false;
            for (size_t k = 0; k < cap_n; ++k) {
                float v = cap_data[k];
                if (!std::isfinite(v)) {
                    throw std::runtime_error(
                        "hidden capture contains non-finite value at index " +
                        std::to_string(k));
                }
                if (v != 0.0f) {
                    any_nonzero = true;
                }
            }
            if (!any_nonzero) {
                throw std::runtime_error(
                    "hidden capture is all-zero; capture hook is likely not wired");
            }
            LOG_INF("capture: no NaN/Inf, at least one non-zero value — OK\n");

            write_capture(out_capture_path, cap_data, feat_dim, n_prompt);
            LOG_INF("Mode A: capture written to %s\n", out_capture_path.c_str());

            clear_kv(ctx);
        }

        // ==================================================================
        // Mode B: capture disabled — must produce bit-equal logits
        // ==================================================================
        LOG_INF("--- Mode B: capture disabled ---\n");
        llama_set_capture_hidden(ctx, false);

        logits_nocapture = decode_chain(ctx, prompt, vocab_size);

        if (!no_capture) {
            // Compare bit-for-bit against Mode A.
            bool mismatch = false;
            for (int32_t v = 0; v < vocab_size; ++v) {
                if (logits_capture[v] != logits_nocapture[v]) {
                    fprintf(stderr,
                        "FAIL: logit mismatch at vocab index %d: "
                        "capture=%.8e  no-capture=%.8e\n",
                        v, logits_capture[v], logits_nocapture[v]);
                    mismatch = true;
                    break; // report first discrepancy only
                }
            }
            if (mismatch) {
                throw std::runtime_error(
                    "Mode A and Mode B logits are not bit-equal; "
                    "hidden capture hook may be altering the compute graph");
            }
            LOG_INF("Mode B: logits bit-equal to Mode A — OK\n");
        } else {
            // no-capture-only run: write logits so the caller can compare
            // against a Phase 1 golden dump externally.
            write_logits(out_logits_path, logits_nocapture.data(), 1, vocab_size);
            LOG_INF("Mode B only: logits written to %s\n", out_logits_path.c_str());
        }

        LOG_INF("all assertions passed\n");
        ret = 0;

    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        ret = 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ret;
}
