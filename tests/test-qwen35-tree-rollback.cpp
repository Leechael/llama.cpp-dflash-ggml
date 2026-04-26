// test-qwen35-tree-rollback.cpp
//
// Phase 2 acceptance test for DDTree snapshot/restore symmetry (Test 2.A).
// Loads a Qwen3.5-27B GGUF, decodes a prompt chain, takes a recurrent-state
// snapshot, runs N decode steps, restores the snapshot, runs the same N steps
// again from the same starting token, and dumps both runs' final logits.
// The two logit files must be bit-equal (--abs-tol 0 with compare_logits.py).
//
// Build: requires -DLLAMA_BUILD_TESTS_QWEN35_TREE_ROLLBACK=ON (not in ctest).
//
// API assumptions (implementation agent deliverables):
//   typedef int32_t llama_mem_snapshot_id;
//   llama_mem_snapshot_id llama_seq_snapshot(llama_context *, llama_seq_id);
//   bool                  llama_seq_restore (llama_context *, llama_mem_snapshot_id);
//   void                  llama_seq_release (llama_context *, llama_mem_snapshot_id);
//
// Output binary format (--out-logits-pre / --out-logits-post):
//   int32_t n_tokens          (= 1, the single last-step logit row)
//   int32_t vocab_size
//   float   logits[vocab_size]

#include "llama.h"
#include "log.h"

#include <algorithm>
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
        "  --gen N                (chain decode steps per run; default 8)\n"
        "  --out-logits-pre PATH  (logit dump from first run; required)\n"
        "  --out-logits-post PATH (logit dump from second run after restore; required)\n"
        "  --n-gpu-layers N       (default 99)\n"
        "  --n-ctx N              (default 4096)\n",
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
    f.write(reinterpret_cast<const char *>(&n_tokens),  sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(&vocab_size), sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data), (std::streamsize)(n_tokens * vocab_size * sizeof(float)));
}

// Return the argmax token id from a logits row.
static llama_token argmax(const float * logits, int32_t vocab_size) {
    return (llama_token)(std::max_element(logits, logits + vocab_size) - logits);
}

// Decode a single token at the given position and return the logits pointer.
// The returned pointer is valid until the next llama_decode call.
static const float * decode_single(llama_context * ctx,
                                   llama_token tok,
                                   llama_pos pos,
                                   int32_t vocab_size) {
    llama_batch batch  = llama_batch_init(1, /*embd=*/0, /*n_seq_max=*/1);
    batch.token[0]     = tok;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed for single token");
    }

    const float * row = llama_get_logits_ith(ctx, 0);
    // Copy before freeing the batch (logits buffer owned by context, not batch)
    llama_batch_free(batch);
    (void)vocab_size; // size used by caller
    return row;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt_tokens_path;
    std::string out_logits_pre_path;
    std::string out_logits_post_path;
    int32_t gen          = 8;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx        = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt-tokens" && i + 1 < argc) {
            prompt_tokens_path = argv[++i];
        } else if (arg == "--gen" && i + 1 < argc) {
            gen = std::atoi(argv[++i]);
        } else if (arg == "--out-logits-pre" && i + 1 < argc) {
            out_logits_pre_path = argv[++i];
        } else if (arg == "--out-logits-post" && i + 1 < argc) {
            out_logits_post_path = argv[++i];
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

    if (model_path.empty()) {
        fprintf(stderr, "--model is required\n");
        return 1;
    }
    if (prompt_tokens_path.empty()) {
        fprintf(stderr, "--prompt-tokens is required\n");
        return 1;
    }
    if (out_logits_pre_path.empty()) {
        fprintf(stderr, "--out-logits-pre is required\n");
        return 1;
    }
    if (out_logits_post_path.empty()) {
        fprintf(stderr, "--out-logits-post is required\n");
        return 1;
    }
    if (gen < 1) {
        fprintf(stderr, "--gen must be >= 1\n");
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
        const auto * vocab    = llama_model_get_vocab(model);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);

        // ------------------------------------------------------------------
        // Step 1: decode the prompt as a single chain batch to prime state
        // ------------------------------------------------------------------
        std::vector<int32_t> prompt = read_prompt_tokens(prompt_tokens_path);
        const int32_t n_prompt = (int32_t)prompt.size();

        {
            llama_batch batch = llama_batch_init(n_prompt, /*embd=*/0, /*n_seq_max=*/1);
            for (int32_t i = 0; i < n_prompt; ++i) {
                batch.token[i]      = (llama_token)prompt[i];
                batch.pos[i]        = (llama_pos)i;
                batch.n_seq_id[i]   = 1;
                batch.seq_id[i][0]  = 0;
                batch.logits[i]     = (i == n_prompt - 1) ? 1 : 0;
            }
            batch.n_tokens = n_prompt;

            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("prompt decode failed");
            }
            llama_batch_free(batch);
        }
        LOG_INF("prompt decoded (%d tokens)\n", n_prompt);

        // ------------------------------------------------------------------
        // Step 2: snapshot the recurrent state BEFORE the decode loop
        // The first token for both runs is the argmax of the prompt's last
        // position logits, captured now so both runs start identically.
        // ------------------------------------------------------------------
        const float * prompt_logits = llama_get_logits_ith(ctx, n_prompt - 1);
        llama_token tok_first = argmax(prompt_logits, vocab_size);
        LOG_INF("first token after prompt: %d\n", (int)tok_first);

        llama_mem_snapshot_id snap = llama_seq_snapshot(ctx, /*seq_id=*/0);
        if (snap < 0) {
            throw std::runtime_error("llama_seq_snapshot returned negative id");
        }
        LOG_INF("snapshot id: %d\n", (int)snap);

        // ------------------------------------------------------------------
        // Step 3: run K decode steps (run 1), save logits of last step
        // ------------------------------------------------------------------
        std::vector<float> last_logits_pre((size_t)vocab_size);
        {
            llama_token cur = tok_first;
            llama_pos   pos = (llama_pos)n_prompt;
            for (int step = 0; step < gen; ++step) {
                const float * row = decode_single(ctx, cur, pos, vocab_size);
                if (step == gen - 1) {
                    memcpy(last_logits_pre.data(), row, vocab_size * sizeof(float));
                }
                cur = argmax(row, vocab_size);
                ++pos;
            }
        }
        write_logits(out_logits_pre_path, last_logits_pre.data(), 1, vocab_size);
        LOG_INF("run 1 complete, logits written to %s\n", out_logits_pre_path.c_str());

        // ------------------------------------------------------------------
        // Step 4: restore snapshot and run K steps again with same first token
        // ------------------------------------------------------------------
        if (!llama_seq_restore(ctx, snap)) {
            throw std::runtime_error("llama_seq_restore failed");
        }
        LOG_INF("snapshot restored\n");

        std::vector<float> last_logits_post((size_t)vocab_size);
        {
            llama_token cur = tok_first;  // same first token as run 1
            llama_pos   pos = (llama_pos)n_prompt;
            for (int step = 0; step < gen; ++step) {
                const float * row = decode_single(ctx, cur, pos, vocab_size);
                if (step == gen - 1) {
                    memcpy(last_logits_post.data(), row, vocab_size * sizeof(float));
                }
                cur = argmax(row, vocab_size);
                ++pos;
            }
        }
        write_logits(out_logits_post_path, last_logits_post.data(), 1, vocab_size);
        LOG_INF("run 2 complete, logits written to %s\n", out_logits_post_path.c_str());

        // ------------------------------------------------------------------
        // Step 5: release snapshot
        // ------------------------------------------------------------------
        llama_seq_release(ctx, snap);
        LOG_INF("snapshot released\n");

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
