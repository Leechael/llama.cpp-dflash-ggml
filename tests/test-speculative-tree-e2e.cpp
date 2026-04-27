// test-speculative-tree-e2e.cpp
//
// Phase 4 end-to-end acceptance test (Test 4.A) for DDTree speculative decoding.
//
// Two decode runs are performed back-to-back:
//
//   Run 1 (chain reference):
//     Load target model.  Decode the prompt as a plain chain batch.  Then loop
//     greedy-argmax N times to collect N reference tokens.  Write to --out-chain.
//
//   Run 2 (spec decode):
//     Reload a fresh target context with capture_hidden=true.  Load draft model.
//     Decode prompt as chain batch (primes hidden capture).  Init the DDTree
//     speculative driver.  Loop spec steps until N tokens are collected.
//     Write (first N) to --out-spec.
//
// Acceptance criterion (--temp 0 / greedy):
//   The first min(chain_n, spec_n) tokens MUST be bit-equal.
//   DDTree is lossless speculative decoding: each accepted draft token has
//   been verified as matching target-argmax at that position.
//
// Build: requires -DLLAMA_BUILD_TESTS_SPECULATIVE_TREE_E2E=ON (not in ctest).
//
// API assumptions (implementation agent deliverables):
//   -- From Phase 3 gap:
//   void llama_set_target_feat_raw(llama_context * ctx,
//                                  const float * data,
//                                  int64_t n_embd_fc,
//                                  int64_t ctx_len);
//
//   -- Phase 4 driver (common/speculative-tree-driver.h):
//   struct llama_speculative_tree_driver;
//   llama_speculative_tree_driver * llama_speculative_tree_driver_init(
//       llama_context * target_ctx,
//       llama_context * draft_ctx,
//       const llama_ddtree_params & params);
//   void llama_speculative_tree_driver_free(llama_speculative_tree_driver * d);
//   std::vector<llama_token> llama_speculative_tree_driver_step(
//       llama_speculative_tree_driver * d,
//       llama_token root_token,
//       llama_pos   committed_pos);
//
//   -- Phase 3 (already landed):
//   void llama_set_capture_hidden(llama_context * ctx, bool enable);
//
// Output binary format (--out-chain / --out-spec):
//   int32_t n_tokens           (number of generated tokens written)
//   int32_t tokens[n_tokens]   (little-endian int32, one per generated token)
//
// Note: --temp 0 is required for the bit-equal trajectory guarantee.
// Non-zero temperature introduces stochastic sampling and invalidates the
// comparison.

#include "llama.h"
#include "common.h"
#include "log.h"
#include "speculative-tree.h"
#include "speculative-tree-driver.h"

#include <algorithm>
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
// constants
// ---------------------------------------------------------------------------

// Qwen3.5 EOS token id.  Accept this token but stop further generation.
static constexpr llama_token QWEN35_EOS = 248045;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --target-model PATH     (Qwen3.5-27B GGUF; required)\n"
        "  --draft-model PATH      (dflash-draft GGUF; required)\n"
        "  --prompt-tokens PATH    (binary int32 LE token IDs; required)\n"
        "  --gen N                 (tokens to generate; default 32)\n"
        "  --out-spec PATH         (spec-decode output tokens, int32 LE; required)\n"
        "  --out-chain PATH        (chain-decode reference tokens, int32 LE; required)\n"
        "  --ddtree-budget N       (DDTree node budget; default 22)\n"
        "  --ddtree-no-chain-seed  (disable chain-seed heuristic; default: on)\n"
        "  --temp F                (sampling temperature; default 0.0 = greedy)\n"
        "  --n-gpu-layers N        (default 99)\n"
        "  --n-ctx N               (default 4096)\n"
        "\n"
        "Pass --temp 0 (greedy) to enable token-trajectory bit-equal assertion.\n",
        prog);
}

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open file: " + path);
    }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz % sizeof(int32_t) != 0) {
        throw std::runtime_error("file size not a multiple of 4: " + path);
    }
    std::vector<int32_t> buf(sz / sizeof(int32_t));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

static void write_token_file(const std::string & path,
                              const std::vector<llama_token> & tokens) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open for writing: " + path);
    }
    int32_t n = (int32_t)tokens.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(tokens.data()),
            (std::streamsize)(tokens.size() * sizeof(llama_token)));
}

// Decode prompt as a plain chain batch, return last-token logits (copy).
static std::vector<float> decode_chain_prompt(llama_context * ctx,
                                              const std::vector<int32_t> & prompt,
                                              int32_t vocab_size) {
    const int32_t n = (int32_t)prompt.size();
    llama_batch batch = llama_batch_init(n, /*embd=*/0, /*n_seq_max=*/1);

    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = (llama_token)prompt[i];
        batch.pos[i]       = (llama_pos)i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n - 1) ? 1 : 0;
    }
    batch.n_tokens = n;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed on prompt");
    }

    const float * row = llama_get_logits_ith(ctx, n - 1);
    std::vector<float> logits(row, row + vocab_size);
    llama_batch_free(batch);
    return logits;
}

// Greedy argmax over a logit row.
static llama_token argmax(const float * logits, int32_t vocab_size) {
    llama_token best = 0;
    float best_val   = logits[0];
    for (int32_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best     = v;
        }
    }
    return best;
}

// Decode a single token at position pos, return logits for that position.
static std::vector<float> decode_single(llama_context * ctx,
                                        llama_token tok,
                                        llama_pos pos,
                                        int32_t vocab_size) {
    llama_batch batch = llama_batch_init(1, /*embd=*/0, /*n_seq_max=*/1);
    batch.token[0]     = tok;
    batch.pos[0]       = pos;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;
    batch.n_tokens     = 1;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed on single token");
    }

    const float * row = llama_get_logits_ith(ctx, 0);
    std::vector<float> logits(row, row + vocab_size);
    llama_batch_free(batch);
    return logits;
}

// ---------------------------------------------------------------------------
// Run 1: chain reference decode
// ---------------------------------------------------------------------------

static std::vector<llama_token> run_chain(
        llama_model * model,
        const llama_context_params & cparams,
        const std::vector<int32_t> & prompt,
        int32_t gen,
        int32_t vocab_size) {

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        throw std::runtime_error("chain: failed to create target context");
    }

    std::vector<llama_token> out;
    out.reserve(gen);

    // Decode prompt; logits for last prompt token give the first generated token.
    std::vector<float> logits = decode_chain_prompt(ctx, prompt, vocab_size);

    llama_pos pos = (llama_pos)prompt.size();  // next decode position

    for (int32_t i = 0; i < gen; ++i) {
        llama_token tok = argmax(logits.data(), vocab_size);
        out.push_back(tok);
        if (tok == QWEN35_EOS) {
            LOG_INF("chain: EOS at step %d\n", i);
            break;
        }
        logits = decode_single(ctx, tok, pos, vocab_size);
        pos++;
    }

    llama_free(ctx);
    LOG_INF("chain: generated %d tokens\n", (int)out.size());
    return out;
}

// ---------------------------------------------------------------------------
// Run 2: speculative decode
// ---------------------------------------------------------------------------

static std::vector<llama_token> run_spec(
        llama_model * target_model,
        llama_model * draft_model,
        const llama_context_params & target_cparams,
        const llama_context_params & draft_cparams,
        const std::vector<int32_t> & prompt,
        int32_t gen,
        int32_t vocab_size,
        const llama_ddtree_params & ddparams) {

    // Target context with hidden capture enabled (required by the driver).
    llama_context * target_ctx = llama_init_from_model(target_model, target_cparams);
    if (!target_ctx) {
        throw std::runtime_error("spec: failed to create target context");
    }
    llama_set_capture_hidden(target_ctx, true);

    llama_context * draft_ctx = llama_init_from_model(draft_model, draft_cparams);
    if (!draft_ctx) {
        llama_free(target_ctx);
        throw std::runtime_error("spec: failed to create draft context");
    }

    // Prime hidden capture: decode prompt as chain batch on target.
    std::vector<float> prompt_logits =
        decode_chain_prompt(target_ctx, prompt, vocab_size);

    // Root token = argmax of last prompt position.
    llama_token root_token = argmax(prompt_logits.data(), vocab_size);
    llama_pos   committed_pos = (llama_pos)prompt.size();

    // Init spec driver.
    llama_speculative_tree_driver * driver =
        llama_speculative_tree_driver_init(target_ctx, draft_ctx, ddparams);
    if (!driver) {
        llama_free(draft_ctx);
        llama_free(target_ctx);
        throw std::runtime_error("spec: llama_speculative_tree_driver_init returned NULL");
    }

    // Ingest the prompt prefill capture into the driver's ring buffer.
    llama_speculative_tree_driver_ingest_prompt_capture(driver, (int32_t)prompt.size());

    std::vector<llama_token> out;
    out.reserve(gen);

    bool hit_eos = false;
    while ((int32_t)out.size() < gen && !hit_eos) {
        std::vector<llama_token> accepted =
            llama_speculative_tree_driver_step(driver, root_token, committed_pos);

        if (accepted.empty()) {
            // Driver signals terminal condition (e.g. EOS from target).
            LOG_INF("spec: driver returned empty accepted list at out_n=%d\n",
                    (int)out.size());
            break;
        }

        // Driver returns [committed_tokens..., bonus]. The bonus is the next
        // step's root_token and is NOT yet in the KV cache, so it's not part
        // of the committed output and doesn't advance committed_pos.
        const int32_t n_committed = (int32_t)accepted.size() - 1;
        for (int32_t i = 0; i < n_committed; ++i) {
            llama_token t = accepted[i];
            out.push_back(t);
            if (t == QWEN35_EOS) {
                hit_eos = true;
                break;
            }
            if ((int32_t)out.size() >= gen) {
                break;
            }
        }

        root_token    = accepted.back(); // bonus, fed as next step's tree[0]
        committed_pos += (llama_pos)n_committed;
    }

    llama_speculative_tree_driver_free(driver);
    llama_free(draft_ctx);
    llama_free(target_ctx);

    LOG_INF("spec: generated %d tokens\n", (int)out.size());
    return out;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string target_model_path;
    std::string draft_model_path;
    std::string prompt_tokens_path;
    std::string out_spec_path;
    std::string out_chain_path;
    int32_t     gen           = 32;
    int32_t     n_gpu_layers  = 99;
    int32_t     n_ctx         = 4096;
    float       temp          = 0.0f;

    llama_ddtree_params ddparams;  // defaults: budget=22, chain_seed=true
    // temp is set separately below after arg parsing

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--target-model" && i + 1 < argc) {
            target_model_path = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (arg == "--prompt-tokens" && i + 1 < argc) {
            prompt_tokens_path = argv[++i];
        } else if (arg == "--gen" && i + 1 < argc) {
            gen = std::atoi(argv[++i]);
        } else if (arg == "--out-spec" && i + 1 < argc) {
            out_spec_path = argv[++i];
        } else if (arg == "--out-chain" && i + 1 < argc) {
            out_chain_path = argv[++i];
        } else if (arg == "--ddtree-budget" && i + 1 < argc) {
            ddparams.budget = std::atoi(argv[++i]);
        } else if (arg == "--ddtree-no-chain-seed") {
            ddparams.chain_seed = false;
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
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

    if (target_model_path.empty()) { fprintf(stderr, "--target-model is required\n"); return 1; }
    if (draft_model_path.empty())  { fprintf(stderr, "--draft-model is required\n");  return 1; }
    if (prompt_tokens_path.empty()){ fprintf(stderr, "--prompt-tokens is required\n");return 1; }
    if (out_spec_path.empty())     { fprintf(stderr, "--out-spec is required\n");      return 1; }
    if (out_chain_path.empty())    { fprintf(stderr, "--out-chain is required\n");     return 1; }
    if (gen <= 0)                  { fprintf(stderr, "--gen must be > 0\n");           return 1; }

    ddparams.temp = temp;

    const bool greedy = (temp == 0.0f);
    if (!greedy) {
        fprintf(stderr,
            "warning: --temp %.4f is non-zero; token-trajectory bit-equal assertion "
            "is DISABLED (stochastic sampling makes sequences non-deterministic)\n",
            (double)temp);
    }

    llama_backend_init();

    int ret = 1;

    llama_model * target_model = nullptr;
    llama_model * draft_model  = nullptr;

    try {
        std::vector<int32_t> prompt = read_int32_file(prompt_tokens_path);
        if (prompt.empty()) {
            throw std::runtime_error("prompt-tokens file is empty");
        }

        // Load target model.
        {
            auto mparams         = llama_model_default_params();
            mparams.n_gpu_layers = n_gpu_layers;
            target_model         = llama_model_load_from_file(target_model_path.c_str(), mparams);
            if (!target_model) {
                throw std::runtime_error("failed to load target model: " + target_model_path);
            }
        }

        // Load draft model.
        {
            auto mparams         = llama_model_default_params();
            mparams.n_gpu_layers = n_gpu_layers;
            draft_model          = llama_model_load_from_file(draft_model_path.c_str(), mparams);
            if (!draft_model) {
                throw std::runtime_error("failed to load draft model: " + draft_model_path);
            }
        }

        const auto * vocab   = llama_model_get_vocab(target_model);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);

        // Context params shared by both target contexts (chain and spec runs).
        auto target_cparams    = llama_context_default_params();
        target_cparams.n_ctx   = (uint32_t)n_ctx;
        target_cparams.n_batch = (uint32_t)n_ctx;

        // Draft context params.
        auto draft_cparams    = llama_context_default_params();
        draft_cparams.n_ctx   = (uint32_t)n_ctx;
        draft_cparams.n_batch = (uint32_t)n_ctx;

        // ---------------------------------------------------------------
        // Run 1: chain reference
        // ---------------------------------------------------------------
        LOG_INF("=== Run 1: chain reference decode ===\n");
        std::vector<llama_token> chain_tokens =
            run_chain(target_model, target_cparams, prompt, gen, vocab_size);

        write_token_file(out_chain_path, chain_tokens);
        LOG_INF("chain: wrote %d tokens to %s\n",
                (int)chain_tokens.size(), out_chain_path.c_str());

        // ---------------------------------------------------------------
        // Run 2: speculative decode
        // ---------------------------------------------------------------
        LOG_INF("=== Run 2: speculative decode ===\n");
        std::vector<llama_token> spec_tokens =
            run_spec(target_model, draft_model,
                     target_cparams, draft_cparams,
                     prompt, gen, vocab_size, ddparams);

        // Truncate to gen if the driver produced more tokens than requested.
        if ((int32_t)spec_tokens.size() > gen) {
            spec_tokens.resize(gen);
        }

        write_token_file(out_spec_path, spec_tokens);
        LOG_INF("spec: wrote %d tokens to %s\n",
                (int)spec_tokens.size(), out_spec_path.c_str());

        // ---------------------------------------------------------------
        // Compare trajectories
        // ---------------------------------------------------------------
        const int32_t chain_n = (int32_t)chain_tokens.size();
        const int32_t spec_n  = (int32_t)spec_tokens.size();
        const int32_t cmp_n   = std::min(chain_n, spec_n);

        int32_t first_divergence = -1;
        int32_t match_count      = 0;
        for (int32_t k = 0; k < cmp_n; ++k) {
            if (chain_tokens[k] == spec_tokens[k]) {
                match_count++;
            } else if (first_divergence < 0) {
                first_divergence = k;
                break;
            }
        }

        if (first_divergence < 0 && match_count == cmp_n) {
            // All positions matched.
            printf("chain_n=%d spec_n=%d first_divergence=none bytes_match=%d/%d\n",
                   chain_n, spec_n, match_count, cmp_n);
        } else {
            printf("chain_n=%d spec_n=%d first_divergence=%d bytes_match=%d/%d\n",
                   chain_n, spec_n, first_divergence, match_count, cmp_n);
        }

        if (greedy) {
            if (first_divergence >= 0) {
                fprintf(stderr,
                    "FAIL: token-trajectory divergence at position %d "
                    "(greedy decoding MUST produce bit-equal sequences)\n"
                    "  chain[%d] = %d\n"
                    "  spec[%d]  = %d\n",
                    first_divergence,
                    first_divergence, (int)chain_tokens[first_divergence],
                    first_divergence, (int)spec_tokens[first_divergence]);
                ret = 1;
            } else {
                LOG_INF("PASS: all %d token positions are bit-equal\n", cmp_n);
                ret = 0;
            }
        } else {
            // Non-greedy: no hard assertion, just report.
            LOG_INF("non-greedy mode: token-trajectory comparison is informational only\n");
            ret = 0;
        }

    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        ret = 1;
    }

    if (draft_model)  { llama_model_free(draft_model);  }
    if (target_model) { llama_model_free(target_model); }
    llama_backend_free();
    return ret;
}
