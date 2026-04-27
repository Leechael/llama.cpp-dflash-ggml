// test-dflash-draft.cpp
//
// Phase 3 acceptance test (Test 3.B) for the dflash-draft model forward pass.
// Loads a Qwen3.5-27B target model and a dflash-draft GGUF, looks up token
// embeddings for a 16-token batch ([last_tok, MASK_TOKEN_ID*15]), reads a
// target-hidden-state feature binary, runs a single draft forward, and dumps
// the 16-position logits.
//
// Build: requires -DLLAMA_BUILD_TESTS_DFLASH_DRAFT=ON (not added to ctest).
//
// API assumptions (implementation agent deliverables):
//   LLM_ARCH_DFLASH_DRAFT  -- arch string "dflash-draft"
//   llama_model_token_embd_lookup(model, tokens, n, out_buf, embd_dim)
//       -- fills out_buf with n rows of embd_dim F32 values from model's token
//          embedding table.  out_buf must be caller-allocated (n * embd_dim floats).
//   llama_set_capture_hidden(ctx, bool) -- opt-in to hidden-capture in target
//       model; not needed here but shares the header.
//   llama_get_hidden_capture(ctx) -- not used here; see test-qwen35-chain-capture.
//
//   Draft forward with target_feat injection:
//     The draft graph builder (llm_build_dflash_draft) reads a named graph
//     input tensor "dflash_target_feat" of shape [ctx_len, n_embd*5] from the
//     batch's embd pointer.  The caller feeds it by creating a batch with
//     embd != NULL, where embd points to:
//       [ token_embd_row (embd_dim floats)  *  16   (token rows) ]
//       [ target_feat    (5*embd_dim floats) * ctx_len (feature rows) ]
//     The exact layout is defined by the implementation agent.  If that layout
//     differs from the above, the user will reconcile before compiling.
//
// Output binary format (--out-logits):
//   int32_t n_tokens    (= 16)
//   int32_t vocab_size
//   float   logits[16 * vocab_size]   (row-major, little-endian)

#include "llama.h"
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
// constants
// ---------------------------------------------------------------------------

static constexpr int32_t DRAFT_BATCH_SIZE    = 16;
static constexpr int32_t DEFAULT_MASK_TOK_ID = 248070;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --target-model PATH      (Qwen3.5-27B GGUF; source of token_embd lookup; required)\n"
        "  --draft-model  PATH      (dflash-draft GGUF; required)\n"
        "  --last-tok N             (int32 token id; required)\n"
        "  --target-feat-bin PATH   (F32 binary [ctx_len * 5 * embd_dim]; required)\n"
        "  --ctx-len N              (number of positions in target-feat-bin; 0 = derive from file)\n"
        "  --out-logits PATH        (F32 binary output for 16 positions; required)\n"
        "  --mask-token-id N        (override mask token id; default %d)\n"
        "  --n-gpu-layers N         (default 99)\n",
        prog, DEFAULT_MASK_TOK_ID);
}

// Read a raw F32 binary file into a host buffer.
static std::vector<float> read_f32_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open binary file: " + path);
    }
    f.seekg(0, std::ios::end);
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz % sizeof(float) != 0) {
        throw std::runtime_error("binary file size not a multiple of 4: " + path);
    }
    std::vector<float> buf(sz / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), sz);
    return buf;
}

static void write_logits(const std::string & path,
                         const std::vector<float> & data,
                         int32_t n_tokens,
                         int32_t vocab_size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open out-logits for writing: " + path);
    }
    f.write(reinterpret_cast<const char *>(&n_tokens),   sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(&vocab_size), sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data.data()),
            (std::streamsize)(data.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string target_model_path;
    std::string draft_model_path;
    std::string target_feat_path;
    std::string out_logits_path;
    int32_t last_tok      = -1;
    int32_t ctx_len       = 0;
    int32_t mask_tok_id   = DEFAULT_MASK_TOK_ID;
    int32_t n_gpu_layers  = 99;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--target-model" && i + 1 < argc) {
            target_model_path = argv[++i];
        } else if (arg == "--draft-model" && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (arg == "--last-tok" && i + 1 < argc) {
            last_tok = std::atoi(argv[++i]);
        } else if (arg == "--target-feat-bin" && i + 1 < argc) {
            target_feat_path = argv[++i];
        } else if (arg == "--ctx-len" && i + 1 < argc) {
            ctx_len = std::atoi(argv[++i]);
        } else if (arg == "--out-logits" && i + 1 < argc) {
            out_logits_path = argv[++i];
        } else if (arg == "--mask-token-id" && i + 1 < argc) {
            mask_tok_id = std::atoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            n_gpu_layers = std::atoi(argv[++i]);
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
    if (last_tok < 0)              { fprintf(stderr, "--last-tok is required\n");      return 1; }
    if (target_feat_path.empty())  { fprintf(stderr, "--target-feat-bin is required\n"); return 1; }
    if (out_logits_path.empty())   { fprintf(stderr, "--out-logits is required\n");    return 1; }

    llama_backend_init();

    auto mparams         = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    // ------------------------------------------------------------------
    // Step 1: load target model (for token_embd lookup only)
    // ------------------------------------------------------------------
    llama_model * target_model = llama_model_load_from_file(target_model_path.c_str(), mparams);
    if (!target_model) {
        LOG_ERR("failed to load target model: %s\n", target_model_path.c_str());
        llama_backend_free();
        return 1;
    }

    // ------------------------------------------------------------------
    // Step 2: load draft model
    // ------------------------------------------------------------------
    llama_model * draft_model = llama_model_load_from_file(draft_model_path.c_str(), mparams);
    if (!draft_model) {
        LOG_ERR("failed to load draft model: %s\n", draft_model_path.c_str());
        llama_model_free(target_model);
        llama_backend_free();
        return 1;
    }

    int ret = 1;
    try {
        // ------------------------------------------------------------------
        // Step 3: derive embd_dim from target model
        // ------------------------------------------------------------------
        // llama_model_n_embd returns the embedding dimension of the model.
        const int32_t embd_dim = llama_model_n_embd(target_model);
        LOG_INF("embd_dim = %d\n", embd_dim);

        // ------------------------------------------------------------------
        // Step 4: look up token embeddings for the 16-token batch
        //   tokens: [last_tok, mask_tok_id, mask_tok_id, ..., mask_tok_id]
        //           (1 + 15 = 16 tokens)
        // ------------------------------------------------------------------
        std::vector<llama_token> batch_tokens(DRAFT_BATCH_SIZE);
        batch_tokens[0] = (llama_token)last_tok;
        for (int i = 1; i < DRAFT_BATCH_SIZE; ++i) {
            batch_tokens[i] = (llama_token)mask_tok_id;
        }

        // out_embd: [DRAFT_BATCH_SIZE * embd_dim] floats
        std::vector<float> token_embd((size_t)DRAFT_BATCH_SIZE * embd_dim, 0.0f);
        llama_model_token_embd_lookup(target_model,
                                      batch_tokens.data(),
                                      DRAFT_BATCH_SIZE,
                                      token_embd.data(),
                                      embd_dim);
        LOG_INF("token_embd lookup done (%d tokens x %d dim)\n", DRAFT_BATCH_SIZE, embd_dim);

        // ------------------------------------------------------------------
        // Step 5: read target_feat binary
        // ------------------------------------------------------------------
        std::vector<float> target_feat = read_f32_bin(target_feat_path);

        // Derive or validate ctx_len.
        // Expected layout: [ctx_len * 5 * embd_dim] floats
        const int32_t feat_width = 5 * embd_dim;
        if (ctx_len == 0) {
            if ((int32_t)target_feat.size() % feat_width != 0) {
                throw std::runtime_error(
                    "target-feat-bin size not divisible by 5*embd_dim=" +
                    std::to_string(feat_width));
            }
            ctx_len = (int32_t)(target_feat.size() / feat_width);
            LOG_INF("derived ctx_len = %d from target-feat-bin\n", ctx_len);
        } else {
            const size_t expected = (size_t)ctx_len * feat_width;
            if (target_feat.size() != expected) {
                throw std::runtime_error(
                    "target-feat-bin has " + std::to_string(target_feat.size()) +
                    " floats, expected " + std::to_string(expected) +
                    " (ctx_len=" + std::to_string(ctx_len) +
                    " * feat_width=" + std::to_string(feat_width) + ")");
            }
        }
        LOG_INF("target_feat: %d positions x %d floats\n", ctx_len, feat_width);

        // ------------------------------------------------------------------
        // Step 6: init draft context
        //
        // n_ctx must cover both the draft batch (16) and the target feat
        // positions (ctx_len).  Use the larger of the two.
        // ------------------------------------------------------------------
        const int32_t n_ctx_draft = std::max(ctx_len, DRAFT_BATCH_SIZE) + 64;
        auto cparams      = llama_context_default_params();
        cparams.n_ctx     = (uint32_t)n_ctx_draft;
        cparams.n_batch   = (uint32_t)DRAFT_BATCH_SIZE;

        llama_context * draft_ctx = llama_init_from_model(draft_model, cparams);
        if (!draft_ctx) {
            throw std::runtime_error("failed to create draft context");
        }

        // ------------------------------------------------------------------
        // Step 7: build the embedding input buffer for the draft forward.
        //
        // The draft graph builder expects an embd batch where the embd pointer
        // contains the concatenation of:
        //   [token_embd rows: DRAFT_BATCH_SIZE * embd_dim floats]
        //   [target_feat    : ctx_len * 5 * embd_dim floats      ]
        //
        // The batch is created with embd != 0 so llama_decode dispatches the
        // embd path.  Token IDs are left unset (embd takes precedence).
        //
        // NOTE: This layout is the current best guess from the roadmap.  If
        // the implementation agent uses a different mechanism (e.g., a separate
        // set_target_feat() call), the user will reconcile and update this
        // driver before compiling.
        // ------------------------------------------------------------------
        const size_t embd_buf_floats =
            (size_t)DRAFT_BATCH_SIZE * embd_dim +
            (size_t)ctx_len * feat_width;

        std::vector<float> embd_buf(embd_buf_floats);
        // Copy token embeddings first
        memcpy(embd_buf.data(),
               token_embd.data(),
               (size_t)DRAFT_BATCH_SIZE * embd_dim * sizeof(float));
        // Then target_feat
        memcpy(embd_buf.data() + (size_t)DRAFT_BATCH_SIZE * embd_dim,
               target_feat.data(),
               (size_t)ctx_len * feat_width * sizeof(float));

        // Build a batch that feeds embeddings directly.
        // embd = 1 tells llama_batch_init to allocate an embd array; however
        // we want to point at our own buffer, so we create the struct manually.
        llama_batch batch;
        memset(&batch, 0, sizeof(batch));
        batch.n_tokens = DRAFT_BATCH_SIZE;
        // embd points to our concatenated buffer
        batch.embd     = embd_buf.data();

        // Allocate ancillary arrays on the stack/heap.
        std::vector<llama_pos>      pos_arr(DRAFT_BATCH_SIZE);
        std::vector<int32_t>        n_seq_id_arr(DRAFT_BATCH_SIZE, 1);
        std::vector<llama_seq_id>   seq_id_val(DRAFT_BATCH_SIZE, 0);
        std::vector<llama_seq_id *> seq_id_arr(DRAFT_BATCH_SIZE);
        std::vector<int8_t>         logits_arr(DRAFT_BATCH_SIZE, 1);

        for (int i = 0; i < DRAFT_BATCH_SIZE; ++i) {
            pos_arr[i]    = (llama_pos)i;
            seq_id_arr[i] = &seq_id_val[i];
        }
        batch.pos      = pos_arr.data();
        batch.n_seq_id = n_seq_id_arr.data();
        batch.seq_id   = seq_id_arr.data();
        batch.logits   = logits_arr.data();

        // ------------------------------------------------------------------
        // Step 8: run draft forward
        // ------------------------------------------------------------------
        if (llama_decode(draft_ctx, batch) != 0) {
            llama_free(draft_ctx);
            throw std::runtime_error("llama_decode (draft) failed");
        }

        // ------------------------------------------------------------------
        // Step 9: collect logits for all 16 positions and dump
        // ------------------------------------------------------------------
        const auto * vocab     = llama_model_get_vocab(draft_model);
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);

        std::vector<float> logits_out((size_t)DRAFT_BATCH_SIZE * vocab_size);
        for (int i = 0; i < DRAFT_BATCH_SIZE; ++i) {
            const float * row = llama_get_logits_ith(draft_ctx, i);
            memcpy(&logits_out[(size_t)i * vocab_size], row,
                   vocab_size * sizeof(float));
        }

        llama_free(draft_ctx);
        write_logits(out_logits_path, logits_out, DRAFT_BATCH_SIZE, vocab_size);
        LOG_INF("draft forward done: wrote %d x %d logits to %s\n",
                DRAFT_BATCH_SIZE, vocab_size, out_logits_path.c_str());
        ret = 0;

    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        ret = 1;
    }

    llama_model_free(draft_model);
    llama_model_free(target_model);
    llama_backend_free();
    return ret;
}
