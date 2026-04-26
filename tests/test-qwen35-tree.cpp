// test-qwen35-tree.cpp
//
// Phase 1 acceptance test for DDTree tree-mode forward pass.
// Loads a Qwen3.5-27B GGUF, runs either a plain chain forward or a tree forward,
// and dumps raw F32 logits for offline comparison.
//
// Build: requires -DLLAMA_BUILD_TESTS_QWEN35_TREE=ON (not added to ctest by default).
//
// API assumptions (implementation agent deliverables):
//   - llama_batch.parent_id  : int32_t *, NULL in chain mode; -1 = root, else flat parent index
//   - llama_batch_init_tree(n_tokens, embd, n_seq_max) : like llama_batch_init but also
//     allocates parent_id array of size n_tokens
//   - llama_batch_free() frees parent_id when non-NULL
//   - llama_decode() reads batch.parent_id and dispatches tree forward when non-NULL
//
// Output binary format (--out-logits):
//   int32_t n_tokens
//   int32_t vocab_size
//   float   logits[n_tokens * vocab_size]   (row-major, little-endian)

#include "llama.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// nlohmann/json is vendored at vendor/nlohmann/json.hpp
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --mode {chain,tree}      (required)\n"
        "  --model PATH             (GGUF; required)\n"
        "  --prompt-tokens PATH     (binary int32 LE token IDs; required)\n"
        "  --tree-fixture PATH      (JSON; required in tree mode)\n"
        "  --out-logits PATH        (binary F32 output; required)\n"
        "  --n-gpu-layers N         (default 99)\n"
        "  --n-ctx N                (default 4096)\n",
        prog);
}

static std::vector<int32_t> read_prompt_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open prompt-tokens file: " + path);
    }
    f.seekg(0, std::ios::end);
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size % sizeof(int32_t) != 0) {
        throw std::runtime_error("prompt-tokens file size not a multiple of 4: " + path);
    }
    std::vector<int32_t> tokens(size / sizeof(int32_t));
    f.read(reinterpret_cast<char *>(tokens.data()), size);
    return tokens;
}

struct TreeNode {
    int32_t flat_idx;
    int32_t token_id;
    int32_t parent_idx; // -1 = root
    int32_t depth;
};

static std::vector<TreeNode> parse_tree_fixture(const std::string & path, int32_t & committed_offset) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open tree-fixture: " + path);
    }
    json j;
    f >> j;

    committed_offset = j.value("committed_offset", 0);

    std::vector<TreeNode> nodes;
    for (const auto & n : j["nodes"]) {
        TreeNode node;
        node.flat_idx   = n["flat_idx"].get<int32_t>();
        node.token_id   = n["token_id"].get<int32_t>();
        node.parent_idx = n["parent_idx"].get<int32_t>();
        node.depth      = n["depth"].get<int32_t>();
        nodes.push_back(node);
    }
    return nodes;
}

static void write_logits(const std::string & path,
                         const std::vector<float> & data,
                         int32_t n_tokens,
                         int32_t vocab_size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open out-logits for writing: " + path);
    }
    f.write(reinterpret_cast<const char *>(&n_tokens),  sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(&vocab_size), sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// chain mode: plain llama_batch forward
// ---------------------------------------------------------------------------

static int run_chain(llama_model * model,
                     llama_context * ctx,
                     const std::vector<int32_t> & prompt,
                     const std::string & out_path) {
    const int32_t n_tokens  = (int32_t)prompt.size();
    const auto * vocab      = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);

    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]       = (llama_token)prompt[i];
        batch.pos[i]         = i;
        batch.n_seq_id[i]    = 1;
        batch.seq_id[i][0]   = 0;
        batch.logits[i]      = 1; // request logits for every position
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    std::vector<float> logits_out((size_t)n_tokens * vocab_size);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        memcpy(&logits_out[(size_t)i * vocab_size], row, vocab_size * sizeof(float));
    }

    llama_batch_free(batch);
    write_logits(out_path, logits_out, n_tokens, vocab_size);
    LOG_INF("chain: wrote %d x %d logits to %s\n", n_tokens, vocab_size, out_path.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// tree mode: tree-batch forward
// ---------------------------------------------------------------------------
//
// Position assignment (Phase 1, 1D only):
//   pos[i] = committed_offset + node.depth
//
// M-RoPE 4-axis positions are deferred to Phase 3 (UNKNOWN-3 in the roadmap).
// When that work lands, pos[] will need to be a 4-tuple per token and
// llama_batch will need a corresponding multi-axis pos field.

static int run_tree(llama_model * model,
                    llama_context * ctx,
                    const std::string & fixture_path,
                    const std::string & out_path) {
    int32_t committed_offset = 0;
    std::vector<TreeNode> nodes = parse_tree_fixture(fixture_path, committed_offset);

    const int32_t n_tokens   = (int32_t)nodes.size();
    const auto * vocab       = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    // llama_batch_init_tree is the new API added by the implementation agent.
    // It behaves like llama_batch_init but additionally allocates batch.parent_id.
    llama_batch batch = llama_batch_init_tree(n_tokens, /*embd=*/0, /*n_seq_max=*/1);

    for (int32_t i = 0; i < n_tokens; ++i) {
        const TreeNode & node = nodes[i];
        batch.token[i]      = (llama_token)node.token_id;
        batch.pos[i]        = (llama_pos)(committed_offset + node.depth);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.parent_id[i]  = node.parent_idx; // -1 = root
        batch.logits[i]     = 1;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: llama_decode (tree) failed\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    std::vector<float> logits_out((size_t)n_tokens * vocab_size);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        memcpy(&logits_out[(size_t)i * vocab_size], row, vocab_size * sizeof(float));
    }

    llama_batch_free(batch);
    write_logits(out_path, logits_out, n_tokens, vocab_size);
    LOG_INF("tree: wrote %d x %d logits to %s\n", n_tokens, vocab_size, out_path.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string mode;
    std::string model_path;
    std::string prompt_tokens_path;
    std::string tree_fixture_path;
    std::string out_logits_path;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx        = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--prompt-tokens" && i + 1 < argc) {
            prompt_tokens_path = argv[++i];
        } else if (arg == "--tree-fixture" && i + 1 < argc) {
            tree_fixture_path = argv[++i];
        } else if (arg == "--out-logits" && i + 1 < argc) {
            out_logits_path = argv[++i];
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

    if (mode != "chain" && mode != "tree") {
        fprintf(stderr, "--mode must be 'chain' or 'tree'\n");
        usage(argv[0]);
        return 1;
    }
    if (model_path.empty()) {
        fprintf(stderr, "--model is required\n");
        return 1;
    }
    if (prompt_tokens_path.empty() && mode == "chain") {
        fprintf(stderr, "--prompt-tokens is required in chain mode\n");
        return 1;
    }
    if (tree_fixture_path.empty() && mode == "tree") {
        fprintf(stderr, "--tree-fixture is required in tree mode\n");
        return 1;
    }
    if (out_logits_path.empty()) {
        fprintf(stderr, "--out-logits is required\n");
        return 1;
    }

    llama_backend_init();

    auto mparams            = llama_model_default_params();
    mparams.n_gpu_layers    = n_gpu_layers;

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
        if (mode == "chain") {
            std::vector<int32_t> prompt = read_prompt_tokens(prompt_tokens_path);
            ret = run_chain(model, ctx, prompt, out_logits_path);
        } else {
            ret = run_tree(model, ctx, tree_fixture_path, out_logits_path);
        }
    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        ret = 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ret;
}
