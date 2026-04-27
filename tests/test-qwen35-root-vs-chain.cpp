// test-qwen35-root-vs-chain.cpp
//
// DDTree diagnostic: verify that a single tree-mode forward at the root node
// (parent_id = -1) is equivalent to a chain forward of the same token at the
// same position.
//
// Two passes inside the same process (model loaded once):
//   pass A (chain): chain prefill tokens[0 .. N-1], record logits at index N-1
//   pass B (tree-root): chain prefill tokens[0 .. N-2], then a single
//     tree-mode batch with one node {token = tokens[N-1], parent_id = -1,
//     pos = N-1}, record logits at index 0
//
// If the tree kernel + tree input wiring are correct, A and B should match
// within numerical tolerance for that one position.
//
// Build: -DLLAMA_BUILD_TESTS_QWEN35_ROOT_VS_CHAIN=ON
//   ./build-server/bin/test-qwen35-root-vs-chain \
//       --model PATH --prompt-tokens tokens.bin --out-summary diff.txt

#include "llama.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s\n"
        "  --model PATH             (GGUF; required)\n"
        "  --prompt-tokens PATH     (binary int32 LE token IDs)\n"
        "  --prompt-text STR        (alternative to --prompt-tokens; tokenized in-process)\n"
        "  --prompt-text-file PATH  (alternative to --prompt-text; UTF-8 text file)\n"
        "  --out-summary PATH       (text summary; required)\n"
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

static std::vector<float> run_chain_capture_last(llama_model * model,
                                                 const llama_context_params & cparams,
                                                 const std::vector<int32_t> & tokens) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        throw std::runtime_error("failed to create chain context");
    }

    const int32_t n_tokens   = (int32_t)tokens.size();
    const auto * vocab       = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(n_tokens, /*embd=*/0, /*n_seq_max=*/1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i]     = (llama_token)tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        llama_free(ctx);
        throw std::runtime_error("chain llama_decode failed");
    }

    const float * row = llama_get_logits_ith(ctx, n_tokens - 1);
    std::vector<float> out(vocab_size);
    memcpy(out.data(), row, (size_t)vocab_size * sizeof(float));

    llama_batch_free(batch);
    llama_free(ctx);
    return out;
}

static std::vector<float> run_chain_then_tree_root(llama_model * model,
                                                   const llama_context_params & cparams,
                                                   const std::vector<int32_t> & tokens) {
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        throw std::runtime_error("failed to create tree-root context");
    }

    const int32_t n_tokens   = (int32_t)tokens.size();
    const auto * vocab       = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    const int32_t n_prefix   = n_tokens - 1;

    // chain prefill tokens[0 .. n_prefix - 1]
    if (n_prefix > 0) {
        llama_batch batch = llama_batch_init(n_prefix, /*embd=*/0, /*n_seq_max=*/1);
        for (int32_t i = 0; i < n_prefix; ++i) {
            batch.token[i]     = (llama_token)tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 0;
        }
        batch.n_tokens = n_prefix;

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            llama_free(ctx);
            throw std::runtime_error("chain prefill llama_decode failed");
        }
        llama_batch_free(batch);
    }

    // tree batch with one node { token = tokens[N-1], parent = -1, pos = N-1 }
    llama_batch tbatch = llama_batch_init_tree(/*n_tokens=*/1, /*embd=*/0, /*n_seq_max=*/1);
    tbatch.token[0]     = (llama_token)tokens[n_prefix];
    tbatch.pos[0]       = n_prefix;
    tbatch.n_seq_id[0]  = 1;
    tbatch.seq_id[0][0] = 0;
    tbatch.parent_id[0] = -1; // root
    tbatch.logits[0]    = 1;
    tbatch.n_tokens     = 1;

    if (llama_decode(ctx, tbatch) != 0) {
        llama_batch_free(tbatch);
        llama_free(ctx);
        throw std::runtime_error("tree-root llama_decode failed");
    }

    const float * row = llama_get_logits_ith(ctx, 0);
    std::vector<float> out(vocab_size);
    memcpy(out.data(), row, (size_t)vocab_size * sizeof(float));

    llama_batch_free(tbatch);
    llama_free(ctx);
    return out;
}

struct DiffStats {
    double max_abs_diff;
    double mean_abs_diff;
    int    argmax_a;
    int    argmax_b;
    std::vector<int> top5_a;
    std::vector<int> top5_b;
};

static std::vector<int> top_k_indices(const std::vector<float> & v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

static DiffStats diff_logits(const std::vector<float> & a, const std::vector<float> & b) {
    DiffStats s = {};
    if (a.size() != b.size() || a.empty()) {
        throw std::runtime_error("logits size mismatch");
    }
    double sum_abs = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        sum_abs += d;
        if (d > max_abs) max_abs = d;
    }
    s.max_abs_diff  = max_abs;
    s.mean_abs_diff = sum_abs / (double)a.size();
    s.argmax_a = (int)(std::max_element(a.begin(), a.end()) - a.begin());
    s.argmax_b = (int)(std::max_element(b.begin(), b.end()) - b.begin());
    s.top5_a = top_k_indices(a, 5);
    s.top5_b = top_k_indices(b, 5);
    return s;
}

static std::string read_text_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open prompt-text-file: " + path);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

static std::vector<int32_t> tokenize_text(llama_model * model, const std::string & text) {
    const auto * vocab = llama_model_get_vocab(model);
    int32_t n = -llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                                nullptr, 0, /*add_special=*/true, /*parse_special=*/false);
    if (n <= 0) {
        throw std::runtime_error("llama_tokenize sizing failed");
    }
    std::vector<llama_token> tmp(n);
    int32_t got = llama_tokenize(vocab, text.data(), (int32_t)text.size(),
                                 tmp.data(), n, true, false);
    if (got != n) {
        throw std::runtime_error("llama_tokenize result mismatch");
    }
    std::vector<int32_t> out(tmp.begin(), tmp.end());
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompt_tokens_path;
    std::string prompt_text;
    std::string prompt_text_file;
    std::string out_summary;
    int32_t n_gpu_layers = 99;
    int32_t n_ctx        = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"            && i + 1 < argc) model_path         = argv[++i];
        else if (arg == "--prompt-tokens"    && i + 1 < argc) prompt_tokens_path = argv[++i];
        else if (arg == "--prompt-text"      && i + 1 < argc) prompt_text        = argv[++i];
        else if (arg == "--prompt-text-file" && i + 1 < argc) prompt_text_file   = argv[++i];
        else if (arg == "--out-summary"      && i + 1 < argc) out_summary        = argv[++i];
        else if (arg == "--n-gpu-layers"     && i + 1 < argc) n_gpu_layers       = std::atoi(argv[++i]);
        else if (arg == "--n-ctx"            && i + 1 < argc) n_ctx              = std::atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }

    int input_modes = (!prompt_tokens_path.empty()) + (!prompt_text.empty()) + (!prompt_text_file.empty());
    if (model_path.empty() || out_summary.empty() || input_modes != 1) {
        fprintf(stderr, "must provide exactly one of --prompt-tokens, --prompt-text, --prompt-text-file\n");
        usage(argv[0]); return 1;
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

    int rc = 1;
    try {
        std::vector<int32_t> tokens;
        if (!prompt_tokens_path.empty()) {
            tokens = read_prompt_tokens(prompt_tokens_path);
        } else {
            std::string text = !prompt_text.empty()
                ? prompt_text
                : read_text_file(prompt_text_file);
            tokens = tokenize_text(model, text);
            LOG_INF("tokenized %zu tokens from text\n", tokens.size());
        }
        if (tokens.size() < 2) {
            throw std::runtime_error("need at least 2 tokens");
        }

        LOG_INF("loaded %zu tokens; running chain pass...\n", tokens.size());
        std::vector<float> A = run_chain_capture_last(model, cparams, tokens);

        LOG_INF("running chain-prefill + tree-root pass...\n");
        std::vector<float> B = run_chain_then_tree_root(model, cparams, tokens);

        DiffStats s = diff_logits(A, B);

        std::ofstream f(out_summary);
        f << "n_tokens=" << tokens.size() << "\n";
        f << "vocab_size=" << A.size() << "\n";
        f << "max_abs_diff=" << s.max_abs_diff << "\n";
        f << "mean_abs_diff=" << s.mean_abs_diff << "\n";
        f << "argmax_chain=" << s.argmax_a << "\n";
        f << "argmax_tree_root=" << s.argmax_b << "\n";
        f << "top5_chain=";
        for (int x : s.top5_a) f << x << " ";
        f << "\ntop5_tree_root=";
        for (int x : s.top5_b) f << x << " ";
        f << "\n";

        fprintf(stderr, "max_abs_diff = %.6g\n", s.max_abs_diff);
        fprintf(stderr, "mean_abs_diff = %.6g\n", s.mean_abs_diff);
        fprintf(stderr, "argmax: chain=%d tree_root=%d %s\n",
                s.argmax_a, s.argmax_b,
                s.argmax_a == s.argmax_b ? "MATCH" : "DIFF");

        rc = 0;
    } catch (const std::exception & e) {
        LOG_ERR("error: %s\n", e.what());
        rc = 1;
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}
