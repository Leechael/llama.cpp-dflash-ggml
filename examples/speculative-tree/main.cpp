// examples/speculative-tree/main.cpp — DDTree spec-decode CLI driver.
//
// End-to-end command-line tool that loads a target Qwen3.5-27B model and a
// dflash-draft companion model, tokenizes a prompt, runs DDTree speculative
// decoding, and prints the generated text (and optionally timing statistics).
//
// Usage:
//   llama-speculative-tree \
//       -m  <target-gguf>  -md <draft-gguf> \
//       -p  <prompt-text>  [--gen N] [--ddtree-budget N] [--temp F] \
//       [--n-gpu-layers N] [--n-ctx N] [--bench] [--out-tokens PATH]

#include "speculative-tree-driver.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// Qwen3.5 EOS token id.
static constexpr llama_token QWEN35_EOS = 248045;

struct cli_params {
    std::string model_target;
    std::string model_draft;
    std::string prompt;
    std::string prompt_tokens_path;
    std::string out_tokens_path;
    int   gen            = 64;
    int   ddtree_budget  = 22;
    bool  ddtree_chain   = true;
    float temp           = 1.0f;
    int   n_gpu_layers   = 99;
    int   n_ctx          = 4096;
    bool  bench          = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s -m PATH -md PATH [-p TEXT | --prompt-tokens PATH]\n"
        "       [--gen N] [--ddtree-budget N] [--ddtree-no-chain-seed]\n"
        "       [--temp F] [--n-gpu-layers N] [--n-ctx N]\n"
        "       [--out-tokens PATH] [--bench]\n", prog);
}

static cli_params parse_args(int argc, char ** argv) {
    cli_params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[i];
        };
        if      (a == "-m")                   p.model_target       = next();
        else if (a == "-md")                  p.model_draft        = next();
        else if (a == "-p")                   p.prompt             = next();
        else if (a == "--prompt-tokens")       p.prompt_tokens_path = next();
        else if (a == "--gen")                 p.gen                = std::stoi(next());
        else if (a == "--ddtree-budget")       p.ddtree_budget      = std::stoi(next());
        else if (a == "--ddtree-no-chain-seed") p.ddtree_chain       = false;
        else if (a == "--temp")                p.temp               = std::stof(next());
        else if (a == "--n-gpu-layers")        p.n_gpu_layers       = std::stoi(next());
        else if (a == "--n-ctx")               p.n_ctx              = std::stoi(next());
        else if (a == "--out-tokens")          p.out_tokens_path    = next();
        else if (a == "--bench")               p.bench              = true;
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); print_usage(argv[0]); exit(1); }
    }
    if (p.model_target.empty() || p.model_draft.empty()) {
        fprintf(stderr, "error: -m and -md are required\n");
        print_usage(argv[0]);
        exit(1);
    }
    if (p.prompt.empty() && p.prompt_tokens_path.empty()) {
        fprintf(stderr, "error: -p or --prompt-tokens is required\n");
        print_usage(argv[0]);
        exit(1);
    }
    return p;
}

// Write int32 LE binary.
static void write_tokens_bin(const std::string & path, const std::vector<llama_token> & toks) {
    std::ofstream f(path, std::ios::binary);
    for (llama_token t : toks) {
        int32_t v = (int32_t)t;
        f.write(reinterpret_cast<const char *>(&v), 4);
    }
}

// Read int32 LE binary.
static std::vector<llama_token> read_tokens_bin(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<llama_token> toks;
    int32_t v;
    while (f.read(reinterpret_cast<char *>(&v), 4)) {
        toks.push_back((llama_token)v);
    }
    return toks;
}

int main(int argc, char ** argv) {
    cli_params cli = parse_args(argc, argv);

    llama_backend_init();

    // Load target model.
    llama_model_params mparams_tgt = llama_model_default_params();
    mparams_tgt.n_gpu_layers = cli.n_gpu_layers;
    llama_model * model_tgt = llama_model_load_from_file(cli.model_target.c_str(), mparams_tgt);
    if (!model_tgt) {
        fprintf(stderr, "error: failed to load target model: %s\n", cli.model_target.c_str());
        return 1;
    }

    // Load draft model.
    llama_model_params mparams_dft = llama_model_default_params();
    mparams_dft.n_gpu_layers = cli.n_gpu_layers;
    llama_model * model_dft = llama_model_load_from_file(cli.model_draft.c_str(), mparams_dft);
    if (!model_dft) {
        fprintf(stderr, "error: failed to load draft model: %s\n", cli.model_draft.c_str());
        llama_model_free(model_tgt);
        return 1;
    }

    // Create target context.
    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx     = (uint32_t)cli.n_ctx;
    cparams_tgt.n_batch   = 512;
    llama_context * ctx_tgt = llama_init_from_model(model_tgt, cparams_tgt);
    if (!ctx_tgt) {
        fprintf(stderr, "error: failed to create target context\n");
        return 1;
    }

    // Enable hidden capture on target so the draft can read its features.
    llama_set_capture_hidden(ctx_tgt, true);

    // Create draft context.
    // Draft uses small n_ctx (= DRAFT_CTX_MAX + block_size) since the dflash-draft
    // model doesn't have a KV cache (it reuses target features directly).
    llama_context_params cparams_dft = llama_context_default_params();
    cparams_dft.n_ctx     = 2048 + 16; // DRAFT_CTX_MAX + block_size
    cparams_dft.n_batch   = 16;        // one block per decode
    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams_dft);
    if (!ctx_dft) {
        fprintf(stderr, "error: failed to create draft context\n");
        return 1;
    }

    // Tokenize prompt.
    std::vector<llama_token> prompt_tokens;
    if (!cli.prompt_tokens_path.empty()) {
        prompt_tokens = read_tokens_bin(cli.prompt_tokens_path);
    } else {
        const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
        const int n_prompt = llama_tokenize(vocab, cli.prompt.c_str(),
                                             (int32_t)cli.prompt.size(),
                                             nullptr, 0, /*add_special=*/true, /*parse_special=*/false);
        if (n_prompt < 0) {
            fprintf(stderr, "error: tokenize failed\n");
            return 1;
        }
        prompt_tokens.resize(n_prompt);
        llama_tokenize(vocab, cli.prompt.c_str(), (int32_t)cli.prompt.size(),
                       prompt_tokens.data(), n_prompt, true, false);
    }

    if (prompt_tokens.empty()) {
        fprintf(stderr, "error: empty prompt\n");
        return 1;
    }

    // ── Prompt prefill (chain decode on target) ────────────────────────────────
    // Decode the prompt in one batch to fill the target KV cache and
    // populate hidden_capture with the last token's layer features.
    {
        const int n_prompt = (int)prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        batch.n_tokens = n_prompt;
        for (int i = 0; i < n_prompt; ++i) {
            batch.token[i]      = prompt_tokens[i];
            batch.pos[i]        = (llama_pos)i;
            batch.n_seq_id[i]   = 1;
            batch.seq_id[i][0]  = 0;
            batch.logits[i]     = (i == n_prompt - 1) ? 1 : 0;
        }
        int ret = llama_decode(ctx_tgt, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            fprintf(stderr, "error: prompt prefill decode failed: %d\n", ret);
            return 1;
        }
    }

    // Greedy sample from last prompt token to get the first generated token.
    // This becomes the root token for spec-decode step 0.
    llama_token root_token;
    {
        const float * logits = llama_get_logits_ith(ctx_tgt, 0);
        if (!logits) { fprintf(stderr, "error: no logits after prefill\n"); return 1; }
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
        root_token = 0;
        float best = logits[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > best) { best = logits[v]; root_token = (llama_token)v; }
        }
    }

    llama_pos committed_pos = (llama_pos)prompt_tokens.size();

    // ── Build DDTree driver ───────────────────────────────────────────────────
    llama_ddtree_params ddparams;
    ddparams.budget     = cli.ddtree_budget;
    ddparams.temp       = cli.temp;
    ddparams.chain_seed = cli.ddtree_chain;
    ddparams.block_size = 16; // dflash-draft block size

    llama_speculative_tree_driver * driver =
        llama_speculative_tree_driver_init(ctx_tgt, ctx_dft, ddparams);
    if (!driver) {
        fprintf(stderr, "error: failed to init speculative tree driver\n");
        return 1;
    }

    // ── Generation loop ───────────────────────────────────────────────────────
    std::vector<llama_token> generated;
    generated.reserve((size_t)cli.gen + 16);

    int64_t total_steps  = 0;
    int64_t total_accept = 0; // sum of commit_n per step (for accept rate)

    auto t_start = std::chrono::high_resolution_clock::now();

    const llama_vocab * target_vocab = llama_model_get_vocab(model_tgt);

    while ((int)generated.size() < cli.gen && root_token != QWEN35_EOS) {
        std::vector<llama_token> accepted =
            llama_speculative_tree_driver_step(driver, root_token, committed_pos);

        if (accepted.empty()) {
            fprintf(stderr, "error: driver step returned empty result\n");
            break;
        }

        // accepted[0] = root_token (echoed)
        // accepted[1..n-2] = newly accepted draft tokens
        // accepted[n-1] = bonus token (next root)

        // Commit all tokens (root + draft accepted); hold bonus as new root.
        // accepted: [root, draft_1, ..., draft_k, bonus]
        // n_new = accept_depth = number of KV positions consumed this step.
        const int n_new    = (int)accepted.size() - 1; // excludes bonus
        root_token         = accepted.back();           // bonus = new root for next step
        committed_pos     += (llama_pos)n_new;         // advance past all committed slots

        // The first accepted token == root_token == the one we greedy-sampled from prompt
        // OR was returned as bonus from the prior step. Either way, we count it.
        for (int i = 0; i < n_new && (int)generated.size() < cli.gen; ++i) {
            generated.push_back(accepted[i]);
            if (accepted[i] == QWEN35_EOS) {
                root_token = QWEN35_EOS;
                break;
            }
        }

        total_steps++;
        total_accept += n_new;

        // Stream token text to stdout.
        if (!cli.bench) {
            for (int i = 0; i < n_new; ++i) {
                char buf[256] = {0};
                int len = llama_token_to_piece(target_vocab, accepted[i], buf, sizeof(buf)-1,
                                               /*lstrip=*/0, /*special=*/false);
                if (len > 0) { buf[len] = '\0'; fputs(buf, stdout); fflush(stdout); }
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    printf("\n");

    if (cli.bench) {
        const double tps = (double)generated.size() / elapsed_s;
        const double accept_rate = total_steps > 0
            ? (double)total_accept / (double)total_steps : 0.0;
        printf("[bench] generated=%d tokens, elapsed=%.2fs, tokens/s=%.1f, "
               "accept_rate=%.2f tokens/step, steps=%lld\n",
               (int)generated.size(), elapsed_s, tps, accept_rate,
               (long long)total_steps);
    }

    if (!cli.out_tokens_path.empty()) {
        write_tokens_bin(cli.out_tokens_path, generated);
    }

    llama_speculative_tree_driver_free(driver);
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    llama_backend_free();

    return 0;
}
