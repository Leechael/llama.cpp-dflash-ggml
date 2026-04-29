// speculative-tree-driver.cpp — Phase 4 DDTree spec-decode step coordinator.
//
// Mirrors the main loop in test_dflash.cpp:1070-1500 using the llama.cpp public API.
//
// Target_feat layout (from qwen35.cpp capture):
//   The hidden capture tensor is [n_embd, 5*n_tokens] (column-major ggml / row-major C).
//   Layer k's hidden for all decoded positions occupies columns [k*n_total .. (k+1)*n_total).
//   For the draft window, we take the most recent ctx_len positions per layer and pack
//   them into [5*n_embd, ctx_len]:
//     row l*n_embd .. (l+1)*n_embd - 1 = layer l's hidden across ctx_len positions.
//   This matches what dflash-draft.cpp's fc projection expects.
//
// SSM rollback strategy (Phase 2.4):
//   If the accepted node is DFS-last, the live recurrent state already points at the accepted
//   state and copying from persist buffers only adds risk. Non-DFS-last persist rollback is
//   not yet proven correct for DFlash, so the driver falls back to
//   snapshot+restore+chain-replay for that case.

#include "speculative-tree-driver.h"
#include "speculative-tree.h"
#include "log.h"

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

// Maximum target-context window that the draft can attend over.
// Matches test_dflash.cpp:1086 DRAFT_CTX_MAX. The server-port default is
// smaller; raise it with LLAMA_DDTREE_TARGET_FEAT_CTX when needed.
static constexpr int DRAFT_CTX_MAX     = 2048;
static constexpr int DRAFT_CTX_DEFAULT = 128;

// EOS token for Qwen3.5 family.
static constexpr llama_token QWEN35_EOS = 248045;

using ddtree_clock = std::chrono::steady_clock;

static double elapsed_ms(ddtree_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(ddtree_clock::now() - t0).count();
}

struct llama_speculative_tree_driver {
    llama_context * target_ctx = nullptr;
    llama_context * draft_ctx  = nullptr;
    llama_ddtree_params params;

    // n_embd from the target model (for hidden capture slicing).
    int64_t n_embd = 0;
    // vocabulary size (for logit indexing).
    int64_t n_vocab = 0;
    // draft block_size (number of noise tokens per step, typically 16).
    int64_t block_size = 0;
    // mask token id used to fill noise positions in the draft input.
    llama_token mask_token_id = 0;

    // Scratch buffer for packed target_feat: [5*n_embd, ctx_len]
    std::vector<float> target_feat_buf;

    // Cumulative target_feat sliding ring buffer, [5*n_embd, target_feat_cap]
    // Stored column-major: column t = position t, rows = [l*n_embd .. (l+1)*n_embd) for layer l.
    // i.e. ring[(logical_col % cap) * target_feat_n_embd_fc + l*n_embd .. +n_embd] =
    // layer l at committed pos logical_col.
    std::vector<float> target_feat_ring; // size = target_feat_n_embd_fc * target_feat_cap
    int64_t target_feat_n_committed = 0; // total committed positions appended to the ring, not capped
    int64_t target_feat_n_embd_fc   = 0; // = 5 * n_embd
    int64_t target_feat_cap         = DRAFT_CTX_DEFAULT; // target feature context retained for draft

    // Scratch buffers
    std::vector<float>   top_log_probs; // [block_size-1, K]
    std::vector<int32_t> top_token_ids; // [block_size-1, K]
    std::vector<float>   noise_embd_buf; // [block_size * n_embd]
    std::vector<int32_t> posterior;      // [N] argmax per tree node

    llama_speculative_tree_driver_stats stats;
    bool fast_rollback_unavailable = false;
};

static bool ddtree_fast_batched_enabled() {
    const char * e = std::getenv("LLAMA_DDTREE_FAST_BATCHED");
    return e != nullptr && e[0] == '1';
}

static bool ddtree_fast_rollback_enabled() {
    const char * e = std::getenv("LLAMA_DDTREE_FAST_ROLLBACK");
    return e != nullptr && e[0] == '1';
}

static bool ddtree_snapshot_fallback_enabled() {
    const char * e = std::getenv("LLAMA_DDTREE_SNAPSHOT_FALLBACK");
    return e == nullptr || e[0] != '0';
}

static int64_t ddtree_target_feat_cap() {
    const char * e = std::getenv("LLAMA_DDTREE_TARGET_FEAT_CTX");
    if (!e || e[0] == '\0') {
        return DRAFT_CTX_DEFAULT;
    }

    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end == e || v <= 0) {
        return DRAFT_CTX_DEFAULT;
    }

    return std::min<int64_t>(DRAFT_CTX_MAX, std::max<int64_t>(1, (int64_t)v));
}

llama_speculative_tree_driver * llama_speculative_tree_driver_init(
        llama_context            * target_ctx,
        llama_context            * draft_ctx,
        const llama_ddtree_params & params) {

    const llama_model * target_model = llama_get_model(target_ctx);

    if (!target_model || !llama_get_model(draft_ctx)) {
        LOG_ERR("%s: null model pointer\n", __func__);
        return nullptr;
    }

    auto * d = new llama_speculative_tree_driver;
    d->target_ctx = target_ctx;
    d->draft_ctx  = draft_ctx;
    d->params     = params;

    d->n_embd    = llama_model_n_embd(target_model);
    // n_vocab: use target vocab; draft shares the same lm_head.
    const llama_vocab * target_vocab = llama_model_get_vocab(target_model);
    d->n_vocab   = (target_vocab != nullptr) ? llama_vocab_n_tokens(target_vocab) : 0;
    if (d->n_embd <= 0 || d->n_vocab <= 0) {
        LOG_ERR("%s: invalid model dimensions n_embd=%lld n_vocab=%lld\n",
                        __func__, (long long)d->n_embd, (long long)d->n_vocab);
        delete d;
        return nullptr;
    }
    // block_size and mask_token_id are constants baked into the dflash-draft checkpoint.
    // Qwen3.5-27B-DFlash always uses block_size=16 and mask_token_id=248070.
    d->block_size    = (int64_t)params.block_size; // from llama_ddtree_params (default 16)
    d->mask_token_id = 248070; // dflash-draft MASK token

    // Initialize cumulative target_feat ring buffer.
    d->target_feat_n_embd_fc   = 5 * d->n_embd;
    d->target_feat_n_committed = 0;
    d->target_feat_cap         = ddtree_target_feat_cap();
    d->target_feat_ring.assign((size_t)d->target_feat_n_embd_fc * d->target_feat_cap, 0.0f);

    return d;
}

void llama_speculative_tree_driver_free(llama_speculative_tree_driver * d) {
    delete d;
}

llama_speculative_tree_driver_stats llama_speculative_tree_driver_get_stats(
        const llama_speculative_tree_driver * d) {
    return d ? d->stats : llama_speculative_tree_driver_stats{};
}

int32_t llama_speculative_tree_driver_context_window() {
    return (int32_t)ddtree_target_feat_cap();
}

// Pack the hidden capture buffer into [5*n_embd, ctx_len] F32.
// capture   : [n_embd, 5*n_total] — ggml column-major, row-major in C means ne[0]=n_embd.
// n_total   : total positions in the capture tensor (= ne[1] / 5).
// ctx_len   : number of most-recent positions to pack (ctx_len <= n_total).
// out       : caller-allocated [5*n_embd * ctx_len] F32.
static void pack_target_feat(
        const float * capture, int64_t n_embd, int64_t n_total, int64_t ctx_len, float * out) {
    const int64_t start = n_total - ctx_len; // first column to include
    for (int64_t l = 0; l < 5; ++l) {
        for (int64_t t = 0; t < ctx_len; ++t) {
            // Source: capture column (start+t) in layer l's block.
            // In C memory (n_embd fastest): capture[(l*n_total + start + t) * n_embd .. +n_embd)
            const float * src = capture + (l * n_total + start + t) * n_embd;
            // Destination: row l*n_embd in the output, column t.
            // Output is [5*n_embd, ctx_len] → in C: out[t * 5*n_embd + l*n_embd .. +n_embd)
            float * dst = out + (size_t)t * 5 * n_embd + (size_t)l * n_embd;
            memcpy(dst, src, (size_t)n_embd * sizeof(float));
        }
    }
}

// Append hidden capture data from target_ctx into the driver's ring buffer.
// dfs_indices: if non-NULL, selects which capture columns to ingest (the DFS accepted indices).
//              if NULL, ingest the first n_dfs columns linearly (prompt prefill path).
// n_dfs: number of columns to ingest.
enum class ingest_source {
    prompt,
    tree,
    replay,
};

static int32_t driver_ingest_capture(llama_speculative_tree_driver * d,
                                     const int32_t * dfs_indices,
                                     int32_t         n_dfs,
                                     ingest_source   source) {
    const auto t0 = ddtree_clock::now();
    int64_t ne0 = 0, ne1 = 0;
    const float * capture = llama_get_hidden_capture_data(d->target_ctx, &ne0, &ne1);
    if (!capture || ne0 == 0 || ne1 == 0) {
        LOG_ERR("%s: no hidden capture data available\n", __func__);
        d->stats.t_ingest_capture_ms += elapsed_ms(t0);
        return 0;
    }
    // capture layout: [n_embd, 5*n_tokens] → ne0=n_embd, ne1=5*n_tokens
    const int64_t n_embd  = ne0;
    const int64_t n_tokens = ne1 / 5; // number of decoded positions in this capture

    if (n_embd != d->n_embd) {
        LOG_ERR("%s: capture n_embd=%lld != driver n_embd=%lld\n",
                __func__, (long long)n_embd, (long long)d->n_embd);
        d->stats.t_ingest_capture_ms += elapsed_ms(t0);
        return 0;
    }

    // Clamp n_dfs to what the capture actually contains. The server may call
    // ingest_prompt_capture(slot.prompt_size) when only the new (uncached) tail
    // of the prompt actually went through llama_decode — the capture only holds
    // the most recent decode's columns. Out-of-range reads here would be UB.
    int32_t n_to_ingest = n_dfs;
    if (dfs_indices == nullptr && n_to_ingest > (int32_t)n_tokens) {
        d->stats.n_capture_clamps++;
        LOG_WRN("%s: requested n_dfs=%d but capture only has n_tokens=%lld; clamping (ring will be incomplete)\n",
                __func__, n_dfs, (long long)n_tokens);
        n_to_ingest = (int32_t)n_tokens;
    }

    for (int32_t i = 0; i < n_to_ingest; ++i) {
        // Source column index in the capture buffer (within each layer's block).
        const int64_t src_col = (dfs_indices != nullptr) ? (int64_t)dfs_indices[i] : (int64_t)i;
        if (src_col < 0 || src_col >= n_tokens) {
            LOG_ERR("%s: src_col=%lld out of capture range [0, %lld)\n",
                    __func__, (long long)src_col, (long long)n_tokens);
            break;
        }

        const int64_t logical_col = d->target_feat_n_committed + (int64_t)i;
        const int64_t dst_col = logical_col % d->target_feat_cap;

        for (int64_t l = 0; l < 5; ++l) {
            // Source: layer l's block starts at column l*n_tokens; pick column src_col within it.
            const float * src = capture + (l * n_tokens + src_col) * n_embd;
            // Destination: ring column dst_col, row l*n_embd.
            float * dst = d->target_feat_ring.data() + dst_col * d->target_feat_n_embd_fc + l * n_embd;
            memcpy(dst, src, (size_t)n_embd * sizeof(float));
        }
    }

    d->target_feat_n_committed += (int64_t)n_to_ingest;
    const double ingest_ms = elapsed_ms(t0);
    switch (source) {
        case ingest_source::prompt:
            d->stats.n_prompt_ingest_calls++;
            d->stats.n_prompt_ingested_tokens += n_to_ingest;
            d->stats.t_prompt_ingest_ms += ingest_ms;
            break;
        case ingest_source::tree:
            d->stats.n_tree_ingested_tokens += n_to_ingest;
            d->stats.t_tree_ingest_ms += ingest_ms;
            break;
        case ingest_source::replay:
            d->stats.n_replay_ingested_tokens += n_to_ingest;
            d->stats.t_replay_ingest_ms += ingest_ms;
            break;
    }
    d->stats.t_ingest_capture_ms += ingest_ms;
    return n_to_ingest;
}

static bool replay_committed_chain(llama_speculative_tree_driver * d,
                                   const llama_ddtree           & tree,
                                   const int32_t                * accepted_dfs,
                                   int32_t                        commit_n,
                                   llama_pos                      committed_pos) {
    if (commit_n <= 0) {
        return true;
    }

    llama_memory_t mem = llama_get_memory(d->target_ctx);
    if (!llama_memory_seq_rm(mem, /*seq_id=*/0, committed_pos, /*p1=*/-1)) {
        LOG_ERR("%s: failed to remove tree KV/recurrent range at pos >= %d\n",
                __func__, (int)committed_pos);
        return false;
    }

    llama_batch replay = llama_batch_init(commit_n, /*embd=*/0, /*n_seq_max=*/1);
    replay.n_tokens = commit_n;
    for (int32_t i = 0; i < commit_n; ++i) {
        replay.token[i]     = tree.nodes[accepted_dfs[i]].token_id;
        replay.pos[i]       = committed_pos + i;
        replay.n_seq_id[i]  = 1;
        replay.seq_id[i][0] = 0;
        replay.logits[i]    = 0;
    }

    const int ret = llama_decode(d->target_ctx, replay);
    llama_batch_free(replay);
    if (ret != 0) {
        LOG_ERR("%s: chain replay llama_decode failed: %d\n", __func__, ret);
        return false;
    }

    driver_ingest_capture(d, nullptr, commit_n, ingest_source::replay);
    return true;
}

static int32_t find_child_token(const llama_ddtree & tree, int32_t parent, llama_token token) {
    for (int32_t i = 1; i < (int32_t) tree.nodes.size(); ++i) {
        if (tree.nodes[i].parent_idx == parent && tree.nodes[i].token_id == token) {
            return i;
        }
    }
    return -1;
}

static llama_token pick_current_logits(llama_speculative_tree_driver * d,
                                       const llama_speculative_tree_verify_cbs * verify_cbs) {
    if (verify_cbs != nullptr && verify_cbs->sample_cb != nullptr) {
        return (llama_token) verify_cbs->sample_cb(verify_cbs->user_data, /*logits_row_idx=*/0);
    }

    const float * row = llama_get_logits_ith(d->target_ctx, 0);
    if (!row) {
        return LLAMA_TOKEN_NULL;
    }
    int32_t best = 0;
    float best_val = row[0];
    for (int64_t v = 1; v < d->n_vocab; ++v) {
        if (row[v] > best_val) {
            best_val = row[v];
            best = (int32_t) v;
        }
    }
    return (llama_token) best;
}

static bool validate_tree_with_chain(llama_speculative_tree_driver * d,
                                     const llama_ddtree           & tree,
                                     llama_pos                      committed_pos,
                                     const llama_speculative_tree_verify_cbs * verify_cbs,
                                     std::vector<int32_t>        & accepted_dfs,
                                     llama_token                  & next_token) {
    accepted_dfs.clear();
    accepted_dfs.push_back(0);
    next_token = LLAMA_TOKEN_NULL;

    llama_memory_t mem = llama_get_memory(d->target_ctx);
    if (!llama_memory_seq_rm(mem, /*seq_id=*/0, committed_pos, /*p1=*/-1)) {
        LOG_ERR("%s: failed to remove tree KV/recurrent range at pos >= %d\n",
                __func__, (int)committed_pos);
        return false;
    }

    int32_t current = 0;
    for (int32_t depth = 0; depth < (int32_t) tree.nodes.size(); ++depth) {
        llama_batch b = llama_batch_init(1, /*embd=*/0, /*n_seq_max=*/1);
        b.n_tokens     = 1;
        b.token[0]     = tree.nodes[current].token_id;
        b.pos[0]       = committed_pos + depth;
        b.n_seq_id[0]  = 1;
        b.seq_id[0][0] = 0;
        b.logits[0]    = 1;

        const auto t_decode0 = ddtree_clock::now();
        const int ret = llama_decode(d->target_ctx, b);
        d->stats.t_exact_decode_ms += elapsed_ms(t_decode0);
        d->stats.n_exact_validate_nodes++;
        llama_batch_free(b);
        if (ret != 0) {
            LOG_ERR("%s: chain validation llama_decode failed at depth %d: %d\n",
                    __func__, (int) depth, ret);
            return false;
        }

        driver_ingest_capture(d, nullptr, 1, ingest_source::replay);

        const auto t_sample0 = ddtree_clock::now();
        const llama_token picked = pick_current_logits(d, verify_cbs);
        d->stats.t_exact_sample_ms += elapsed_ms(t_sample0);
        if (picked == LLAMA_TOKEN_NULL) {
            LOG_ERR("%s: failed to pick from chain validation logits\n", __func__);
            return false;
        }

        const int32_t child = find_child_token(tree, current, picked);
        if (child < 0) {
            next_token = picked;
            return true;
        }

        if (verify_cbs != nullptr && verify_cbs->advance_cb != nullptr) {
            const auto t_advance0 = ddtree_clock::now();
            verify_cbs->advance_cb(verify_cbs->user_data, picked);
            d->stats.t_exact_advance_ms += elapsed_ms(t_advance0);
        }

        accepted_dfs.push_back(child);
        current = child;
    }

    const auto t_sample0 = ddtree_clock::now();
    const llama_token picked = pick_current_logits(d, verify_cbs);
    d->stats.t_exact_sample_ms += elapsed_ms(t_sample0);
    if (picked == LLAMA_TOKEN_NULL) {
        LOG_ERR("%s: failed to pick final chain validation token\n", __func__);
        return false;
    }
    next_token = picked;
    return true;
}

static llama_token diagnose_chain_root_argmax(llama_speculative_tree_driver * d,
                                              llama_token root_token,
                                              llama_pos   committed_pos) {
    llama_mem_snapshot_id snap = llama_seq_snapshot(d->target_ctx, /*seq_id=*/0);
    if (snap == LLAMA_MEM_SNAPSHOT_INVALID) {
        return LLAMA_TOKEN_NULL;
    }

    llama_batch b = llama_batch_init(1, /*embd=*/0, /*n_seq_max=*/1);
    b.n_tokens = 1;
    b.token[0] = root_token;
    b.pos[0] = committed_pos;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;

    llama_token best = LLAMA_TOKEN_NULL;
    if (llama_decode(d->target_ctx, b) == 0) {
        const float * row = llama_get_logits_ith(d->target_ctx, 0);
        if (row) {
            best = 0;
            float best_val = row[0];
            for (int64_t v = 1; v < d->n_vocab; ++v) {
                if (row[v] > best_val) { best_val = row[v]; best = (llama_token)v; }
            }
        }
    }
    llama_batch_free(b);

    llama_memory_t mem = llama_get_memory(d->target_ctx);
    llama_memory_seq_rm(mem, /*seq_id=*/0, committed_pos, /*p1=*/-1);
    llama_seq_restore(d->target_ctx, snap);
    llama_seq_release(d->target_ctx, snap);
    return best;
}

void llama_speculative_tree_driver_ingest_prompt_capture(
        llama_speculative_tree_driver * d,
        int32_t n_prompt_tokens) {
    // Prompt prefill capture is laid out linearly; ingest columns 0..n_prompt_tokens-1.
    driver_ingest_capture(d, nullptr, n_prompt_tokens, ingest_source::prompt);
}

std::vector<llama_token> llama_speculative_tree_driver_step(
        llama_speculative_tree_driver * d,
        llama_token                     root_token,
        llama_pos                       committed_pos,
        const llama_speculative_tree_verify_cbs * verify_cbs) {

    if (!d) {
        return {};
    }
    const auto t_step0 = ddtree_clock::now();

    const llama_model * target_model = llama_get_model(d->target_ctx);
    const int64_t n_embd     = d->n_embd;
    const int64_t n_vocab    = d->n_vocab;
    const int64_t block_size = d->block_size;

    // ── Step 1: build noise embeddings ────────────────────────────────────────
    // [root_token, mask_token * (block_size-1)] → embed rows → [block_size * n_embd]
    d->noise_embd_buf.resize((size_t)block_size * n_embd);
    {
        // Embed root token at position 0.
        if (llama_model_token_embd_lookup(target_model, root_token,
                                          d->noise_embd_buf.data(), n_embd) != 0) {
            LOG_ERR("%s: token_embd_lookup failed for root_token=%d\n",
                            __func__, (int)root_token);
            return {};
        }
        // Embed mask token for positions 1..block_size-1.
        std::vector<float> mask_embd(n_embd);
        if (llama_model_token_embd_lookup(target_model, d->mask_token_id,
                                          mask_embd.data(), n_embd) != 0) {
            LOG_ERR("%s: token_embd_lookup failed for mask_token=%d\n",
                            __func__, (int)d->mask_token_id);
            return {};
        }
        for (int64_t i = 1; i < block_size; ++i) {
            memcpy(d->noise_embd_buf.data() + i * n_embd, mask_embd.data(),
                   (size_t)n_embd * sizeof(float));
        }
    }

    // ── Step 2: slice and pack target_feat from the cumulative ring buffer ────
    // The ring holds columns 0..target_feat_n_committed-1 in order.
    // We use the most recent ctx_len columns.
    const int64_t n_committed = d->target_feat_n_committed;
    if (n_committed == 0) {
        LOG_ERR("%s: target_feat ring is empty; call llama_speculative_tree_driver_ingest_prompt_capture first\n",
                        __func__);
        return {};
    }
    const int64_t ctx_len = std::min(n_committed, d->target_feat_cap);
    const int64_t ring_start = n_committed - ctx_len; // first ring column to include

    // Copy the selected columns from the ring into a contiguous [5*n_embd, ctx_len] buffer
    // where the output is column-major: out[t * 5*n_embd + l*n_embd .. +n_embd] = layer l at pos t.
    {
        const auto t0 = ddtree_clock::now();
        d->target_feat_buf.resize((size_t)5 * n_embd * ctx_len);
        for (int64_t t = 0; t < ctx_len; ++t) {
            const int64_t ring_col = (ring_start + t) % d->target_feat_cap;
            const float * ring_src = d->target_feat_ring.data() + ring_col * d->target_feat_n_embd_fc;
            float * dst = d->target_feat_buf.data() + t * 5 * n_embd;
            memcpy(dst, ring_src, (size_t)5 * n_embd * sizeof(float));
        }
        llama_set_target_feat_raw(d->draft_ctx, d->target_feat_buf.data(),
                                  5 * n_embd, ctx_len, committed_pos);
        d->stats.t_target_feat_pack_ms += elapsed_ms(t0);
    }

    // ── Step 3: draft forward ─────────────────────────────────────────────────
    // Inject target_feat into draft context and run llama_decode with noise embeddings.
    {
        const auto t0 = ddtree_clock::now();
        llama_batch draft_batch = llama_batch_init((int32_t)block_size, (int32_t)n_embd, 1);
        draft_batch.n_tokens = (int32_t)block_size;
        memcpy(draft_batch.embd, d->noise_embd_buf.data(),
               (size_t)block_size * n_embd * sizeof(float));
        for (int32_t i = 0; i < (int32_t)block_size; ++i) {
            draft_batch.pos[i]      = (llama_pos)(committed_pos + i);
            draft_batch.n_seq_id[i] = 1;
            draft_batch.seq_id[i][0] = 0;
            draft_batch.logits[i]   = 1; // output logits for all positions
        }
        int ret = llama_decode(d->draft_ctx, draft_batch);
        llama_batch_free(draft_batch);
        if (ret != 0) {
            LOG_ERR("%s: draft llama_decode failed: %d\n", __func__, ret);
            return {};
        }
        d->stats.t_draft_decode_ms += elapsed_ms(t0);
    }

    // Read draft logits: [block_size, n_vocab].
    // Skip position 0 (root slot fixed to root_token) and use positions 1..block_size-1.
    const int L = (int)block_size - 1; // draft positions with meaningful predictions
    const int K = (d->params.budget > L) ? 8 : 1;

    d->top_log_probs.resize((size_t)L * K);
    d->top_token_ids.resize((size_t)L * K);

    // ── Step 4: extract top-K log-probs ───────────────────────────────────────
    // Draft logits pointer: llama_get_logits_ith(0) is row 0, etc.
    // We skip row 0 (root slot) and use rows 1..L.
    {
        const auto t0 = ddtree_clock::now();
        const float * draft_logits_row1 = llama_get_logits_ith(d->draft_ctx, 1);
        if (!draft_logits_row1) {
            LOG_ERR("%s: draft logits unavailable\n", __func__);
            return {};
        }
        if (K == 1) {
            // Fast path: argmax per position.
            for (int i = 0; i < L; ++i) {
                const float * row = draft_logits_row1 + (size_t)i * n_vocab;
                int32_t best = 0;
                float best_val = row[0];
                for (int64_t v = 1; v < n_vocab; ++v) {
                    if (row[v] > best_val) { best_val = row[v]; best = (int32_t)v; }
                }
                d->top_log_probs[i] = 0.0f; // log-prob irrelevant for pure-chain budget
                d->top_token_ids[i] = best;
            }
        } else {
            extract_top_k_logprobs(draft_logits_row1, L, (int)n_vocab, K,
                                   d->params.temp, d->top_log_probs.data(),
                                   d->top_token_ids.data());
        }
        if (std::getenv("LLAMA_DDTREE_DUMP_DRAFT_TOP") != nullptr && d->stats.n_steps == 0) {
            LOG_INF("draft_top port: step=%lld committed=%d ctx_len=%lld root=%d K=%d\n",
                    (long long)d->stats.n_steps,
                    (int)committed_pos,
                    (long long)ctx_len,
                    (int)root_token,
                    K);
            LOG_INF("draft_top port: top1:");
            for (int i = 0; i < L; ++i) {
                LOG_INF(" %d", (int)d->top_token_ids[(size_t)i * K]);
            }
            LOG_INF("\n");
            if (K > 1) {
                const int rows = std::min(4, L);
                for (int r = 0; r < rows; ++r) {
                    LOG_INF("draft_top port: row%d:", r + 1);
                    for (int k = 0; k < K; ++k) {
                        LOG_INF(" %d", (int)d->top_token_ids[(size_t)r * K + k]);
                    }
                    LOG_INF("\n");
                }
            }
        }
        d->stats.t_topk_ms += elapsed_ms(t0);
    }

    // ── Step 5: build DDTree ──────────────────────────────────────────────────
    llama_ddtree tree;
    {
        const auto t0 = ddtree_clock::now();
        tree = build_ddtree(
            d->top_log_probs.data(), d->top_token_ids.data(),
            L, K, root_token, d->params);
        d->stats.t_build_tree_ms += elapsed_ms(t0);
    }

    const int N = (int)tree.nodes.size(); // includes root node at index 0
    d->stats.n_steps++;
    d->stats.n_tree_verifies++;
    d->stats.n_tree_nodes_total += N;
    d->stats.max_tree_nodes = std::max(d->stats.max_tree_nodes, N);

    const bool fast_batched  = ddtree_fast_batched_enabled();
    const bool trace_batched = std::getenv("LLAMA_DDTREE_TRACE") != nullptr ||
                               std::getenv("LLAMA_DDTREE_TRACE_CHAIN_ROOT") != nullptr;
    const bool need_batched_tree = fast_batched || trace_batched;

    if (!need_batched_tree) {
        std::vector<int32_t> accepted_dfs;
        llama_token next_token = LLAMA_TOKEN_NULL;
        {
            const auto t0 = ddtree_clock::now();
            if (!validate_tree_with_chain(d, tree, committed_pos, verify_cbs, accepted_dfs, next_token)) {
                return {};
            }
            d->stats.t_exact_validate_ms += elapsed_ms(t0);
        }

        const int commit_n = (int)accepted_dfs.size();
        d->stats.n_committed_tokens += commit_n;
        d->stats.max_committed_tokens_per_step =
            std::max(d->stats.max_committed_tokens_per_step, commit_n);

        std::vector<llama_token> result;
        result.reserve(commit_n + 1);
        for (int i = 0; i < commit_n; ++i) {
            result.push_back(tree.nodes[accepted_dfs[i]].token_id);
        }
        result.push_back(next_token);

        d->stats.t_step_ms += elapsed_ms(t_step0);
        return result;
    }

    const bool fast_rollback = fast_batched && ddtree_fast_rollback_enabled() && !d->fast_rollback_unavailable;
    const bool keep_snapshot = !fast_batched || !fast_rollback || ddtree_snapshot_fallback_enabled();

    // ── Step 6: snapshot before target verify ────────────────────────────────
    // By default fast-rollback mode still keeps a snapshot as a safety net.
    // Set LLAMA_DDTREE_SNAPSHOT_FALLBACK=0 to remove this per-step host bounce
    // after validating that persist allocation succeeds in the target runtime.
    llama_mem_snapshot_id snap = LLAMA_MEM_SNAPSHOT_INVALID;
    auto release_snap = [&]() {
        if (snap != LLAMA_MEM_SNAPSHOT_INVALID) {
            llama_seq_release(d->target_ctx, snap);
            snap = LLAMA_MEM_SNAPSHOT_INVALID;
        }
    };
    if (N > 1) {
        const auto t0 = ddtree_clock::now();
        llama_memory_t mem = llama_get_memory(d->target_ctx);
        if (!llama_memory_seq_rm(mem, /*seq_id=*/0, committed_pos, /*p1=*/-1)) {
            LOG_ERR("%s: failed to clear target future range before tree verify at pos %d\n",
                    __func__, (int)committed_pos);
            return {};
        }

        if (keep_snapshot) {
            snap = llama_seq_snapshot(d->target_ctx, /*seq_id=*/0);
            if (snap == LLAMA_MEM_SNAPSHOT_INVALID) {
                LOG_ERR("%s: llama_seq_snapshot failed before tree verify\n", __func__);
                return {};
            }
        }
        d->stats.t_snapshot_ms += elapsed_ms(t0);
    }

    llama_token diag_chain_root = LLAMA_TOKEN_NULL;
    if (std::getenv("LLAMA_DDTREE_TRACE_CHAIN_ROOT") != nullptr) {
        diag_chain_root = diagnose_chain_root_argmax(d, root_token, committed_pos);
    }

    // ── Step 7: target verify (tree-mode forward) ─────────────────────────────
    // Build a tree batch of N tokens and run target decode.
    {
        const auto t0 = ddtree_clock::now();
        llama_batch tree_batch = llama_batch_init_tree(N, 0, 1);
        tree_batch.n_tokens = N;
        for (int i = 0; i < N; ++i) {
            tree_batch.token[i]      = tree.nodes[i].token_id;
            tree_batch.pos[i]        = committed_pos + tree.nodes[i].depth;
            tree_batch.n_seq_id[i]   = 1;
            tree_batch.seq_id[i][0]  = 0;
            tree_batch.logits[i]     = 1; // output logits for all nodes
            tree_batch.parent_id[i]  = tree.nodes[i].parent_idx; // -1 for root
        }
        int ret = llama_decode(d->target_ctx, tree_batch);
        llama_batch_free(tree_batch);
        if (ret != 0) {
            LOG_ERR("%s: target tree llama_decode failed: %d\n", __func__, ret);
            release_snap();
            return {};
        }
        d->stats.t_target_tree_decode_ms += elapsed_ms(t0);
    }

    // ── Step 8: pick verify chain ─────────────────────────────────────────────
    // Keep the batched tree posterior for diagnostics, but do not trust it for
    // final acceptance. Quantized batched tree logits can drift from one-token
    // AR logits enough to flip argmax on close rows.
    d->posterior.resize(N);
    std::vector<float> posterior_margins;
    posterior_margins.resize(N);
    {
        const auto t0 = ddtree_clock::now();
        for (int i = 0; i < N; ++i) {
            const float * row = llama_get_logits_ith(d->target_ctx, i);
            if (!row) {
                LOG_ERR("%s: target logits[%d] unavailable\n", __func__, i);
                release_snap();
                return {};
            }
            int32_t best = 0;
            float best_val = row[0];
            int32_t second = 0;
            float second_val = row[0];
            if (n_vocab > 1) {
                second = 1;
                second_val = row[1];
                if (second_val > best_val) {
                    std::swap(best, second);
                    std::swap(best_val, second_val);
                }
            }
            for (int64_t v = 2; v < n_vocab; ++v) {
                const float val = row[v];
                if (val > best_val) {
                    second = best;
                    second_val = best_val;
                    best = (int32_t)v;
                    best_val = val;
                } else if (val > second_val) {
                    second = (int32_t)v;
                    second_val = val;
                }
            }
            d->posterior[i] = best;
            posterior_margins[i] = best_val - second_val;
        }
        d->stats.t_posterior_scan_ms += elapsed_ms(t0);
    }

    std::vector<int32_t> batched_accepted_dfs;
    llama_token batched_next_token = LLAMA_TOKEN_NULL;
    {
        const auto t0 = ddtree_clock::now();
        follow_verified_tree(tree, d->posterior.data(), batched_accepted_dfs, batched_next_token);
        d->stats.t_accept_path_ms += elapsed_ms(t0);
    }

    const int batched_commit_n = (int)batched_accepted_dfs.size();
    d->stats.n_batched_posterior_committed_tokens += batched_commit_n;
    d->stats.max_batched_posterior_committed_tokens_per_step =
        std::max(d->stats.max_batched_posterior_committed_tokens_per_step, batched_commit_n);

    std::vector<int32_t> accepted_dfs;
    llama_token next_token = LLAMA_TOKEN_NULL;

    if (fast_batched) {
        if (verify_cbs != nullptr && verify_cbs->sample_cb != nullptr) {
            const auto t0 = ddtree_clock::now();
            follow_verified_tree_cb(
                tree,
                verify_cbs->sample_cb,
                verify_cbs->advance_cb,
                verify_cbs->user_data,
                accepted_dfs,
                next_token);
            d->stats.t_accept_path_ms += elapsed_ms(t0);
            d->stats.n_fast_batched_callback_steps++;
        } else {
            accepted_dfs = batched_accepted_dfs;
            next_token   = batched_next_token;
        }

        const int accept_depth = (int)accepted_dfs.size(); // includes root node (index 0)
        const int commit_n     = accept_depth;

        bool did_commit_state = false;
        if (fast_rollback && N > 1) {
            {
                const auto t0 = ddtree_clock::now();
                llama_kv_cache_seq_compact_tree(
                    d->target_ctx,
                    /*seq_id=*/0,
                    accepted_dfs.data(),
                    (int32_t)accepted_dfs.size(),
                    commit_n,
                    (int32_t)committed_pos);
                d->stats.t_kv_compact_ms += elapsed_ms(t0);
            }

            const int32_t rollback_node = commit_n > 0 ? accepted_dfs[commit_n - 1] : 0;
            bool rollback_ok = false;
            {
                const auto t0 = ddtree_clock::now();
                rollback_ok = llama_dflash_rollback_ssm_to_dfs(d->target_ctx, /*seq_id=*/0, rollback_node);
                d->stats.t_ssm_rollback_ms += elapsed_ms(t0);
            }
            if (!rollback_ok) {
                LOG_WRN("%s: fast rollback failed at DFS node %d; falling back to snapshot replay\n",
                        __func__, (int)rollback_node);
                d->fast_rollback_unavailable = true;
            } else {
                const llama_pos recurrent_tail_pos = committed_pos + commit_n - 1;
                bool tail_ok = false;
                {
                    const auto t0 = ddtree_clock::now();
                    tail_ok = llama_dflash_set_recurrent_tail_pos(d->target_ctx, /*seq_id=*/0, recurrent_tail_pos);
                    d->stats.t_ssm_rollback_ms += elapsed_ms(t0);
                }
                if (!tail_ok) {
                    LOG_WRN("%s: failed to set recurrent tail pos to %d after fast rollback; falling back to snapshot replay\n",
                            __func__, (int)recurrent_tail_pos);
                    d->fast_rollback_unavailable = true;
                } else {
                    driver_ingest_capture(d, accepted_dfs.data(), commit_n, ingest_source::tree);
                    d->stats.n_fast_rollback_steps++;
                    did_commit_state = true;
                }
            }
        }

        if (N > 1 && !did_commit_state) {
            if (snap == LLAMA_MEM_SNAPSHOT_INVALID || !llama_seq_restore(d->target_ctx, snap)) {
                LOG_ERR("%s: fast rollback failed and snapshot fallback is unavailable\n", __func__);
                release_snap();
                return {};
            }
            d->stats.n_snapshot_replays++;
            d->stats.n_fast_batched_replays++;
            {
                const auto t0 = ddtree_clock::now();
                if (!replay_committed_chain(d, tree, accepted_dfs.data(), commit_n, committed_pos)) {
                    release_snap();
                    return {};
                }
                d->stats.t_replay_ms += elapsed_ms(t0);
            }
        } else {
            // Root-only verify is already a normal one-token forward. Keep its
            // live target state and ingest its hidden capture for the next draft.
            driver_ingest_capture(d, nullptr, commit_n, ingest_source::replay);
        }

        d->stats.n_committed_tokens += commit_n;
        d->stats.max_committed_tokens_per_step =
            std::max(d->stats.max_committed_tokens_per_step, commit_n);

        release_snap();

        std::vector<llama_token> result;
        result.reserve(commit_n + 1);
        for (int i = 0; i < commit_n; ++i) {
            result.push_back(tree.nodes[accepted_dfs[i]].token_id);
        }
        result.push_back(next_token);

        d->stats.t_step_ms += elapsed_ms(t_step0);
        return result;
    }

    if (snap == LLAMA_MEM_SNAPSHOT_INVALID || !llama_seq_restore(d->target_ctx, snap)) {
        LOG_ERR("%s: llama_seq_restore failed before exact chain validation\n", __func__);
        release_snap();
        return {};
    }
    d->stats.n_snapshot_replays++;
    {
        const auto t0 = ddtree_clock::now();
        if (!validate_tree_with_chain(d, tree, committed_pos, verify_cbs, accepted_dfs, next_token)) {
            release_snap();
            return {};
        }
        d->stats.t_exact_validate_ms += elapsed_ms(t0);
    }

    const int accept_depth = (int)accepted_dfs.size(); // includes root node (index 0)
    const int commit_n     = accept_depth; // root is always committed
    if (batched_accepted_dfs == accepted_dfs && batched_next_token == next_token) {
        d->stats.n_batched_exact_same++;
    } else {
        d->stats.n_batched_exact_diff++;
    }
    if (batched_commit_n > commit_n) {
        d->stats.n_batched_exact_longer++;
    } else if (batched_commit_n < commit_n) {
        d->stats.n_batched_exact_shorter++;
    }
    d->stats.n_committed_tokens += commit_n;
    d->stats.max_committed_tokens_per_step =
        std::max(d->stats.max_committed_tokens_per_step, commit_n);

    if (std::getenv("LLAMA_DDTREE_TRACE") != nullptr) {
        const int32_t rollback_node = (commit_n > 0) ? accepted_dfs[commit_n - 1] : 0;
        const int32_t posterior0 = d->posterior.empty() ? -1 : d->posterior[0];
        float exact_min_margin = 0.0f;
        for (int i = 0; i < (int)accepted_dfs.size(); ++i) {
            const int32_t idx = accepted_dfs[i];
            const float margin = (idx >= 0 && idx < (int)posterior_margins.size()) ? posterior_margins[idx] : 0.0f;
            exact_min_margin = (i == 0) ? margin : std::min(exact_min_margin, margin);
        }
        float batched_min_margin = 0.0f;
        for (int i = 0; i < (int)batched_accepted_dfs.size(); ++i) {
            const int32_t idx = batched_accepted_dfs[i];
            const float margin = (idx >= 0 && idx < (int)posterior_margins.size()) ? posterior_margins[idx] : 0.0f;
            batched_min_margin = (i == 0) ? margin : std::min(batched_min_margin, margin);
        }
        LOG_INF("ddtree_trace: step=%lld pos=%d root=%d N=%d budget=%d posterior0=%d next=%d commit_n=%d batched_commit_n=%d batched_next=%d exact_min_margin=%.6g batched_min_margin=%.6g rollback_node=%d\n",
                (long long)d->stats.n_steps,
                (int)committed_pos,
                (int)root_token,
                N,
                d->params.budget,
                (int)posterior0,
                (int)next_token,
                commit_n,
                batched_commit_n,
                (int)batched_next_token,
                (double)exact_min_margin,
                (double)batched_min_margin,
                (int)rollback_node);
        if (diag_chain_root != LLAMA_TOKEN_NULL) {
            LOG_INF("ddtree_trace: chain_pre_argmax=%d tree_root_argmax=%d\n",
                    (int)diag_chain_root, (int)posterior0);
        }
        LOG_INF("ddtree_trace: accepted=");
        for (int i = 0; i < (int)accepted_dfs.size(); ++i) {
            LOG_INF("%s%d", i == 0 ? "" : ",", (int)accepted_dfs[i]);
        }
        LOG_INF("\n");
        for (int i = 0; i < N; ++i) {
            const int32_t post = (i < (int)d->posterior.size()) ? d->posterior[i] : -1;
            LOG_INF("ddtree_trace: node=%d parent=%d depth=%d tok=%d posterior=%d\n",
                    i,
                    (int)tree.nodes[i].parent_idx,
                    (int)tree.nodes[i].depth,
                    (int)tree.nodes[i].token_id,
                    (int)post);
        }
    }

    // Step 9 is handled by validate_tree_with_chain(): after restoring the
    // snapshot, it decodes the exact accepted path one token at a time and
    // ingests the corresponding hidden captures into the draft feature ring.
    release_snap();

    // ── Step 10: assemble output ──────────────────────────────────────────────
    // accepted[0] = root_token (always, the input token echoed back).
    // accepted[1..accept_depth-1] = newly accepted draft tokens.
    // accepted[accept_depth] = bonus token (next_token).
    std::vector<llama_token> result;
    result.reserve(commit_n + 1);
    for (int i = 0; i < commit_n; ++i) {
        result.push_back(tree.nodes[accepted_dfs[i]].token_id);
    }
    result.push_back(next_token);

    d->stats.t_step_ms += elapsed_ms(t_step0);
    return result;
}
