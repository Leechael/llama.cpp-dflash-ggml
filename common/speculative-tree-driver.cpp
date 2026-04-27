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
//   After tree verify, call llama_dflash_rollback_ssm_to_dfs() which copies the SSM state
//   captured at the deepest accepted DFS node from the per-layer persist buffers (written by
//   ggml_gated_delta_net_tree_persist during the tree forward) back into the live cache.
//   This is O(1) per layer and requires no chain-replay decode.
//
// KNOWN LIMITATION (Task 4, option b): conv state is NOT rolled back. The conv state reflects
//   the last DFS token processed rather than the deepest accepted node. The divergence decays
//   within K_conv (~4) tokens; the chain-vs-spec test may see a few divergent tokens at each
//   tree boundary before reconverging. This is a known Phase 2.4 limitation; see roadmap §6.1.

#include "speculative-tree-driver.h"
#include "speculative-tree.h"
#include "log.h"

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

// Maximum target-context window that the draft attends over.
// Matches test_dflash.cpp:1086 DRAFT_CTX_MAX.
static constexpr int DRAFT_CTX_MAX = 2048;

// EOS token for Qwen3.5 family.
static constexpr llama_token QWEN35_EOS = 248045;

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

    // Cumulative target_feat ring buffer, [5*n_embd, target_feat_cap]
    // Stored column-major: column t = position t, rows = [l*n_embd .. (l+1)*n_embd) for layer l.
    // i.e. ring[col * target_feat_n_embd_fc + l*n_embd .. +n_embd] = layer l at committed pos col.
    std::vector<float> target_feat_ring; // size = target_feat_n_embd_fc * target_feat_cap
    int64_t target_feat_n_committed = 0; // total committed positions appended to the ring
    int64_t target_feat_n_embd_fc   = 0; // = 5 * n_embd
    int64_t target_feat_cap         = 8192; // max ring depth (linear buffer, no rotation for now)

    // Scratch buffers
    std::vector<float>   top_log_probs; // [block_size-1, K]
    std::vector<int32_t> top_token_ids; // [block_size-1, K]
    std::vector<float>   noise_embd_buf; // [block_size * n_embd]
    std::vector<int32_t> posterior;      // [N] argmax per tree node
};

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
    d->target_feat_ring.assign((size_t)d->target_feat_n_embd_fc * d->target_feat_cap, 0.0f);

    return d;
}

void llama_speculative_tree_driver_free(llama_speculative_tree_driver * d) {
    delete d;
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
static void driver_ingest_capture(llama_speculative_tree_driver * d,
                                  const int32_t * dfs_indices,
                                  int32_t         n_dfs) {
    int64_t ne0 = 0, ne1 = 0;
    const float * capture = llama_get_hidden_capture_data(d->target_ctx, &ne0, &ne1);
    if (!capture || ne0 == 0 || ne1 == 0) {
        LOG_ERR("%s: no hidden capture data available\n", __func__);
        return;
    }
    // capture layout: [n_embd, 5*n_tokens] → ne0=n_embd, ne1=5*n_tokens
    const int64_t n_embd  = ne0;
    const int64_t n_tokens = ne1 / 5; // number of decoded positions in this capture

    if (n_embd != d->n_embd) {
        LOG_ERR("%s: capture n_embd=%lld != driver n_embd=%lld\n",
                __func__, (long long)n_embd, (long long)d->n_embd);
        return;
    }

    // Clamp n_dfs to what the capture actually contains. The server may call
    // ingest_prompt_capture(slot.prompt_size) when only the new (uncached) tail
    // of the prompt actually went through llama_decode — the capture only holds
    // the most recent decode's columns. Out-of-range reads here would be UB.
    int32_t n_to_ingest = n_dfs;
    if (dfs_indices == nullptr && n_to_ingest > (int32_t)n_tokens) {
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

        // Destination ring column (linear, no rotation in first cut).
        const int64_t dst_col = d->target_feat_n_committed + (int64_t)i;
        if (dst_col >= d->target_feat_cap) {
            LOG_ERR("%s: target_feat ring full (cap=%lld); aborting ingest\n",
                    __func__, (long long)d->target_feat_cap);
            break;
        }

        for (int64_t l = 0; l < 5; ++l) {
            // Source: layer l's block starts at column l*n_tokens; pick column src_col within it.
            const float * src = capture + (l * n_tokens + src_col) * n_embd;
            // Destination: ring column dst_col, row l*n_embd.
            float * dst = d->target_feat_ring.data() + dst_col * d->target_feat_n_embd_fc + l * n_embd;
            memcpy(dst, src, (size_t)n_embd * sizeof(float));
        }
    }

    d->target_feat_n_committed += (int64_t)n_to_ingest;
    if (d->target_feat_n_committed > d->target_feat_cap) {
        d->target_feat_n_committed = d->target_feat_cap;
    }
}

void llama_speculative_tree_driver_ingest_prompt_capture(
        llama_speculative_tree_driver * d,
        int32_t n_prompt_tokens) {
    // Prompt prefill capture is laid out linearly; ingest columns 0..n_prompt_tokens-1.
    driver_ingest_capture(d, nullptr, n_prompt_tokens);
}

std::vector<llama_token> llama_speculative_tree_driver_step(
        llama_speculative_tree_driver * d,
        llama_token                     root_token,
        llama_pos                       committed_pos) {

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
    const int64_t ctx_len = std::min(n_committed, (int64_t)DRAFT_CTX_MAX);
    const int64_t ring_start = n_committed - ctx_len; // first ring column to include

    static int dbg_step = 0;
    if (dbg_step < 3) {
        LOG_INF("ddtree-step[%d]: n_committed=%lld committed_pos=%d ctx_len=%lld ring_start=%lld\n",
                dbg_step, (long long)n_committed, (int)committed_pos,
                (long long)ctx_len, (long long)ring_start);
        ++dbg_step;
    }

    // Copy the selected columns from the ring into a contiguous [5*n_embd, ctx_len] buffer
    // where the output is column-major: out[t * 5*n_embd + l*n_embd .. +n_embd] = layer l at pos t.
    d->target_feat_buf.resize((size_t)5 * n_embd * ctx_len);
    for (int64_t t = 0; t < ctx_len; ++t) {
        const int64_t ring_col = ring_start + t;
        const float * ring_src = d->target_feat_ring.data() + ring_col * d->target_feat_n_embd_fc;
        float * dst = d->target_feat_buf.data() + t * 5 * n_embd;
        memcpy(dst, ring_src, (size_t)5 * n_embd * sizeof(float));
    }

    // ── Step 3: draft forward ─────────────────────────────────────────────────
    // Inject target_feat into draft context and run llama_decode with noise embeddings.
    llama_set_target_feat_raw(d->draft_ctx, d->target_feat_buf.data(),
                              5 * n_embd, ctx_len, committed_pos);

    {
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
    }

    // ── Step 5: build DDTree ──────────────────────────────────────────────────
    llama_ddtree tree = build_ddtree(
        d->top_log_probs.data(), d->top_token_ids.data(),
        L, K, root_token, d->params);

    const int N = (int)tree.nodes.size(); // includes root node at index 0

    // ── Step 6: (Phase 2.4) no snapshot needed before target verify ──────────
    // The tree forward writes per-layer SSM intermediate states to persist buffers
    // (ggml_gated_delta_net_tree_persist); rollback after accept reads directly from
    // those buffers instead of restoring a pre-verify snapshot and replaying.

    // ── Step 7: target verify (tree-mode forward) ─────────────────────────────
    // Build a tree batch of N tokens and run target decode.
    {
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
            return {};
        }
    }

    // ── Step 8: compute posterior and follow tree ─────────────────────────────
    d->posterior.resize(N);
    for (int i = 0; i < N; ++i) {
        const float * row = llama_get_logits_ith(d->target_ctx, i);
        if (!row) {
            LOG_ERR("%s: target logits[%d] unavailable\n", __func__, i);
            return {};
        }
        int32_t best = 0;
        float best_val = row[0];
        for (int64_t v = 1; v < n_vocab; ++v) {
            if (row[v] > best_val) { best_val = row[v]; best = (int32_t)v; }
        }
        d->posterior[i] = best;
    }

    std::vector<int32_t> accepted_dfs;
    llama_token next_token = LLAMA_TOKEN_NULL;
    follow_verified_tree(tree, d->posterior.data(), accepted_dfs, next_token);

    const int accept_depth = (int)accepted_dfs.size(); // includes root node (index 0)
    const int commit_n     = accept_depth; // root is always committed

    // ── Step 8b: ingest accepted hidden states into the cumulative ring ───────
    // The tree forward populated the capture buffer with hidden states for all N tree nodes.
    // We keep only the accepted_dfs columns and append them to the ring.
    driver_ingest_capture(d, accepted_dfs.data(), (int32_t)commit_n);

    // ── Step 9: compact KV cache and rollback SSM state via persist buffers ───
    // KV compaction: tree was placed at slots [committed_pos, committed_pos+N), so the
    // spine starts at committed_pos. This preserves the prompt's KV slots [0, committed_pos).
    // No-op on pure SSM models.
    llama_kv_cache_seq_compact_tree(d->target_ctx, 0,
                                    accepted_dfs.data(), (int32_t)accept_depth,
                                    (int32_t)commit_n,
                                    (int32_t)committed_pos);

    // SSM rollback (Phase 2.4): copy the SSM state from the persist buffer column at the
    // deepest accepted DFS node back into the live recurrent cache for seq_id=0.
    // This is always needed: the tree forward leaves SSM at tree[N-1] (last DFS node),
    // but we need it at accepted_dfs[commit_n-1] (deepest accepted node).
    // accept_depth == N is not a special case — we still rollback to the correct node.
    {
        const int32_t rollback_node = (commit_n > 0) ? accepted_dfs[commit_n - 1] : 0;
        bool ok = llama_dflash_rollback_ssm_to_dfs(d->target_ctx, /*seq_id=*/0, rollback_node);
        if (!ok) {
            LOG_WRN("%s: llama_dflash_rollback_ssm_to_dfs failed (persist buffers may be unallocated)\n",
                    __func__);
        }
    }
    // NOTE: conv state is NOT rolled back here (Phase 2.4 Task 4, option b).
    // The conv state holds the window for the last DFS-processed token instead of the
    // deepest accepted node. Divergence is bounded by K_conv (~4 tokens) and decays
    // naturally. A future phase can add ggml_ssm_conv_tree_persist to fix this exactly.

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

    return result;
}
