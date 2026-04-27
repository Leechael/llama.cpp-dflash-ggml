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
// SSM rollback strategy (first-cut, see comment at step 9):
//   snapshot the recurrent state BEFORE target verify, restore on accept_depth < tree_depth.
//   After restoring, replay accepted tokens through chain-mode decode to re-advance SSM state.
//   This is slower than per-layer rollback in the reference but correct.

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

    // ── Step 2: slice and pack target_feat ────────────────────────────────────
    // Read hidden capture: [n_embd, 5*n_total] F32 host buffer.
    int64_t cap_ne0 = 0, cap_ne1 = 0;
    const float * capture = llama_get_hidden_capture_data(d->target_ctx, &cap_ne0, &cap_ne1);
    if (!capture || cap_ne0 == 0 || cap_ne1 == 0) {
        LOG_ERR("%s: no hidden capture data available; call llama_decode on target first\n",
                        __func__);
        return {};
    }
    // cap_ne0 == n_embd, cap_ne1 == 5 * n_total
    const int64_t n_total = cap_ne1 / 5;
    if (cap_ne0 != n_embd || n_total <= 0) {
        LOG_ERR("%s: unexpected capture shape [%lld, %lld], n_embd=%lld\n",
                        __func__, (long long)cap_ne0, (long long)cap_ne1, (long long)n_embd);
        return {};
    }
    const int64_t ctx_len = std::min((int64_t)committed_pos, (int64_t)DRAFT_CTX_MAX);
    if (ctx_len > n_total) {
        LOG_ERR("%s: ctx_len=%lld > n_total=%lld (capture buffer too small)\n",
                        __func__, (long long)ctx_len, (long long)n_total);
        return {};
    }

    d->target_feat_buf.resize((size_t)5 * n_embd * ctx_len);
    pack_target_feat(capture, n_embd, n_total, ctx_len, d->target_feat_buf.data());

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

    // ── Step 6: snapshot SSM state before target verify ───────────────────────
    // For the first-cut Phase 4 implementation we snapshot before the tree forward
    // and restore if the accepted path doesn't reach the full tree depth.
    // This is slower than per-layer rollback from intermediate captures but is
    // correct and avoids depending on SSM intermediate state capture APIs.
    llama_mem_snapshot_id snap_id = llama_seq_snapshot(d->target_ctx, 0);

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
            if (snap_id != LLAMA_MEM_SNAPSHOT_INVALID) {
                llama_seq_release(d->target_ctx, snap_id);
            }
            return {};
        }
    }

    // ── Step 8: compute posterior and follow tree ─────────────────────────────
    d->posterior.resize(N);
    for (int i = 0; i < N; ++i) {
        const float * row = llama_get_logits_ith(d->target_ctx, i);
        if (!row) {
            LOG_ERR("%s: target logits[%d] unavailable\n", __func__, i);
            if (snap_id != LLAMA_MEM_SNAPSHOT_INVALID) {
                llama_seq_release(d->target_ctx, snap_id);
            }
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

    // ── Step 9: compact KV cache and rollback recurrent state ─────────────────
    // KV compaction: copies K/V rows from accepted_dfs[0..commit_n) to spine [0..commit_n).
    // This is a no-op on pure SSM models.
    llama_kv_cache_seq_compact_tree(d->target_ctx, 0,
                                    accepted_dfs.data(), (int32_t)accept_depth,
                                    (int32_t)commit_n);

    // Recurrent state rollback: two cases.
    //   (a) accept_depth == N (full tree accepted): SSM state is already correct.
    //   (b) accept_depth <  N: we snapped before verify, restore and replay.
    //
    // Trade-off vs reference: the reference uses per-layer intermediate captures
    // (ssm_intermediate_states) written during the tree forward to roll back to
    // exactly the accepted DFS node in O(1) per layer. Here we restore to the
    // pre-verify snapshot and replay the accepted tokens through chain-mode decode.
    // This adds ~accept_depth sequential target decodes per step, which is expensive
    // but correct. A future optimisation can add per-layer intermediate capture APIs.
    if (snap_id != LLAMA_MEM_SNAPSHOT_INVALID) {
        if (accept_depth < N) {
            // Restore SSM to pre-verify state then replay accepted path.
            bool ok = llama_seq_restore(d->target_ctx, snap_id);
            if (!ok) {
                LOG_WRN("%s: seq_restore failed; SSM state may be stale\n", __func__);
            }
            // Replay: chain-decode root + accepted children (skip root which is already committed).
            for (int i = 1; i < commit_n; ++i) {
                llama_batch replay_batch = llama_batch_init(1, 0, 1);
                replay_batch.n_tokens    = 1;
                replay_batch.token[0]    = tree.nodes[accepted_dfs[i]].token_id;
                replay_batch.pos[0]      = committed_pos + tree.nodes[accepted_dfs[i]].depth;
                replay_batch.n_seq_id[0] = 1;
                replay_batch.seq_id[0][0] = 0;
                replay_batch.logits[0]   = 0; // don't need logits from replay
                int ret = llama_decode(d->target_ctx, replay_batch);
                llama_batch_free(replay_batch);
                if (ret != 0) {
                    LOG_WRN("%s: SSM replay decode failed at step %d\n", __func__, i);
                    break;
                }
            }
        }
        llama_seq_release(d->target_ctx, snap_id);
    }

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
