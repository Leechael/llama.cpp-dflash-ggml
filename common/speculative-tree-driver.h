#pragma once

// speculative-tree-driver.h — Phase 4 DDTree spec-decode coordinator.
//
// Binds a target context (Qwen3.5-27B with capture_hidden) and a draft context
// (LLM_ARCH_DFLASH_DRAFT) and implements one spec-decode step per call.
//
// Lifecycle:
//   driver = llama_speculative_tree_driver_init(target_ctx, draft_ctx, params)
//   while (not done):
//     accepted = llama_speculative_tree_driver_step(driver, root_token, committed_pos)
//     ... append accepted to output ...
//     root_token = accepted.back(); committed_pos += accepted.size() - 1
//   llama_speculative_tree_driver_free(driver)

#include "llama.h"
#include "speculative-tree.h"

#include <cstdint>
#include <vector>

struct llama_speculative_tree_driver;

struct llama_speculative_tree_driver_stats {
    int64_t n_steps             = 0;
    int64_t n_tree_verifies     = 0;
    int64_t n_tree_nodes_total  = 0;
    int32_t max_tree_nodes      = 0;
    int64_t n_dfs_last_commits  = 0;
    int64_t n_snapshot_replays  = 0;
    int64_t n_committed_tokens  = 0;
    int32_t max_committed_tokens_per_step = 0;
    int64_t n_batched_posterior_committed_tokens = 0;
    int32_t max_batched_posterior_committed_tokens_per_step = 0;
    int64_t n_batched_exact_same = 0;
    int64_t n_batched_exact_diff = 0;
    int64_t n_batched_exact_longer = 0;
    int64_t n_batched_exact_shorter = 0;
    int64_t n_fast_batched_replays = 0;
    int64_t n_fast_batched_callback_steps = 0;
    int64_t n_fast_rollback_steps = 0;
    int64_t n_prompt_ingest_calls         = 0;
    int64_t n_prompt_ingested_tokens      = 0;
    int64_t n_tree_ingested_tokens        = 0;
    int64_t n_replay_ingested_tokens      = 0;
    int64_t n_capture_clamps              = 0;
    int64_t n_exact_validate_nodes        = 0;

    double t_step_ms                 = 0.0;
    double t_target_feat_pack_ms     = 0.0;
    double t_draft_decode_ms         = 0.0;
    double t_topk_ms                 = 0.0;
    double t_build_tree_ms           = 0.0;
    double t_snapshot_ms             = 0.0;
    double t_target_tree_decode_ms   = 0.0;
    double t_posterior_scan_ms       = 0.0;
    double t_accept_path_ms          = 0.0;
    double t_kv_compact_ms           = 0.0;
    double t_ssm_rollback_ms         = 0.0;
    double t_ingest_capture_ms       = 0.0;
    double t_prompt_ingest_ms        = 0.0;
    double t_tree_ingest_ms          = 0.0;
    double t_replay_ingest_ms        = 0.0;
    double t_replay_ms               = 0.0;
    double t_exact_validate_ms       = 0.0;
    double t_exact_decode_ms         = 0.0;
    double t_exact_sample_ms         = 0.0;
    double t_exact_advance_ms        = 0.0;
};

// Allocate a driver.  target_ctx must have capture_hidden enabled before any
// llama_decode() calls that prime the context.  draft_ctx must use the
// LLM_ARCH_DFLASH_DRAFT architecture.
llama_speculative_tree_driver * llama_speculative_tree_driver_init(
        llama_context            * target_ctx,
        llama_context            * draft_ctx,
        const llama_ddtree_params & params);

void llama_speculative_tree_driver_free(llama_speculative_tree_driver * d);

// Run one spec-decode step.
//
// root_token   — the last committed token (bonus token from the previous step,
//                or the last prompt token on the very first call).
// committed_pos — number of KV positions committed in the target context so far
//                 (i.e. seq_pos_max + 1 for the next token to be placed).
//
// Returns accepted tokens in chronological order (length >= 1):
//   accepted[0]   = root_token (the input, echoed for convenience)
//   accepted[1..] = newly accepted draft tokens
//   accepted.back() = bonus token from the target (the next root for the next step)
//
// The KV cache of target_ctx is compacted to hold only the accepted path after
// each step.  The SSM/conv state is snapshot-before and restore-on-mismatch.
//
// Optional verify callbacks. If non-null, the driver picks the next token at
// each verify-chain step via sample_cb instead of internal argmax. advance_cb
// is invoked whenever the chain accepts a child, so callers can advance their
// sampler/grammar state to mirror the chain.
struct llama_speculative_tree_verify_cbs {
    llama_speculative_pick_cb    sample_cb  = nullptr;
    llama_speculative_advance_cb advance_cb = nullptr;
    void *                       user_data  = nullptr;
};

// Returns an empty vector on internal failure.
std::vector<llama_token> llama_speculative_tree_driver_step(
        llama_speculative_tree_driver * d,
        llama_token                     root_token,
        llama_pos                       committed_pos,
        const llama_speculative_tree_verify_cbs * verify_cbs = nullptr);

// Ingest the most recent target_ctx capture as the initial ring contents.
// Call this AFTER the chain-mode prompt prefill that primed target capture,
// BEFORE the first spec step.
// n_prompt_tokens: number of tokens in the prompt that were decoded in the prefill batch.
void llama_speculative_tree_driver_ingest_prompt_capture(
        llama_speculative_tree_driver * d,
        int32_t                         n_prompt_tokens);

int32_t llama_speculative_tree_driver_context_window();

llama_speculative_tree_driver_stats llama_speculative_tree_driver_get_stats(
        const llama_speculative_tree_driver * d);
