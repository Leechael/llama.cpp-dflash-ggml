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
// Returns an empty vector on internal failure.
std::vector<llama_token> llama_speculative_tree_driver_step(
        llama_speculative_tree_driver * d,
        llama_token                     root_token,
        llama_pos                       committed_pos);

// Ingest the most recent target_ctx capture as the initial ring contents.
// Call this AFTER the chain-mode prompt prefill that primed target capture,
// BEFORE the first spec step.
// n_prompt_tokens: number of tokens in the prompt that were decoded in the prefill batch.
void llama_speculative_tree_driver_ingest_prompt_capture(
        llama_speculative_tree_driver * d,
        int32_t                         n_prompt_tokens);
