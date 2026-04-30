#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

struct llama_ddtree_params {
    int   budget     = 22;    // total tree node count cap (including root)
    float temp       = 1.0f;  // temperature for log-prob computation
    bool  chain_seed = true;  // seed heap with greedy chain (recommended)
    int   block_size = 16;    // matches dflash draft block_size
    int   top_k      = 0;     // 0 = auto (8 if budget > L else 1)
};

struct llama_ddtree_node {
    llama_token token_id;
    int32_t     parent_idx;  // -1 for root (index 0)
    int32_t     depth;       // root = 0, root's children = 1, etc.
};

struct llama_ddtree {
    // nodes[0] is always the root (last committed token).
    // nodes[1..N-1] are DFS-ordered tree branches.
    std::vector<llama_ddtree_node> nodes;

    // visibility[i*N + j] = 1 iff node j is an ancestor of node i (inclusive).
    // Row i of this matrix is the attention mask row for tree position i.
    std::vector<uint8_t> visibility;
};

// Build a DDTree from per-position top-K log-probabilities.
//
// top_log_probs  [L, K]  draft top-K log-probabilities, descending per row
// top_token_ids  [L, K]  matching token ids
// L              number of draft positions (depth extent of the tree)
// K              top-K width per position
// root_token     the root token (last committed token)
// p              build parameters
llama_ddtree build_ddtree(
    const float   * top_log_probs,
    const int32_t * top_token_ids,
    int             L,
    int             K,
    llama_token     root_token,
    const llama_ddtree_params & p);

// Walk the tree greedily following the target's per-node argmax (posterior).
//
// Starting at the root (index 0), at each step the walk looks for a child
// whose token_id matches posterior[current_index]. The walk stops when no
// matching child exists. The root is always in the accepted list.
//
// posterior   [N]  target argmax token at each tree node position
// accepted    output: flat node indices of the accepted path (starts with 0)
// next_token  output: target argmax at the deepest accepted node (bonus token)
void follow_verified_tree(
    const llama_ddtree        & tree,
    const int32_t             * posterior,
    std::vector<int32_t>      & accepted,
    llama_token               & next_token);

// Variant of follow_verified_tree that pulls the picked token at each chain
// step from caller-provided callbacks instead of a precomputed posterior[].
// Lets callers (server) plug in grammar-aware sampling so the chain only
// accepts tokens the sampler+grammar would have produced.
//
//   sample_cb (ud, logits_row_idx) -> picked token at this row (no state advance)
//   advance_cb(ud, accepted_token)  -> caller must advance its sampler/grammar
//
// advance_cb is invoked every time the chain accepts a child (= the picked
// token matched a child of `current`). It is NOT invoked for the bonus token.
typedef int32_t (*llama_speculative_pick_cb)   (void * user_data, int32_t logits_row_idx);
typedef void    (*llama_speculative_advance_cb)(void * user_data, llama_token accepted_token);

void follow_verified_tree_cb(
    const llama_ddtree           & tree,
    llama_speculative_pick_cb      sample_cb,
    llama_speculative_advance_cb   advance_cb,
    void                         * user_data,
    std::vector<int32_t>         & accepted,
    llama_token                  & next_token);

// Compute the [N, N] ancestor visibility mask from nodes[].parent_idx.
// dst must point to an N*N uint8 buffer (caller-allocated).
// Row i: dst[i*N + j] = 1 iff node j is an ancestor of i (inclusive).
void build_tree_visibility(
    const std::vector<llama_ddtree_node> & nodes,
    uint8_t                              * dst);

// Extract per-position top-K log-probabilities from a [L, V] logits matrix.
//
// Uses online logsumexp + a size-K min-heap for a single-pass O(L*V) scan.
// Output rows are sorted descending by log-probability (rank 0 = argmax).
//
// logits         [L, V]  row-major F32
// L              number of rows (draft positions)
// V              vocabulary size
// K              top-K width
// temp           temperature: logits are divided by temp before softmax
// out_log_probs  [L, K]  caller-allocated output
// out_token_ids  [L, K]  caller-allocated output
void extract_top_k_logprobs(
    const float * logits,
    int           L,
    int           V,
    int           K,
    float         temp,
    float       * out_log_probs,
    int32_t     * out_token_ids);
