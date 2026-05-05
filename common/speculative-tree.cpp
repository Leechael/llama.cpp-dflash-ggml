#include "speculative-tree.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// build_ddtree
// ---------------------------------------------------------------------------

llama_ddtree build_ddtree(
        const float   * top_log_probs,
        const int32_t * top_token_ids,
        int             L,
        int             K,
        llama_token     root_token,
        const llama_ddtree_params & p) {

    llama_ddtree tree;

    // Node 0 is always the root (last committed token).
    tree.nodes.push_back({root_token, /*parent_idx*/ -1, /*depth*/ 0});

    // Match standalone DFlash: --ddtree-budget counts non-root nodes, while
    // flat tree storage also includes slot 0 = root.
    const int budget = (p.budget < 0) ? 1 : p.budget + 1;

    if (budget <= 1 || L <= 0) {
        tree.visibility.assign(1, 1);
        return tree;
    }

    // child_maps[flat_index] maps token_id → child flat index.
    std::vector<std::unordered_map<int32_t, int>> child_maps;
    child_maps.emplace_back();  // root's children (index 0)

    // Heap entry: a candidate node waiting to be inserted.
    struct HeapEntry {
        float   neg_logw;     // stored as negative so max-heap by logw
        int     parent_index; // flat index of already-inserted parent
        int     depth;        // absolute depth (1..L)
        int     rank;         // rank within top_token_ids row (depth-1)
        float   logw;         // cumulative path log-prob from root to candidate
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            return a.neg_logw > b.neg_logw;  // pop smallest neg_logw = highest logw
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    if (p.chain_seed) {
        // Pre-insert the top-1 greedy chain to depth min(L, budget-1).
        // Guarantees the tree always contains at least the greedy chain path.
        const int chain_depth = std::min(L, budget - 1);
        float cum_logw = 0.0f;
        int   prev_idx = 0;

        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = (int)tree.nodes.size();
            tree.nodes.push_back({tok_id, prev_idx, d});
            child_maps.emplace_back();
            child_maps[prev_idx][tok_id] = cur_idx;

            // Queue rank-1 sibling so best-first can branch off the chain.
            if (K > 1) {
                const float sib_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({-sib_logw, prev_idx, d, 1, sib_logw});
            }

            prev_idx = cur_idx;
        }
    } else {
        // Pure best-first: seed with depth-1 top-1 candidate only.
        const float logw0 = top_log_probs[0];
        heap.push({-logw0, 0, 1, 0, logw0});
    }

    // Expand candidates in log-prob order until budget is reached.
    while (!heap.empty() && (int)tree.nodes.size() < budget) {
        const HeapEntry top = heap.top();
        heap.pop();

        const int     dm1    = top.depth - 1;
        const int     rank   = top.rank;
        const int32_t tok_id = top_token_ids[(size_t)dm1 * K + rank];

        // Skip duplicates (chain_seed may have already inserted this token
        // under the same parent).
        if (child_maps[top.parent_index].count(tok_id)) {
            continue;
        }

        const int cur_idx = (int)tree.nodes.size();
        tree.nodes.push_back({tok_id, top.parent_index, top.depth});
        child_maps.emplace_back();
        child_maps[top.parent_index][tok_id] = cur_idx;

        // Next sibling (same depth, rank+1).
        if (rank + 1 < K) {
            const float sib_logw = top.logw
                - top_log_probs[(size_t)dm1 * K + rank]
                + top_log_probs[(size_t)dm1 * K + rank + 1];
            heap.push({-sib_logw, top.parent_index, top.depth, rank + 1, sib_logw});
        }

        // First child (depth+1, top-1 under this node).
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            heap.push({-child_logw, cur_idx, top.depth + 1, 0, child_logw});
        }
    }

    // Build ancestor-only visibility mask for attention masking.
    const int N = (int)tree.nodes.size();
    tree.visibility.assign((size_t)N * N, 0);
    build_tree_visibility(tree.nodes, tree.visibility.data());

    return tree;
}

// ---------------------------------------------------------------------------
// follow_verified_tree
// ---------------------------------------------------------------------------

void follow_verified_tree(
        const llama_ddtree        & tree,
        const int32_t             * posterior,
        std::vector<int32_t>      & accepted,
        llama_token               & next_token) {

    const int N = (int)tree.nodes.size();

    // Build per-node child maps from parent_idx links.
    std::vector<std::unordered_map<int32_t, int>> child_maps(N);
    for (int i = 1; i < N; i++) {
        const int p = tree.nodes[i].parent_idx;
        child_maps[p][tree.nodes[i].token_id] = i;
    }

    accepted.clear();
    accepted.reserve(N);
    accepted.push_back(0);  // root is always accepted

    int current = 0;
    while (true) {
        // posterior[current] is the target model's argmax at this tree position.
        const auto it = child_maps[current].find((llama_token)posterior[current]);
        if (it == child_maps[current].end()) {
            break;
        }
        current = it->second;
        accepted.push_back(current);
    }

    // Bonus token: the target's argmax at the deepest accepted node.
    next_token = (llama_token)posterior[current];
}

// follow_verified_tree_cb: same chain-walk semantics as follow_verified_tree
// but the picked token at each step comes from sample_cb (caller-side
// grammar/sampler), and chain advances notify the caller via advance_cb.
void follow_verified_tree_cb(
        const llama_ddtree           & tree,
        const std::vector<int32_t>   & posterior,
        llama_speculative_pick_cb      sample_cb,
        llama_speculative_advance_cb   advance_cb,
        void                         * user_data,
        std::vector<int32_t>         & accepted,
        llama_token                  & next_token) {
    const int N = (int)tree.nodes.size();

    // Build per-node child maps from parent_idx links.
    std::vector<std::unordered_map<int32_t, int>> child_maps(N);
    for (int i = 1; i < N; i++) {
        const int p = tree.nodes[i].parent_idx;
        child_maps[p][tree.nodes[i].token_id] = i;
    }

    accepted.clear();
    accepted.reserve(N);
    accepted.push_back(0);  // root is always accepted

    int current = 0;
    while (true) {
        const llama_token batched_pick = (current < (int)posterior.size())
                                         ? (llama_token)posterior[current]
                                         : LLAMA_TOKEN_NULL;
        const int32_t picked = sample_cb(user_data, current, batched_pick);
        const auto it = child_maps[current].find(picked);
        if (it == child_maps[current].end()) {
            next_token = (llama_token)picked;
            break;
        }
        if (advance_cb != nullptr) {
            advance_cb(user_data, (llama_token)tree.nodes[it->second].token_id);
        }
        current = it->second;
        accepted.push_back(current);
    }
}

// ---------------------------------------------------------------------------
// build_tree_visibility
// ---------------------------------------------------------------------------

void build_tree_visibility(
        const std::vector<llama_ddtree_node> & nodes,
        uint8_t                              * dst) {

    const int N = (int)nodes.size();

    // Root only sees itself.
    dst[0 * N + 0] = 1;

    for (int i = 1; i < N; i++) {
        const int p = nodes[i].parent_idx;
        // DFS order guarantees p < i, so row p is already complete.
        // Inherit the parent's visibility row, then mark self.
        for (int j = 0; j < i; j++) {
            dst[(size_t)i * N + j] = dst[(size_t)p * N + j];
        }
        dst[(size_t)i * N + i] = 1;
    }
}

// ---------------------------------------------------------------------------
// extract_top_k_logprobs
// ---------------------------------------------------------------------------

void extract_top_k_logprobs(
        const float * logits,
        int           L,
        int           V,
        int           K,
        float         temp,
        float       * out_log_probs,
        int32_t     * out_token_ids) {

    const float inv_t = 1.0f / std::max(1e-6f, temp);

    struct Entry {
        float   logit;  // temperature-scaled
        int32_t id;
    };
    // Min-heap: smallest scaled logit at top, evicted when a larger one arrives.
    auto cmp_min = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };

    for (int i = 0; i < L; i++) {
        const float * row = logits + (size_t)i * V;

        std::vector<Entry> heap;
        heap.reserve(K + 1);

        // Single pass: top-K min-heap.  Approximate row normalization from
        // the retained top-K only to avoid a full-vocab exp/logsumexp pass.
        for (int j = 0; j < V; j++) {
            const float l = row[j] * inv_t;

            // Maintain top-K min-heap.
            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_min);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_min);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_min);
            }
        }

        // sort_heap with a greater-than comparator (cmp_min) produces descending
        // order — same as std::sort with std::greater — so no reversal needed.
        std::sort_heap(heap.begin(), heap.end(), cmp_min);

        const float row_best = heap.empty() ? 0.0f : heap[0].logit;
        float sum_exp_top = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum_exp_top += std::exp(heap[k].logit - row_best);
        }
        const float log_z_approx = row_best + std::log(sum_exp_top);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z_approx;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}
