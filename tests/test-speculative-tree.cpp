// Standalone unit tests for speculative-tree.{h,cpp}.
// No model or GPU required. Uses hand-computed fixtures.
// Exits non-zero on first failure.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "speculative-tree.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// Test 1: build_ddtree small case
//
// L=2, K=2, budget=4 (total nodes including root), chain_seed=true, root=10.
//
// top_log_probs[L*K]:
//   position 0 (depth 1): [-0.1, -1.5]   tokens [20, 21]
//   position 1 (depth 2): [-0.2, -1.6]   tokens [30, 31]
//
// Chain seeding (chain_depth = min(2, budget-1=3) = 2):
//   d=1: insert tok=20 as node 1 (parent=0, depth=1), cum_logw=-0.1
//         push sibling: logw=-1.5, parent=0, depth=1, rank=1, tok=21
//   d=2: insert tok=30 as node 2 (parent=1, depth=2), cum_logw=-0.3
//         push sibling: logw=-0.3-(-0.2)+(-1.6)=-1.7, parent=1, depth=2, rank=1, tok=31
//
// Heap after chain: {logw=-1.5,tok=21} and {logw=-1.7,tok=31}
// Pop best: logw=-1.5 → tok=21 inserted as node 3 (parent=0, depth=1). Done.
//
// Expected tree nodes:
//   [0] root(10),  parent=-1, depth=0
//   [1] tok=20,    parent=0,  depth=1
//   [2] tok=30,    parent=1,  depth=2
//   [3] tok=21,    parent=0,  depth=1
// ---------------------------------------------------------------------------
static void test_build_ddtree_small() {
    const int L = 2, K = 2;
    const float top_log_probs[] = {
        -0.1f, -1.5f,  // position 0
        -0.2f, -1.6f,  // position 1
    };
    const int32_t top_token_ids[] = {
        20, 21,  // position 0
        30, 31,  // position 1
    };

    llama_ddtree_params p;
    p.budget     = 4;    // total node cap including root
    p.chain_seed = true;
    p.temp       = 1.0f;

    const llama_ddtree tree = build_ddtree(
        top_log_probs, top_token_ids, L, K, /*root_token*/ 10, p);

    assert(tree.nodes.size() == 4);

    // Node 0: root
    assert(tree.nodes[0].token_id  == 10);
    assert(tree.nodes[0].parent_idx == -1);
    assert(tree.nodes[0].depth     == 0);

    // Node 1: tok=20, depth-1 chain top-1
    assert(tree.nodes[1].token_id  == 20);
    assert(tree.nodes[1].parent_idx == 0);
    assert(tree.nodes[1].depth     == 1);

    // Node 2: tok=30, depth-2 chain top-1
    assert(tree.nodes[2].token_id  == 30);
    assert(tree.nodes[2].parent_idx == 1);
    assert(tree.nodes[2].depth     == 2);

    // Node 3: tok=21, best heap candidate (sibling of 20 at depth 1)
    assert(tree.nodes[3].token_id  == 21);
    assert(tree.nodes[3].parent_idx == 0);
    assert(tree.nodes[3].depth     == 1);

    // Visibility: 4x4 mask, row i has 1 at all ancestors of i (inclusive).
    // node 0 ancestors: {0}
    // node 1 ancestors: {0, 1}
    // node 2 ancestors: {0, 1, 2}
    // node 3 ancestors: {0, 3}
    const int N = 4;
    assert(tree.visibility.size() == (size_t)(N * N));
    // row 0
    assert(tree.visibility[0*N+0] == 1);
    assert(tree.visibility[0*N+1] == 0);
    assert(tree.visibility[0*N+2] == 0);
    assert(tree.visibility[0*N+3] == 0);
    // row 1
    assert(tree.visibility[1*N+0] == 1);
    assert(tree.visibility[1*N+1] == 1);
    assert(tree.visibility[1*N+2] == 0);
    assert(tree.visibility[1*N+3] == 0);
    // row 2
    assert(tree.visibility[2*N+0] == 1);
    assert(tree.visibility[2*N+1] == 1);
    assert(tree.visibility[2*N+2] == 1);
    assert(tree.visibility[2*N+3] == 0);
    // row 3
    assert(tree.visibility[3*N+0] == 1);
    assert(tree.visibility[3*N+1] == 0);
    assert(tree.visibility[3*N+2] == 0);
    assert(tree.visibility[3*N+3] == 1);

    printf("PASS: test_build_ddtree_small\n");
}

// ---------------------------------------------------------------------------
// Test 2: follow_verified_tree
//
// Uses the same 4-node tree from test 1.
// Tree structure:
//   node 0: root(10), no parent
//   node 1: tok=20, parent=0
//   node 2: tok=30, parent=1
//   node 3: tok=21, parent=0
//
// child_maps derived from parent_idx:
//   node 0 children: {20→1, 21→3}
//   node 1 children: {30→2}
//   node 2 children: {}
//   node 3 children: {}
//
// Semantic: posterior[i] is the target model's argmax prediction at node i.
// The walk starts at node 0; at each step it looks for a child whose
// token_id matches posterior[current]. If found, advance; otherwise stop.
// accepted = [visited indices]; next_token = posterior[deepest accepted].
//
// Case A: posterior = [20, 30, 99, 99]
//   Start at 0. posterior[0]=20 → child 1 exists. Move to 1.
//   posterior[1]=30 → child 2 exists. Move to 2.
//   posterior[2]=99 → no child. Stop.
//   accepted=[0,1,2], next_token=99.
//
// Case B: posterior = [21, 99, 99, 99]
//   Start at 0. posterior[0]=21 → child 3 exists. Move to 3.
//   posterior[3]=99 → no child. Stop.
//   accepted=[0,3], next_token=99.
//
// Case C: posterior = [5, 99, 99, 99]
//   Start at 0. posterior[0]=5 → no child. Stop immediately.
//   accepted=[0], next_token=5.
// ---------------------------------------------------------------------------
static void test_follow_verified_tree() {
    // Build the same 4-node tree via build_ddtree.
    const int L = 2, K = 2;
    const float top_log_probs[] = { -0.1f, -1.5f, -0.2f, -1.6f };
    const int32_t top_token_ids[] = { 20, 21, 30, 31 };

    llama_ddtree_params p;
    p.budget     = 4;
    p.chain_seed = true;
    p.temp       = 1.0f;

    const llama_ddtree tree = build_ddtree(
        top_log_probs, top_token_ids, L, K, 10, p);

    std::vector<int32_t> accepted;
    llama_token next_tok = -1;

    // Case A: greedy chain match
    {
        const int32_t posterior[] = { 20, 30, 99, 99 };
        follow_verified_tree(tree, posterior, accepted, next_tok);
        assert(accepted.size() == 3);
        assert(accepted[0] == 0);
        assert(accepted[1] == 1);
        assert(accepted[2] == 2);
        assert(next_tok == 99);
    }

    // Case B: branch match (tok=21 path)
    {
        const int32_t posterior[] = { 21, 99, 99, 99 };
        follow_verified_tree(tree, posterior, accepted, next_tok);
        assert(accepted.size() == 2);
        assert(accepted[0] == 0);
        assert(accepted[1] == 3);
        assert(next_tok == 99);
    }

    // Case C: no match at root level — only root accepted
    {
        const int32_t posterior[] = { 5, 99, 99, 99 };
        follow_verified_tree(tree, posterior, accepted, next_tok);
        assert(accepted.size() == 1);
        assert(accepted[0] == 0);
        assert(next_tok == 5);
    }

    printf("PASS: test_follow_verified_tree\n");
}

// ---------------------------------------------------------------------------
// Test 3: build_tree_visibility — hand-constructed 5-node tree
//
// Manually define a tree:
//   node 0: root,        parent=-1
//   node 1: child of 0,  parent=0
//   node 2: child of 1,  parent=1
//   node 3: child of 0,  parent=0
//   node 4: child of 3,  parent=3
//
// Expected visibility (5x5):
//   row 0: {0}          → [1,0,0,0,0]
//   row 1: {0,1}        → [1,1,0,0,0]
//   row 2: {0,1,2}      → [1,1,1,0,0]
//   row 3: {0,3}        → [1,0,0,1,0]
//   row 4: {0,3,4}      → [1,0,0,1,1]
// ---------------------------------------------------------------------------
static void test_build_tree_visibility() {
    std::vector<llama_ddtree_node> nodes = {
        { 10, -1, 0 },  // 0: root
        { 20,  0, 1 },  // 1: child of 0
        { 30,  1, 2 },  // 2: child of 1
        { 40,  0, 1 },  // 3: child of 0
        { 50,  3, 2 },  // 4: child of 3
    };

    const int N = (int)nodes.size();
    std::vector<uint8_t> vis(N * N, 0);
    build_tree_visibility(nodes, vis.data());

    // Row 0
    assert(vis[0*N+0] == 1); assert(vis[0*N+1] == 0);
    assert(vis[0*N+2] == 0); assert(vis[0*N+3] == 0); assert(vis[0*N+4] == 0);
    // Row 1
    assert(vis[1*N+0] == 1); assert(vis[1*N+1] == 1);
    assert(vis[1*N+2] == 0); assert(vis[1*N+3] == 0); assert(vis[1*N+4] == 0);
    // Row 2
    assert(vis[2*N+0] == 1); assert(vis[2*N+1] == 1);
    assert(vis[2*N+2] == 1); assert(vis[2*N+3] == 0); assert(vis[2*N+4] == 0);
    // Row 3
    assert(vis[3*N+0] == 1); assert(vis[3*N+1] == 0);
    assert(vis[3*N+2] == 0); assert(vis[3*N+3] == 1); assert(vis[3*N+4] == 0);
    // Row 4
    assert(vis[4*N+0] == 1); assert(vis[4*N+1] == 0);
    assert(vis[4*N+2] == 0); assert(vis[4*N+3] == 1); assert(vis[4*N+4] == 1);

    printf("PASS: test_build_tree_visibility\n");
}

// ---------------------------------------------------------------------------
// Test 4: extract_top_k_logprobs
//
// Feed a [3, 8] logits matrix with known values at temp=1.0.
// K=3. Verify output ordering (descending log-prob) and values to 1e-5.
//
// Row 0: logits = [0,1,2,3,4,5,6,7]  (argmax = id=7)
// Row 1: logits = [7,6,5,4,3,2,1,0]  (argmax = id=0)
// Row 2: logits = [0,0,0,0,10,0,0,0] (argmax = id=4, dominant)
//
// For row 0: log_z = logsumexp([0,1,2,3,4,5,6,7])
//   log_z = 7 + log(sum of exp(k-7) for k=0..7) = 7 + log(exp(-7)+...+exp(0))
//   Top 3 by logit: ids [7,6,5], logprobs = [7-log_z, 6-log_z, 5-log_z]
//
// We compute expected values in the test itself using std::log and std::exp.
// ---------------------------------------------------------------------------
static float logsumexp_vec(const float * v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += std::exp(v[i] - mx);
    return mx + std::log(s);
}

static void test_extract_top_k_logprobs() {
    const int L = 3, V = 8, K = 3;

    const float logits[L * V] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,  // row 0
        7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,  // row 1
        0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, // row 2
    };

    std::vector<float>   out_lp(L * K);
    std::vector<int32_t> out_id(L * K);

    extract_top_k_logprobs(logits, L, V, K, 1.0f,
                           out_lp.data(), out_id.data());

    // Check row 0: top-3 tokens by logit are ids 7, 6, 5.
    assert(out_id[0*K+0] == 7);
    assert(out_id[0*K+1] == 6);
    assert(out_id[0*K+2] == 5);
    // Check log-prob values against manual logsumexp.
    {
        const float log_z = logsumexp_vec(logits + 0*V, V);
        assert(std::fabs(out_lp[0*K+0] - (7.0f - log_z)) < 1e-5f);
        assert(std::fabs(out_lp[0*K+1] - (6.0f - log_z)) < 1e-5f);
        assert(std::fabs(out_lp[0*K+2] - (5.0f - log_z)) < 1e-5f);
    }

    // Check row 1: top-3 tokens are ids 0, 1, 2.
    assert(out_id[1*K+0] == 0);
    assert(out_id[1*K+1] == 1);
    assert(out_id[1*K+2] == 2);
    {
        const float log_z = logsumexp_vec(logits + 1*V, V);
        assert(std::fabs(out_lp[1*K+0] - (7.0f - log_z)) < 1e-5f);
        assert(std::fabs(out_lp[1*K+1] - (6.0f - log_z)) < 1e-5f);
        assert(std::fabs(out_lp[1*K+2] - (5.0f - log_z)) < 1e-5f);
    }

    // Check row 2: id=4 dominates with logit=10.
    assert(out_id[2*K+0] == 4);
    {
        const float log_z = logsumexp_vec(logits + 2*V, V);
        assert(std::fabs(out_lp[2*K+0] - (10.0f - log_z)) < 1e-5f);
    }

    // Verify descending order within each row.
    for (int row = 0; row < L; row++) {
        for (int k = 0; k < K - 1; k++) {
            assert(out_lp[row*K+k] >= out_lp[row*K+k+1]);
        }
    }

    printf("PASS: test_extract_top_k_logprobs\n");
}

// ---------------------------------------------------------------------------

int main() {
    test_build_ddtree_small();
    test_follow_verified_tree();
    test_build_tree_visibility();
    test_extract_top_k_logprobs();
    printf("All tests passed.\n");
    return 0;
}
