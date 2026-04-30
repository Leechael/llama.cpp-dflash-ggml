#include "speculative-draft-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static void require(bool ok, const char * expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "test-speculative-draft-backend:%d: check failed: %s\n", line, expr);
        std::abort();
    }
}

#define REQUIRE(expr) require((expr), #expr, __LINE__)

static void test_top_k_width() {
    llama_ddtree_params p;
    p.block_size = 16;
    p.budget     = 40;
    p.top_k      = 0;
    REQUIRE(llama_speculative_draft_top_k_width(p.block_size, p) == 8);

    p.budget = 8;
    REQUIRE(llama_speculative_draft_top_k_width(p.block_size, p) == 1);

    p.top_k = 4;
    REQUIRE(llama_speculative_draft_top_k_width(p.block_size, p) == 4);
}

static void test_pack_target_feat_no_wrap() {
    const int64_t fc  = 3;
    const int64_t cap = 4;
    std::vector<float> ring((size_t) fc * cap);
    for (int64_t col = 0; col < cap; ++col) {
        for (int64_t row = 0; row < fc; ++row) {
            ring[(size_t) col * fc + row] = (float) (10 * col + row);
        }
    }

    llama_speculative_draft_target_feat_view view{ ring.data(), 3, cap, fc };
    std::vector<float>                       out;
    int64_t                                  ctx_len = 0;
    REQUIRE(llama_speculative_draft_pack_target_feat(view, out, ctx_len));
    REQUIRE(ctx_len == 3);
    REQUIRE(out.size() == 9);

    for (int64_t col = 0; col < ctx_len; ++col) {
        for (int64_t row = 0; row < fc; ++row) {
            REQUIRE(out[(size_t) col * fc + row] == ring[(size_t) col * fc + row]);
        }
    }
}

static void test_pack_target_feat_wrap() {
    const int64_t fc  = 3;
    const int64_t cap = 4;
    std::vector<float> ring((size_t) fc * cap);

    // logical columns 2, 3, 4, 5 live in ring slots 2, 3, 0, 1.
    const int64_t logical_by_slot[4] = { 4, 5, 2, 3 };
    for (int64_t slot = 0; slot < cap; ++slot) {
        const int64_t logical = logical_by_slot[slot];
        for (int64_t row = 0; row < fc; ++row) {
            ring[(size_t) slot * fc + row] = (float) (100 * logical + row);
        }
    }

    llama_speculative_draft_target_feat_view view{ ring.data(), 6, cap, fc };
    std::vector<float>                       out;
    int64_t                                  ctx_len = 0;
    REQUIRE(llama_speculative_draft_pack_target_feat(view, out, ctx_len));
    REQUIRE(ctx_len == cap);
    REQUIRE(out.size() == (size_t) fc * cap);

    for (int64_t col = 0; col < ctx_len; ++col) {
        const int64_t logical = 2 + col;
        for (int64_t row = 0; row < fc; ++row) {
            REQUIRE(out[(size_t) col * fc + row] == (float) (100 * logical + row));
        }
    }
}

static void test_pack_target_feat_empty() {
    std::vector<float> out{ 1.0f };
    int64_t            ctx_len = 123;
    llama_speculative_draft_target_feat_view view{};
    REQUIRE(!llama_speculative_draft_pack_target_feat(view, out, ctx_len));
    REQUIRE(ctx_len == 0);
}

int main() {
    test_top_k_width();
    test_pack_target_feat_no_wrap();
    test_pack_target_feat_wrap();
    test_pack_target_feat_empty();
    return 0;
}
