#pragma once

#include "llama.h"
#include "speculative-tree.h"

#include <cstdint>
#include <memory>
#include <vector>

struct llama_speculative_draft_target_feat_view {
    const float * ring        = nullptr;
    int64_t       n_committed = 0;
    int64_t       cap         = 0;
    int64_t       n_embd_fc   = 0;
};

struct llama_speculative_draft_decode_info {
    int     L       = 0;
    int     K       = 0;
    int64_t ctx_len = 0;

    double t_target_feat_pack_ms = 0.0;
    double t_draft_decode_ms     = 0.0;
    double t_topk_ms             = 0.0;
};

int llama_speculative_draft_top_k_width(int block_size, const llama_ddtree_params & params);

bool llama_speculative_draft_pack_target_feat(const llama_speculative_draft_target_feat_view & view,
                                              std::vector<float> &                             out,
                                              int64_t &                                        ctx_len);

class llama_speculative_draft_backend {
  public:
    virtual ~llama_speculative_draft_backend() = default;

    virtual const char * name() const = 0;

    virtual bool decode_topk(llama_token                                      root_token,
                             llama_pos                                        committed_pos,
                             const llama_speculative_draft_target_feat_view & target_feat,
                             std::vector<float> &                             top_log_probs,
                             std::vector<int32_t> &                           top_token_ids,
                             llama_speculative_draft_decode_info &            info) = 0;
};

std::unique_ptr<llama_speculative_draft_backend> llama_speculative_draft_backend_init_llama(
    llama_context *             draft_ctx,
    const llama_model *         target_model,
    int64_t                     n_embd,
    int64_t                     n_vocab,
    int64_t                     block_size,
    llama_token                 mask_token_id,
    const llama_ddtree_params & params);
