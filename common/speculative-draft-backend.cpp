#include "speculative-draft-backend.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using ddtree_draft_clock = std::chrono::steady_clock;

static double draft_elapsed_ms(ddtree_draft_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(ddtree_draft_clock::now() - t0).count();
}

int llama_speculative_draft_top_k_width(int block_size, const llama_ddtree_params & params) {
    if (params.top_k > 0) {
        return params.top_k;
    }
    // Standalone DFlash only asks the draft for top-K branches when the DDTree
    // budget can grow beyond the greedy chain; it uses K=8 in that case.
    return (params.budget > std::max(0, block_size - 1)) ? 8 : 1;
}

bool llama_speculative_draft_pack_target_feat(const llama_speculative_draft_target_feat_view & view,
                                              std::vector<float> &                             out,
                                              int64_t &                                        ctx_len) {
    ctx_len = 0;
    if (view.ring == nullptr || view.n_committed <= 0 || view.cap <= 0 || view.n_embd_fc <= 0) {
        return false;
    }

    ctx_len                  = std::min(view.n_committed, view.cap);
    const int64_t ring_start = view.n_committed - ctx_len;

    out.resize((size_t) view.n_embd_fc * ctx_len);
    for (int64_t t = 0; t < ctx_len; ++t) {
        const int64_t ring_col = (ring_start + t) % view.cap;
        const float * src      = view.ring + ring_col * view.n_embd_fc;
        float *       dst      = out.data() + t * view.n_embd_fc;
        memcpy(dst, src, (size_t) view.n_embd_fc * sizeof(float));
    }
    return true;
}

class llama_speculative_llama_draft_backend final : public llama_speculative_draft_backend {
  public:
    llama_speculative_llama_draft_backend(llama_context *             draft_ctx,
                                          const llama_model *         target_model,
                                          int64_t                     n_embd,
                                          int64_t                     n_vocab,
                                          int64_t                     block_size,
                                          llama_token                 mask_token_id,
                                          const llama_ddtree_params & params) :
        draft_ctx(draft_ctx),
        target_model(target_model),
        n_embd(n_embd),
        n_vocab(n_vocab),
        block_size(block_size),
        mask_token_id(mask_token_id),
        params(params) {
        mask_embd.resize((size_t) n_embd);
        noise_embd.resize((size_t) block_size * n_embd);
        pos.resize((size_t) block_size);
        n_seq_id.assign((size_t) block_size, 1);
        seq_id_values.assign((size_t) block_size, 0);
        seq_id_ptrs.resize((size_t) block_size);
        logits.assign((size_t) block_size, 1);
        for (int64_t i = 0; i < block_size; ++i) {
            seq_id_ptrs[(size_t) i] = &seq_id_values[(size_t) i];
        }
    }

    bool init() {
        if (draft_ctx == nullptr || target_model == nullptr || n_embd <= 0 || n_vocab <= 0 || block_size <= 1) {
            return false;
        }
        if (llama_model_token_embd_lookup(target_model, mask_token_id, mask_embd.data(), n_embd) != 0) {
            LOG_ERR("%s: token_embd_lookup failed for mask_token=%d\n", __func__, (int) mask_token_id);
            return false;
        }
        for (int64_t i = 1; i < block_size; ++i) {
            memcpy(noise_embd.data() + i * n_embd, mask_embd.data(), (size_t) n_embd * sizeof(float));
        }
        llama_set_dflash_draft_top_k(draft_ctx,
                                     std::min<int64_t>(llama_speculative_draft_top_k_width((int) block_size, params),
                                                       n_vocab));
        return true;
    }

    const char * name() const override { return "dflash-topk"; }

    bool ingest_target_capture(llama_context * target_ctx,
                               const int32_t * dfs_indices,
                               int32_t         n_dfs,
                               int64_t         first_pos,
                               int64_t         cap,
                               double &        elapsed_ms) override {
        const auto t0 = ddtree_draft_clock::now();
        elapsed_ms = 0.0;
        if (target_ctx == nullptr || n_dfs <= 0 || cap <= 0) {
            return false;
        }
        const int ret = llama_dflash_draft_update_fused_cache_from_capture(draft_ctx, target_ctx, dfs_indices,
                                                                           n_dfs, first_pos, cap);
        elapsed_ms = draft_elapsed_ms(t0);
        if (ret != 0) {
            return false;
        }
        fused_target_feat_cap = cap;
        fused_target_feat_n_embd = n_embd;
        fused_target_feat_n_committed = first_pos + n_dfs;
        return true;
    }

    bool decode_topk(llama_token                                      root_token,
                     llama_pos                                        committed_pos,
                     const llama_speculative_draft_target_feat_view & target_feat,
                     std::vector<float> &                             top_log_probs,
                     std::vector<int32_t> &                           top_token_ids,
                     llama_speculative_draft_decode_info &            info) override {
        info   = {};
        info.L = (int) block_size - 1;
        info.K = std::min<int64_t>(llama_speculative_draft_top_k_width((int) block_size, params), n_vocab);

        if (target_feat.ring == nullptr || target_feat.n_committed <= 0 || target_feat.cap <= 0 ||
            target_feat.n_embd_fc <= 0 || target_feat.n_embd_fc % 5 != 0) {
            LOG_ERR(
                "%s: target_feat ring is empty; call llama_speculative_tree_driver_ingest_prompt_capture first\n",
                __func__);
            return false;
        }

        const int64_t fused_n_embd = target_feat.n_embd_fc / 5;
        {
            const auto t0 = ddtree_draft_clock::now();
            if (!ensure_fused_target_feat(target_feat, fused_n_embd)) {
                info.t_draft_decode_ms += draft_elapsed_ms(t0);
                return false;
            }
            info.t_draft_decode_ms += draft_elapsed_ms(t0);
        }

        info.ctx_len = std::min(target_feat.n_committed, target_feat.cap);
        const int64_t ring_start = target_feat.n_committed - info.ctx_len;

        {
            const auto t0 = ddtree_draft_clock::now();

            if (llama_model_token_embd_lookup(target_model, root_token, noise_embd.data(), n_embd) != 0) {
                LOG_ERR("%s: token_embd_lookup failed for root_token=%d\n", __func__, (int) root_token);
                info.t_draft_decode_ms += draft_elapsed_ms(t0);
                return false;
            }
            for (int32_t i = 0; i < (int32_t) block_size; ++i) {
                pos[(size_t) i] = committed_pos + i;
            }

            llama_batch draft_batch{};
            draft_batch.n_tokens  = (int32_t) block_size;
            draft_batch.token     = nullptr;
            draft_batch.embd      = noise_embd.data();
            draft_batch.pos       = pos.data();
            draft_batch.n_seq_id  = n_seq_id.data();
            draft_batch.seq_id    = seq_id_ptrs.data();
            draft_batch.logits    = logits.data();
            draft_batch.parent_id = nullptr;

            const int ret = llama_dflash_draft_encode_top_k_cached(draft_ctx, draft_batch,
                                                                   fused_n_embd, info.ctx_len,
                                                                   ring_start, target_feat.cap,
                                                                   committed_pos, info.K);
            if (ret != 0) {
                LOG_ERR("%s: dflash draft encode-topK failed: %d\n", __func__, ret);
                info.t_draft_decode_ms += draft_elapsed_ms(t0);
                return false;
            }
            info.t_draft_decode_ms += draft_elapsed_ms(t0);
        }

        top_log_probs.resize((size_t) info.L * info.K);
        top_token_ids.resize((size_t) info.L * info.K);

        {
            const auto t0 = ddtree_draft_clock::now();

            float proposal_temp = params.temp;
            if (const char * e = std::getenv("LLAMA_DDTREE_PROPOSAL_TEMP")) {
                char *      end = nullptr;
                const float v   = std::strtof(e, &end);
                if (end != e && v > 0.0f) {
                    proposal_temp = v;
                }
            }
            const float inv_t = 1.0f / std::max(1e-6f, proposal_temp);

            const float *     draft_top_logits = nullptr;
            const llama_token * draft_top_tokens = nullptr;
            int32_t           top_rows         = 0;
            int32_t           top_k            = 0;
            if (!llama_get_dflash_draft_top_k(draft_ctx, &draft_top_logits, &draft_top_tokens, &top_rows, &top_k) ||
                draft_top_logits == nullptr || draft_top_tokens == nullptr || top_rows < (int32_t) block_size ||
                top_k < info.K) {
                LOG_ERR("%s: dflash draft top-K unavailable\n", __func__);
                info.t_topk_ms += draft_elapsed_ms(t0);
                return false;
            }

            struct Entry {
                float       logit;
                llama_token token;
            };
            std::vector<Entry> row_top((size_t) info.K);

            for (int i = 0; i < info.L; ++i) {
                const int row_idx = i + 1;
                for (int k = 0; k < info.K; ++k) {
                    row_top[(size_t) k] = {
                        draft_top_logits[(size_t) row_idx * top_k + k],
                        draft_top_tokens[(size_t) row_idx * top_k + k],
                    };
                }
                std::sort(row_top.begin(), row_top.end(), [](const Entry & a, const Entry & b) {
                    return a.logit > b.logit;
                });

                if (std::abs(proposal_temp - 1.0f) < 1e-6f) {
                    for (int k = 0; k < info.K; ++k) {
                        top_log_probs[(size_t) i * info.K + k] = row_top[(size_t) k].logit;
                        top_token_ids[(size_t) i * info.K + k] = row_top[(size_t) k].token;
                    }
                    continue;
                }

                const float row_best = row_top[0].logit * inv_t;
                float sum_exp_top = 0.0f;
                for (int k = 0; k < info.K; ++k) {
                    sum_exp_top += std::exp(row_top[(size_t) k].logit * inv_t - row_best);
                }
                const float log_z_approx = row_best + std::log(sum_exp_top);
                for (int k = 0; k < info.K; ++k) {
                    top_log_probs[(size_t) i * info.K + k] = row_top[(size_t) k].logit * inv_t - log_z_approx;
                    top_token_ids[(size_t) i * info.K + k] = row_top[(size_t) k].token;
                }
            }
            info.t_topk_ms += draft_elapsed_ms(t0);
        }

        return true;
    }

  private:
    bool ensure_fused_target_feat(const llama_speculative_draft_target_feat_view & target_feat,
                                  int64_t fused_n_embd) {
        const int64_t ctx_len = std::min(target_feat.n_committed, target_feat.cap);
        const int64_t ring_start = target_feat.n_committed - ctx_len;

        if (fused_target_feat_cap != target_feat.cap || fused_target_feat_n_embd != fused_n_embd) {
            fused_target_feat_cap = target_feat.cap;
            fused_target_feat_n_embd = fused_n_embd;
            fused_target_feat_n_committed = ring_start;
        }

        if (fused_target_feat_n_committed < ring_start || fused_target_feat_n_committed > target_feat.n_committed) {
            fused_target_feat_n_committed = ring_start;
        }

        const int64_t missing = target_feat.n_committed - fused_target_feat_n_committed;
        if (missing <= 0) {
            return true;
        }

        raw_fuse_buf.resize((size_t) target_feat.n_embd_fc * missing);
        for (int64_t t = 0; t < missing; ++t) {
            const int64_t logical_col = fused_target_feat_n_committed + t;
            const int64_t ring_col = logical_col % target_feat.cap;
            const float * src = target_feat.ring + ring_col * target_feat.n_embd_fc;
            float * dst = raw_fuse_buf.data() + t * target_feat.n_embd_fc;
            memcpy(dst, src, (size_t) target_feat.n_embd_fc * sizeof(float));
        }

        const int ret = llama_dflash_draft_update_fused_cache(draft_ctx, raw_fuse_buf.data(), target_feat.n_embd_fc,
                                                              missing, fused_target_feat_n_committed,
                                                              target_feat.cap);
        if (ret != 0) {
            LOG_ERR("%s: dflash target_feat cache update failed: %d\n", __func__, ret);
            return false;
        }

        fused_target_feat_n_committed = target_feat.n_committed;
        return true;
    }

    llama_context *     draft_ctx     = nullptr;
    const llama_model * target_model  = nullptr;
    int64_t             n_embd        = 0;
    int64_t             n_vocab       = 0;
    int64_t             block_size    = 0;
    llama_token         mask_token_id = 0;
    llama_ddtree_params params;

    std::vector<float>          target_feat_buf;
    std::vector<float>          raw_fuse_buf;
    int64_t                     fused_target_feat_n_committed = 0;
    int64_t                     fused_target_feat_n_embd = 0;
    int64_t                     fused_target_feat_cap = 0;
    std::vector<float>          mask_embd;
    std::vector<float>          noise_embd;
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_values;
    std::vector<llama_seq_id *> seq_id_ptrs;
    std::vector<int8_t>         logits;
};

std::unique_ptr<llama_speculative_draft_backend> llama_speculative_draft_backend_init_llama(
    llama_context *             draft_ctx,
    const llama_model *         target_model,
    int64_t                     n_embd,
    int64_t                     n_vocab,
    int64_t                     block_size,
    llama_token                 mask_token_id,
    const llama_ddtree_params & params) {
    auto backend = std::make_unique<llama_speculative_llama_draft_backend>(draft_ctx, target_model, n_embd, n_vocab,
                                                                           block_size, mask_token_id, params);
    if (!backend->init()) {
        return nullptr;
    }
    return backend;
}
