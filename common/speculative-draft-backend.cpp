#include "speculative-draft-backend.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

using ddtree_draft_clock = std::chrono::steady_clock;

static double draft_elapsed_ms(ddtree_draft_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(ddtree_draft_clock::now() - t0).count();
}

int llama_speculative_draft_top_k_width(int block_size, const llama_ddtree_params & params) {
    const int L = block_size - 1;
    return (params.top_k > 0) ? params.top_k : ((params.budget > L) ? 8 : 1);
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
        return true;
    }

    const char * name() const override { return "llama"; }

    bool decode_topk(llama_token                                      root_token,
                     llama_pos                                        committed_pos,
                     const llama_speculative_draft_target_feat_view & target_feat,
                     std::vector<float> &                             top_log_probs,
                     std::vector<int32_t> &                           top_token_ids,
                     llama_speculative_draft_decode_info &            info) override {
        info   = {};
        info.L = (int) block_size - 1;
        info.K = llama_speculative_draft_top_k_width((int) block_size, params);

        {
            const auto t0 = ddtree_draft_clock::now();
            if (!llama_speculative_draft_pack_target_feat(target_feat, target_feat_buf, info.ctx_len)) {
                LOG_ERR(
                    "%s: target_feat ring is empty; call llama_speculative_tree_driver_ingest_prompt_capture first\n",
                    __func__);
                info.t_target_feat_pack_ms += draft_elapsed_ms(t0);
                return false;
            }
            llama_set_target_feat_raw(draft_ctx, target_feat_buf.data(), target_feat.n_embd_fc, info.ctx_len,
                                      committed_pos);
            info.t_target_feat_pack_ms += draft_elapsed_ms(t0);
        }

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

            const int ret = llama_encode(draft_ctx, draft_batch);
            if (ret != 0) {
                LOG_ERR("%s: draft llama_encode failed: %d\n", __func__, ret);
                info.t_draft_decode_ms += draft_elapsed_ms(t0);
                return false;
            }
            info.t_draft_decode_ms += draft_elapsed_ms(t0);
        }

        top_log_probs.resize((size_t) info.L * info.K);
        top_token_ids.resize((size_t) info.L * info.K);

        {
            const auto    t0                = ddtree_draft_clock::now();
            const float * draft_logits_row1 = llama_get_logits_ith(draft_ctx, 1);
            if (!draft_logits_row1) {
                LOG_ERR("%s: draft logits unavailable\n", __func__);
                info.t_topk_ms += draft_elapsed_ms(t0);
                return false;
            }
            if (info.K == 1) {
                for (int i = 0; i < info.L; ++i) {
                    const float * row      = draft_logits_row1 + (size_t) i * n_vocab;
                    int32_t       best     = 0;
                    float         best_val = row[0];
                    for (int64_t v = 1; v < n_vocab; ++v) {
                        if (row[v] > best_val) {
                            best_val = row[v];
                            best     = (int32_t) v;
                        }
                    }
                    top_log_probs[(size_t) i] = 0.0f;
                    top_token_ids[(size_t) i] = best;
                }
            } else {
                float proposal_temp = params.temp;
                if (const char * e = std::getenv("LLAMA_DDTREE_PROPOSAL_TEMP")) {
                    char *      end = nullptr;
                    const float v   = std::strtof(e, &end);
                    if (end != e && v > 0.0f) {
                        proposal_temp = v;
                    }
                }
                extract_top_k_logprobs(draft_logits_row1, info.L, (int) n_vocab, info.K, proposal_temp,
                                       top_log_probs.data(), top_token_ids.data());
            }
            info.t_topk_ms += draft_elapsed_ms(t0);
        }

        return true;
    }

  private:
    llama_context *     draft_ctx     = nullptr;
    const llama_model * target_model  = nullptr;
    int64_t             n_embd        = 0;
    int64_t             n_vocab       = 0;
    int64_t             block_size    = 0;
    llama_token         mask_token_id = 0;
    llama_ddtree_params params;

    std::vector<float>          target_feat_buf;
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
