// dflash-draft.cpp — Graph builder for the z-lab/Qwen3.5-27B-DFlash speculative draft model.
//
// Architecture: 5-layer non-causal transformer.
// Inputs (host-provided, set via ggml_set_input):
//   - noise_embed    : [n_embd, block_size]      — pre-looked-up rows from target tok_embd
//   - target_feat_raw: [5*n_embd, ctx_len]        — stacked hidden captures from target layers
// Forward pass (mirrors qwen3_dflash_graph.cpp:53-164):
//   1. fc(target_feat_raw) → rms_norm(hidden_norm) → target_feat [n_embd, ctx_len]
//   2. For each of 5 layers:
//      - Q: from noise_embed only (attn_norm, wq, q_norm, RoPE-NEOX)
//      - K/V: from concat(target_feat, noise), then wk/wv, k_norm, RoPE-NEOX
//      - Non-causal FlashAttention (mask=nullptr), GQA 32:8
//      - SwiGLU FFN
//   3. out_norm + shared lm_head → logits [vocab, block_size]

#include "models.h"

llm_build_dflash_draft::llm_build_dflash_draft(
        const llama_model  & model,
        const llm_graph_params & params) : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_k();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());

    // draft constants derived from hparams
    const int64_t block_size = (int64_t)hparams.dflash_block_size;  // 16 noise tokens
    const int64_t n_embd_fc  = (int64_t)5 * n_embd;                 // 5*hidden for fc input

    // rope_theta = 10M for draft (matches DFLASH27B_ROPE_THETA)
    const float draft_rope_theta = 10000000.0f;
    const float scale = 1.0f / sqrtf((float)n_embd_head);

    // ── Draft-specific inputs ─────────────────────────────────────────────────
    // noise_embed: pre-computed embedding rows [n_embd, block_size] — host fills this.
    // Shape uses n_tokens from ubatch (== block_size when calling the draft).
    ggml_tensor * noise_embed = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(noise_embed, "dflash_noise_embed");
    ggml_set_input(noise_embed);

    // target_feat_raw: [5*n_embd, ctx_len] packed hidden captures from target.
    // We represent ctx_len as a second batch dimension via a 2D tensor.
    // For Phase 3, the host uploads this for each draft step.
    // The tensor ne[1] is the number of context tokens captured from the target.
    // We store it as a separate graph input and split via view inside the graph.
    //
    // Use GGML_TYPE_F32 to avoid FA mask type issues.
    // ctx_len comes from ubatch metadata; for Phase 3 we fix it to n_ctx.
    const int64_t ctx_len = n_ctx; // will be overridden by actual context at runtime
    ggml_tensor * target_feat_raw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_fc, ctx_len);
    ggml_set_name(target_feat_raw, "dflash_target_feat_raw");
    ggml_set_input(target_feat_raw);

    // ── Step 1: feature fusion ────────────────────────────────────────────────
    // target_feat = rms_norm(fc @ target_feat_raw, hidden_norm)
    // fc:              [n_embd_fc, n_embd]   (ggml: ne[0]=n_embd_fc, ne[1]=n_embd)
    // target_feat_raw: [n_embd_fc, ctx_len]
    // Result:          [n_embd, ctx_len]
    ggml_tensor * target_feat = ggml_mul_mat(ctx0, model.dflash_fc, target_feat_raw);
    cb(target_feat, "dflash_fc_out", -1);

    target_feat = ggml_rms_norm(ctx0, target_feat, hparams.f_norm_rms_eps);
    target_feat = ggml_mul(ctx0, target_feat, model.dflash_hidden_norm);
    cb(target_feat, "dflash_target_feat", -1);

    // ── Step 2: position tensors ──────────────────────────────────────────────
    // Q positions: [ctx_len .. ctx_len + block_size - 1]  (noise tokens follow context)
    // K positions: [0 .. ctx_len + block_size - 1]        (context then noise)
    //
    // For Phase 3, we create constant position tensors.
    // In Phase 4 these will be driven by the actual KV fill position.
    ggml_tensor * pos_q = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);  // n_tokens == block_size
    ggml_set_name(pos_q, "dflash_pos_q");
    ggml_set_input(pos_q);

    const int64_t total_k = ctx_len + n_tokens;
    ggml_tensor * pos_k = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, total_k);
    ggml_set_name(pos_k, "dflash_pos_k");
    ggml_set_input(pos_k);

    // ── Step 3: 5-layer decoder ───────────────────────────────────────────────
    ggml_tensor * h = noise_embed; // [n_embd, block_size]

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // -- Attention pre-norm on noise hidden state
        ggml_tensor * hn = ggml_rms_norm(ctx0, h, hparams.f_norm_rms_eps);
        hn = ggml_mul(ctx0, hn, layer.attn_norm);  // layer.attn_norm: [n_embd]
        cb(hn, "attn_norm", il);

        // -- Q from noise only: wq [n_embd, n_head*n_embd_head], reshaped, q_norm, RoPE
        ggml_tensor * Q = ggml_mul_mat(ctx0, layer.wq, hn);  // [n_head*n_embd_head, block_size]
        Q = ggml_reshape_3d(ctx0, Q, n_embd_head, n_head, n_tokens);  // [n_embd_head, n_head, block_size]
        Q = ggml_rms_norm(ctx0, Q, hparams.f_norm_rms_eps);            // per-head rms_norm along n_embd_head
        Q = ggml_mul(ctx0, Q, layer.attn_q_norm);                       // broadcast [n_embd_head]
        cb(Q, "Q_normed", il);

        // Q RoPE-NEOX
        Q = ggml_rope_ext(ctx0, Q, pos_q, nullptr,
                          (int)n_embd_head,
                          (int)LLAMA_ROPE_TYPE_NEOX,
                          /*n_ctx_orig=*/0,
                          draft_rope_theta,
                          /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f,
                          /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f,
                          /*beta_slow=*/0.0f);
        cb(Q, "Q_rope", il);

        // -- K and V from concat(target_feat, noise)
        // First compute K/V from target_feat (ctx_len tokens)
        ggml_tensor * Kctx = ggml_mul_mat(ctx0, layer.wk, target_feat); // [n_head_kv*n_embd_head, ctx_len]
        ggml_tensor * Vctx = ggml_mul_mat(ctx0, layer.wv, target_feat);

        // Then from noise (block_size tokens)
        ggml_tensor * Kn = ggml_mul_mat(ctx0, layer.wk, hn);            // [n_head_kv*n_embd_head, block_size]
        ggml_tensor * Vn = ggml_mul_mat(ctx0, layer.wv, hn);

        // Concat along sequence dimension (ne[1])
        ggml_tensor * K = ggml_concat(ctx0, Kctx, Kn, 1);  // [n_head_kv*n_embd_head, total_k]
        ggml_tensor * V = ggml_concat(ctx0, Vctx, Vn, 1);

        // Per-head K norm
        K = ggml_reshape_3d(ctx0, K, n_embd_head, n_head_kv, total_k);
        K = ggml_rms_norm(ctx0, K, hparams.f_norm_rms_eps);
        K = ggml_mul(ctx0, K, layer.attn_k_norm);
        cb(K, "K_normed", il);

        V = ggml_reshape_3d(ctx0, V, n_embd_head, n_head_kv, total_k);

        // K RoPE-NEOX
        K = ggml_rope_ext(ctx0, K, pos_k, nullptr,
                          (int)n_embd_head,
                          (int)LLAMA_ROPE_TYPE_NEOX,
                          0,
                          draft_rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        cb(K, "K_rope", il);

        // Permute into flash_attn_ext layout
        // Q: [n_embd_head, n_head,    block_size, 1]
        // K: [n_embd_head, n_head_kv, total_k,    1]
        // V: [n_embd_head, n_head_kv, total_k,    1] (not transposed)
        Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
        Q = ggml_cont(ctx0, Q);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        K = ggml_cont(ctx0, K);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        V = ggml_cont(ctx0, V);

        // Non-causal flash attention; mask=nullptr, GQA broadcast handled internally.
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, Q, K, V,
                                                  /*mask=*/nullptr,
                                                  scale,
                                                  /*max_bias=*/0.0f,
                                                  /*logit_softcap=*/0.0f);
        // attn: [n_embd_head, n_head, block_size, 1]
        attn = ggml_reshape_2d(ctx0, attn, n_embd_head * n_head, n_tokens);
        cb(attn, "attn_out", il);

        // Output projection + residual
        ggml_tensor * attn_proj = ggml_mul_mat(ctx0, layer.wo, attn);
        h = ggml_add(ctx0, h, attn_proj);
        cb(h, "attn_residual", il);

        // -- FFN pre-norm
        ggml_tensor * hf = ggml_rms_norm(ctx0, h, hparams.f_norm_rms_eps);
        hf = ggml_mul(ctx0, hf, layer.ffn_norm);
        cb(hf, "ffn_norm", il);

        // SwiGLU: down(silu(gate(x)) * up(x))
        ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate, hf);
        g = ggml_silu(ctx0, g);
        ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up, hf);
        ggml_tensor * gu = ggml_mul(ctx0, g, u);
        ggml_tensor * ffn_out = ggml_mul_mat(ctx0, layer.ffn_down, gu);
        cb(ffn_out, "ffn_out", il);

        h = ggml_add(ctx0, h, ffn_out);
        cb(h, "l_out", il);
    }

    // ── Step 4: final norm + lm_head ─────────────────────────────────────────
    ggml_tensor * out = ggml_rms_norm(ctx0, h, hparams.f_norm_rms_eps);
    out = ggml_mul(ctx0, out, model.output_norm); // model.output_norm == dflash out_norm
    cb(out, "result_norm", -1);
    res->t_embd = out;

    // lm_head is shared from target model — it must be provided via model.output.
    // If not yet wired (Phase 3 test mode), output is the hidden state only.
    if (model.output != nullptr) {
        ggml_tensor * logits = ggml_mul_mat(ctx0, model.output, out);
        cb(logits, "result_output", -1);
        res->t_logits = logits;
        ggml_build_forward_expand(gf, logits);
    } else {
        ggml_build_forward_expand(gf, out);
    }
}
