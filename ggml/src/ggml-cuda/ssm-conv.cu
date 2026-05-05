#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// dflash27b_ggml: tree-mode ssm_conv kernel. For each new-token t, walks up
// the parent chain K-1 times via parent_ids[] to find the (K-1) ancestor slots
// in the conv input, then convolves with the kernel weights. Virtual-slot
// encoding: a non-negative parent index `p` maps to sx slot (K-1 + p). A
// parent index of -1 means "before the block" — i.e., the old conv state.
// Each successive walk beyond -1 decrements by 1, so virtual slot -k maps to
// sx slot (K-1 - k), which indexes into the old state region [0, K-1). This
// matches SGLang's causal_conv1d_triton HAS_EAGLE_TREE_CUSTOM_ATTN_MASK path.
// dflash27b_ggml: tree-mode + per-token persistent conv state. When
// WITH_PERSIST is true, every token writes its (K-1)-element conv "post-state"
// (the last K-1 cols of its parent-chain window) into persist_inter so the
// driver can roll the live conv state back to the accepted DFS node.
template <bool apply_silu, size_t split_d_inner, size_t d_conv, bool WITH_PERSIST = false>
static __global__ void ssm_conv_tree_f32(
        const float * __restrict__ src0,        // sx: [K-1+n_t, d_inner, n_s]
        const float * __restrict__ src1,        // c:  [K, d_inner]
        const int * __restrict__   parent_ids,  // [n_t, n_s]
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * __restrict__ dst,               // [d_inner, n_t, n_s]
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        float * __restrict__ persist_inter,     // [K-1, d_inner, n_t, n_s] when WITH_PERSIST, else nullptr
        const int64_t d_inner_total,            // full d_inner for persist row stride
        const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;  // sequence
    const int bidy = blockIdx.y;  // d_inner / split_d_inner

    const float * x_block = (const float *) ((const char *) src0
        + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1
        + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst
        + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // Load kernel weights into registers.
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    const int * parent_ids_seq = parent_ids + bidx * n_t;

    // Channel index this thread owns within the full d_inner dimension.
    // Used both for indexing persist_inter (when enabled) and as bookkeeping.
    const int channel = (int)(bidy * split_d_inner) + tid;

    for (int64_t i = 0; i < n_t; i++) {
        // Walk the parent chain K-1 times to fill the conv window.
        // ancestor_virtual[k] gives the "virtual slot" for kernel position k,
        // where the most recent slot is at k=K-1 (= token i itself) and older
        // slots are at k=K-2, K-3, ..., 0.
        //
        // ancestor_virtual[K-1] = i
        // ancestor_virtual[K-2] = parent_of(i) (or i-1 decay for negative)
        // ancestor_virtual[k  ] = parent_of(ancestor_virtual[k+1])
        int ancestors[d_conv];
        ancestors[d_conv - 1] = (int)i;
#pragma unroll
        for (int k = (int)d_conv - 2; k >= 0; k--) {
            int prev = ancestors[k + 1];
            int next;
            if (prev >= 0) {
                next = parent_ids_seq[prev];  // -1 if parent is before block
            } else {
                next = prev - 1;  // keep decaying through old state slots
            }
            ancestors[k] = next;
        }

        float sumf = 0.0f;
        // Cache window values so we can both convolve and (optionally) persist
        // them without re-reading from global memory.
        float window[d_conv];
#pragma unroll
        for (size_t k = 0; k < d_conv; k++) {
            // Map virtual slot → sx slot: sx_slot = (K-1) + ancestors[k].
            const int sx_slot = (int)(d_conv - 1) + ancestors[k];
            window[k] = x_block[tid * stride_x + sx_slot];
            sumf += window[k] * w[k];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

        if constexpr (WITH_PERSIST) {
            // Per-token "post-state": the (K-1) most recent cols of this token's
            // window — i.e. ancestors[1..K-1]. Layout matches the live conv state
            // tensor (r_l): [K-1, d_inner, ...] with K-1 fastest. Persist memory
            // layout: persist_inter[s][t][channel][k] flat = ((s*n_t + t)*d_inner + channel) * (K-1) + k.
            float * persist_token = persist_inter
                + ((bidx * n_t + i) * d_inner_total + channel) * (int64_t)(d_conv - 1);
#pragma unroll
            for (size_t k = 0; k < d_conv - 1; k++) {
                // ancestors[1] is the oldest col we keep; ancestors[K-1] = self.
                persist_token[k] = window[k + 1];
            }
        }
    }
}

template <bool apply_silu>
static void ssm_conv_tree_f32_cuda(const float * src0, const float * src1, const int * parent_ids,
                                   const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                   const int dst_nb2, float * persist_inter,
                                   const int64_t nc, const int64_t nr,
                                   const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (persist_inter != nullptr) {
            ssm_conv_tree_f32<apply_silu, threads, kNC, /*WITH_PERSIST=*/true><<<blocks, threads, 0, stream>>>(
                src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                dst, dst_nb0, dst_nb1, dst_nb2, persist_inter, nr, n_t);
        } else {
            ssm_conv_tree_f32<apply_silu, threads, kNC, /*WITH_PERSIST=*/false><<<blocks, threads, 0, stream>>>(
                src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                dst, dst_nb0, dst_nb1, dst_nb2, nullptr, nr, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Tree ssm_conv only supports kernel sizes 3, 4, 5, 9.");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    // dflash27b_ggml: optional src[2] = parent_ids (i32) enables tree mode
    const struct ggml_tensor * parent_ids = dst->src[2];
    const bool fuse_silu = silu_dst != nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);

    if (parent_ids != nullptr) {
        GGML_ASSERT(parent_ids->type == GGML_TYPE_I32);
        const int * parent_ids_d = (const int *) parent_ids->data;
        // dflash27b_ggml: optional src[3] = persist_inter (F32) buffer where
        // each token's [K-1, d_inner] post-state is written for SSM rollback.
        const struct ggml_tensor * persist_inter = dst->src[3];
        float * persist_d = nullptr;
        if (persist_inter != nullptr) {
            GGML_ASSERT(persist_inter->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(persist_inter));
            GGML_ASSERT(ggml_nelements(persist_inter) >= (int64_t)(nc - 1) * nr * n_t * n_s);
            persist_d = (float *) persist_inter->data;
        }
        if (fuse_silu) {
            ssm_conv_tree_f32_cuda<true>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                persist_d, nc, nr, n_t, n_s, stream);
        } else {
            ssm_conv_tree_f32_cuda<false>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                persist_d, nc, nr, n_t, n_s, stream);
        }
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
