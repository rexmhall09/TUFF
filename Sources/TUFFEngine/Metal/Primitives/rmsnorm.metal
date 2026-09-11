#include <metal_stdlib>
using namespace metal;

// ============================================================================
// rmsnorm — RMS-norm over hidden dim.
//
//   inv     = rsqrt(mean(x[i]^2) + eps)
//   y[i]    = x[i] * inv * weight[i]
//
// FP32 accumulator (numerical stability: D=2816 with FP16 inputs can overflow
// FP16 sum-of-squares once activations grow past ~1.2 in magnitude).
// FP16 storage in and out. Learned weights are BF16 where present.
//
// Dispatch: one threadgroup per row, 256 threads per group. Two-stage block
// reduce — SIMD-group simd_sum, then a single SIMD-group merges the partials.
// ============================================================================

// Threadgroup memory carries at most simdgroups_per_threadgroup = 256/32 = 8
// partial sums. Slot 0 is reused after the merge to broadcast the final inv.
constant constexpr uint kRmsMaxSimdGroups = 8;
constant uint FC_RMS_D [[function_constant(30)]];
constant bool FC_RMS_USE_FC [[function_constant(31)]];

static inline uint rms_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_RMS_USE_FC) &&
            FC_RMS_USE_FC &&
            is_function_constant_defined(FC_RMS_D)) ? FC_RMS_D : D;
}

// Common block reduction. Returns `inv = rsqrt(mean(x^2) + eps)` broadcast to
// every thread via threadgroup memory slot 0.
static inline float rms_block_inv(
    device const half* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        float v = float(x[i]);
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            float mean_sq = v / float(D);
            partial[0] = rsqrt(mean_sq + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

static inline float rms_block_inv_float(
    device const float* x,
    uint  D,
    float eps,
    uint  lid,
    uint  lsize,
    uint  simd_lane_id,
    uint  simd_group_id,
    uint  simdgroups,
    threadgroup float* partial
) {
    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const float v = x[i];
        acc = fma(v, v, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        v = simd_sum(v);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(v / float(D) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// Gemma 4 RMS norms ship as BF16 weight vectors (all 30 layers' input /
// post-attn / pre-FFN / post-FFN norms, plus q/k_norm). Math is identical to
// the no-scale form below, with a learned weight applied after normalization.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w(
    device const half*   x          [[buffer(0)]],   // [D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [D] BF16
    device       half*   out        [[buffer(2)]],   // [D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        float wv = float(weight[i]);
        out[i] = half(xv * inv * wv);
    }
}

// GPT-OSS keeps its residual stream in FP32 so the deeper 120B checkpoint
// cannot overflow FP16 between pre-norm blocks. The normalized projection
// input remains FP16, so attention and expert kernels are unchanged.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_float_bf16w_half(
    device const float*  x          [[buffer(0)]],
    device const bfloat* weight     [[buffer(1)]],
    device       half*   out        [[buffer(2)]],
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const float inv = rms_block_inv_float(x, D, eps, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    for (uint i = lid; i < D; i += lsize) {
        out[i] = half(x[i] * inv * float(weight[i]));
    }
}

// Batched form for GPT-OSS chunked/speculative verification. Each
// threadgroup owns one FP32 residual row and writes one FP16 normalized row.
// The row strides are explicit because the caller keeps the rows in shared
// bounded scratch rather than allocating a temporary matrix per block.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_float_bf16w_half_block(
    device const float*  x          [[buffer(0)]],
    device const bfloat* weight     [[buffer(1)]],
    device       half*   out        [[buffer(2)]],
    constant     uint&   T          [[buffer(3)]],
    constant     uint&   D          [[buffer(4)]],
    constant     float&  eps        [[buffer(5)]],
    constant     uint&   x_stride   [[buffer(6)]],
    constant     uint&   out_stride [[buffer(7)]],
    uint  row             [[threadgroup_position_in_grid]],
    uint  lid             [[thread_position_in_threadgroup]],
    uint  lsize           [[threads_per_threadgroup]],
    uint  simd_lane_id    [[thread_index_in_simdgroup]],
    uint  simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint  simdgroups      [[simdgroups_per_threadgroup]]) {
    if (row >= T) return;
    threadgroup float partial[kRmsMaxSimdGroups];
    device const float* xr = x + row * x_stride;
    device half* outRow = out + row * out_stride;
    const float inv = rms_block_inv_float(xr, D, eps, lid, lsize,
                                          simd_lane_id, simd_group_id,
                                          simdgroups, partial);
    for (uint i = lid; i < D; i += lsize) {
        outRow[i] = half(xr[i] * inv * float(weight[i]));
    }
}

// Gemma 4 applies q_norm/k_norm
// (BF16 weight, shared across heads) and v_norm (no-scale) to each attention
// head independently. These kernels process all heads in one dispatch, with
// one threadgroup per head, avoiding a chain of tiny serialized encoders.
// Math is identical to the single-row kernels applied per head.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, shared per head
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(weight[i]);
        oh[i] = half(xv * inv * wv);
    }
}

// Qwen4-Exp norms store weights centered at zero and scale by (1 + w), the
// way Gemma's reference does and unlike Qwen3.6, whose weights are centered at
// one. Every norm in the architecture takes this form except the gated
// DeltaNet output norm: q_norm, k_norm, the sparse indexer's q/k layernorms,
// the three PLE norms, and the hyper-connection hc_norm.
//
// The offset is applied here rather than folded into the weights at repack
// time on purpose. These weights are BF16, with eight mantissa bits; a stored
// 0.02 carries far more precision than a stored 1.02 does, so adding the one
// before the cast would quantize away most of what the tensor says. The
// reference adds it in float32, and so does this.

// Grouped variant: `groups` independent normalizations of `D` elements each,
// every group carrying its own slice of a [groups * D] weight. One
// threadgroup per group.
//
// This covers the hyper-connection hc_norm, whose four residual streams are
// normalized separately across a 10,240-wide weight, and a plain single-row
// norm is the same kernel with one group.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_grouped_centered(
    device const half*   x          [[buffer(0)]],   // [groups * D] FP16
    device const bfloat* weight     [[buffer(1)]],   // [groups * D] BF16
    device       half*   out        [[buffer(2)]],   // [groups * D] FP16
    constant     uint&   D          [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  group            [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    device const half*   xg = x      + group * DD;
    device const bfloat* wg = weight + group * DD;
    device       half*   og = out    + group * DD;
    const float inv = rms_block_inv(xg, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);
    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(xg[i]);
        float wv = 1.0f + float(wg[i]);
        og[i] = half(xv * inv * wv);
    }
}

// Per-head variant: one [headDim] weight shared across every head, as q_norm
// and k_norm use it.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_bf16w_perhead_centered(
    device const half*   x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device const bfloat* weight     [[buffer(1)]],   // [headDim] BF16, shared
    device       half*   out        [[buffer(2)]],   // [numHeads * headDim] FP16
    constant     uint&   headDim    [[buffer(3)]],
    constant     float&  eps        [[buffer(4)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(xh[i]);
        float wv = 1.0f + float(weight[i]);
        oh[i] = half(xv * inv * wv);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale_perhead(
    device const half*  x          [[buffer(0)]],   // [numHeads * headDim] FP16
    device       half*  out        [[buffer(1)]],   // [numHeads * headDim] FP16
    constant     uint&  headDim    [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  head             [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint HD = rms_fc_d(headDim);
    device const half* xh = x   + head * HD;
    device       half* oh = out + head * HD;
    const float inv = rms_block_inv(xh, HD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups, partial);
    for (uint i = lid; i < HD; i += lsize) {
        oh[i] = half(float(xh[i]) * inv);
    }
}

// Gemma 4 v_norm and the MoE router's internal norm are no-scale RMSNorm:
// y[i] = x[i] * rsqrt(mean(x^2) + eps). There is no resident weight tensor.
[[kernel, max_total_threads_per_threadgroup(256)]]
void rmsnorm_no_scale(
    device const half*  x          [[buffer(0)]],   // [D] FP16
    device       half*  out        [[buffer(1)]],   // [D] FP16
    constant     uint&  D          [[buffer(2)]],
    constant     float& eps        [[buffer(3)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[kRmsMaxSimdGroups];
    const uint DD = rms_fc_d(D);
    const float inv = rms_block_inv(x, DD, eps, lid, lsize,
                                    simd_lane_id, simd_group_id, simdgroups,
                                    partial);

    for (uint i = lid; i < DD; i += lsize) {
        float xv = float(x[i]);
        out[i] = half(xv * inv);
    }
}
