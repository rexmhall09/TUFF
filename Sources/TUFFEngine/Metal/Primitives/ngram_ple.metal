#include <metal_stdlib>
using namespace metal;

// ============================================================================
// ngram_ple — Qwen4-Exp's n-gram per-layer embedding.
//
// One decoder layer looks up a hashed n-gram of the recent tokens and folds it
// into the residual. The lookup itself happens on the host — see
// NgramTableReader, which reads rows out of a 29.80 GiB table too large to
// wrap in a Metal buffer. What follows is a gate between the looked-up keys
// and the layer's own queries, and a dilated depthwise convolution:
//
//   keys   = norm_key(key_proj(embedding))        // [S * D]
//   values = value_proj(embedding)                // [D]
//   query  = norm_query(hidden)                   // [S * D]
//   gate   = sum_d keys[s,d] * query[s,d] / sqrt(D)
//   gate   = sign(gate) * sqrt(max(|gate|, 1e-6))
//   gated  = sigmoid(gate[s]) * values[d]         // [S * D]
//   out    = gated + silu(depthwise_conv(norm_conv(gated)))
//
// The square root on the gate is not a normalization: it compresses the
// dot product's dynamic range while keeping its sign, so a strongly negative
// match stays negative rather than saturating the sigmoid.
// ============================================================================

static inline float ple_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

// One threadgroup per stream. Reduces the key/query dot product across the
// hidden dimension, applies the signed square root and the sigmoid, then
// writes `values` scaled by that gate into the stream's slice.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_gate_values_fp16(
    device const half*  keys    [[buffer(0)]],   // [S * D] FP16
    device const half*  queries [[buffer(1)]],   // [S * D] FP16
    device const half*  values  [[buffer(2)]],   // [D]     FP16
    device       half*  out     [[buffer(3)]],   // [S * D] FP16
    constant     uint&  D       [[buffer(4)]],
    constant     float& invSqrtD [[buffer(5)]],
    uint  stream           [[threadgroup_position_in_grid]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup float partial[8];
    threadgroup float gateShared;

    device const half* k = keys + stream * D;
    device const half* q = queries + stream * D;

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        acc = fma(float(k[i]), float(q[i]), acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) { partial[simd_group_id] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < simdgroups; ++i) { total += partial[i]; }
        float gate = total * invSqrtD;
        // Signed square root: keep the sign, compress the magnitude, and hold
        // the magnitude off zero so the derivative stays finite.
        const float magnitude = sqrt(max(fabs(gate), 1e-6f));
        gate = (gate < 0.0f) ? -magnitude : magnitude;
        gateShared = ple_sigmoid(gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float g = gateShared;
    device half* o = out + stream * D;
    for (uint i = lid; i < D; i += lsize) {
        o[i] = half(g * float(values[i]));
    }
}

// Causal depthwise convolution with dilation, one output row.
//
// `history` holds the (K-1)*dilation previous rows followed by the current
// one, so tap k reads row k * dilation and the last tap is the current token.
// The result is passed through silu, which is how the reference composes it.
[[kernel]]
void ple_depthwise_conv_silu_fp16(
    device const half*   history [[buffer(0)]],  // [(K-1)*dil + 1, W] FP16
    device const bfloat* weight  [[buffer(1)]],  // [W, K] BF16
    device       half*   out     [[buffer(2)]],  // [W] FP16
    constant     uint&   width   [[buffer(3)]],
    constant     uint&   taps    [[buffer(4)]],
    constant     uint&   dilation [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= width) return;
    float acc = 0.0f;
    for (uint k = 0; k < taps; ++k) {
        const uint row = k * dilation;
        acc = fma(float(weight[gid * taps + k]),
                  float(history[row * width + gid]), acc);
    }
    out[gid] = half(acc * ple_sigmoid(acc));
}

// Shift the convolution history down by one row and append `row`, so the next
// token sees the same window a contiguous sequence would.
[[kernel]]
void ple_conv_history_push_fp16(
    device       half*  history [[buffer(0)]],   // [len, W] FP16
    device const half*  row     [[buffer(1)]],   // [W] FP16
    constant     uint&  width   [[buffer(2)]],
    constant     uint&  length  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= width) return;
    for (uint r = 0; r + 1 < length; ++r) {
        history[r * width + gid] = history[(r + 1) * width + gid];
    }
    history[(length - 1) * width + gid] = row[gid];
}
