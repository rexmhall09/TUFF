#include <metal_stdlib>
using namespace metal;

// ============================================================================
// hyper_connection — Qwen4-Exp's multi-stream residual.
//
// Where every other architecture in this runtime carries one residual stream,
// qwen4_exp carries `G` of them (four in the pinned checkpoint) and has no
// input_layernorm or post_attention_layernorm at all. Each block reads a
// low-rank mixture of the streams and writes its output back into all of them
// through a learned per-stream gate:
//
//   normed = hc_norm(hyper)                      // grouped, centered; rmsnorm
//   t      = silu(W_down @ normed / G)           // [lowRank]     hc_lowrank_silu
//   mix    = sigmoid(W_up @ t)                   // [G * D]
//   mixed  = mean_g(mix[g] * normed[g])          // [D]           hc_combine
//   inj    = 2 * sigmoid(W_inject @ normed / G)  // [G]
//   hyper[g] += branch * inj[g]                  // [G * D]       hc_inject
//
// The three projections are ordinary INT4 GEMVs and go through the existing
// dequant path. What is here is the arithmetic between them.
//
// `hyper` stays unnormalized across the whole layer: `normed` feeds the mix
// and the injection gate, never the residual itself.
// ============================================================================

static inline float hc_sigmoid(float v) {
    return 1.0f / (1.0f + exp(-v));
}

// t[i] = silu(x[i] * scale), with scale = 1/G.
//
// The division by the stream count happens before the activation, not after,
// which matters: silu is not homogeneous.
[[kernel]]
void hc_lowrank_silu_fp16(
    device const half*  x      [[buffer(0)]],   // [lowRank] FP16
    device       half*  out    [[buffer(1)]],   // [lowRank] FP16
    constant     uint&  count  [[buffer(2)]],
    constant     float& scale  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float v = float(x[gid]) * scale;
    out[gid] = half(v * hc_sigmoid(v));
}

// mixed[i] = (1/G) * sum_g sigmoid(up[g * D + i]) * normed[g * D + i]
//
// The sigmoid is folded in here rather than run as its own pass: the mix is
// G * D wide and read exactly once.
[[kernel]]
void hc_combine_fp16(
    device const half*  up      [[buffer(0)]],   // [G * D] FP16, pre-sigmoid
    device const half*  normed  [[buffer(1)]],   // [G * D] FP16
    device       half*  mixed   [[buffer(2)]],   // [D]     FP16
    constant     uint&  d       [[buffer(3)]],
    constant     uint&  groups  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const uint index = g * d + gid;
        acc += hc_sigmoid(float(up[index])) * float(normed[index]);
    }
    mixed[gid] = half(acc / float(groups));
}

// hyper[g * D + i] += branch[i] * 2 * sigmoid(injRaw[g] * scale)
//
// One thread per element of the stacked residual. The per-stream gate is
// recomputed per thread rather than staged through threadgroup memory: it is
// `groups` values wide, and at G = 4 the arithmetic is cheaper than the
// barrier would be.
[[kernel]]
void hc_inject_fp16(
    device       half*  hyper   [[buffer(0)]],   // [G * D] FP16, accumulated
    device const half*  branch  [[buffer(1)]],   // [D]     FP16
    device const half*  injRaw  [[buffer(2)]],   // [G]     FP16, pre-sigmoid
    constant     uint&  d       [[buffer(3)]],
    constant     uint&  groups  [[buffer(4)]],
    constant     float& scale   [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = d * groups;
    if (gid >= total) return;
    const uint g = gid / d;
    const uint i = gid - g * d;
    const float gate = 2.0f * hc_sigmoid(float(injRaw[g]) * scale);
    hyper[gid] = half(float(hyper[gid]) + float(branch[i]) * gate);
}
