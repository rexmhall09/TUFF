#include <metal_stdlib>
using namespace metal;

constant constexpr uint kPLEMaxSimdGroups = 8;

// Extract one token-major packed PLE slice, normalize its context component,
// and combine it with the already-scaled token-identity embedding.
[[kernel, max_total_threads_per_threadgroup(256)]]
void ple_prepare_layer_fp16(
    device const half* identity       [[buffer(0)]],
    device const half* context        [[buffer(1)]],
    device const bfloat* norm_weight  [[buffer(2)]],
    device half* out                  [[buffer(3)]],
    constant uint& rows               [[buffer(4)]],
    constant uint& packed_width       [[buffer(5)]],
    constant uint& ple_width          [[buffer(6)]],
    constant uint& layer              [[buffer(7)]],
    constant float& context_scale     [[buffer(8)]],
    constant float& combined_scale    [[buffer(9)]],
    constant float& eps               [[buffer(10)]],
    uint row                           [[threadgroup_position_in_grid]],
    uint lid                           [[thread_position_in_threadgroup]],
    uint lsize                         [[threads_per_threadgroup]],
    uint lane                          [[thread_index_in_simdgroup]],
    uint simd_group                    [[simdgroup_index_in_threadgroup]],
    uint simd_groups                   [[simdgroups_per_threadgroup]]
) {
    if (row >= rows) return;
    threadgroup float partial[kPLEMaxSimdGroups];
    const uint base = row * packed_width + layer * ple_width;
    float sum = 0.0f;
    for (uint i = lid; i < ple_width; i += lsize) {
        const float value = float(context[base + i]) * context_scale;
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float value = lane < simd_groups ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0) partial[0] = rsqrt(value / float(ple_width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partial[0];
    for (uint i = lid; i < ple_width; i += lsize) {
        const float projected = float(context[base + i]) * context_scale;
        const float normalized = projected * inv * float(norm_weight[i]);
        out[row * ple_width + i] = half(
            (normalized + float(identity[base + i])) * combined_scale);
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_scale_fp16(
    device half* hidden              [[buffer(0)]],
    device const half* delta         [[buffer(1)]],
    constant uint& count             [[buffer(2)]],
    constant float& scale            [[buffer(3)]],
    uint tid                          [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half((float(hidden[tid]) + float(delta[tid])) * scale);
}
