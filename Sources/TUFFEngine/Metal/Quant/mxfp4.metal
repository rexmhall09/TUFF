#include <metal_stdlib>
using namespace metal;

// GPT-OSS MXFP4 E2M1 GEMV. One SIMD group computes one row while eight SIMD
// groups share a threadgroup. Each lane owns one value from every 32-value
// block, so the UE8M0 scale is read once per lane and simd_sum completes the
// row reduction.

constant constexpr uint kMXFP4GroupSize = 32;

static inline float mxfp4_e2m1(uint code) {
    const uint magnitude = code & 7u;
    float value;
    switch (magnitude) {
    case 0u: value = 0.0f; break;
    case 1u: value = 0.5f; break;
    case 2u: value = 1.0f; break;
    case 3u: value = 1.5f; break;
    case 4u: value = 2.0f; break;
    case 5u: value = 3.0f; break;
    case 6u: value = 4.0f; break;
    default: value = 6.0f; break;
    }
    return (code & 8u) == 0u ? value : -value;
}

kernel void mxfp4_gemv_simd(
    device const uint8_t* weights [[buffer(0)]],
    device const uint8_t* scales  [[buffer(1)]],
    device const half* input      [[buffer(2)]],
    device half* output           [[buffer(3)]],
    constant uint& rows           [[buffer(4)]],
    constant uint& columns        [[buffer(5)]],
    device const bfloat* bias     [[buffer(6)]],
    constant uint& has_bias       [[buffer(7)]],
    uint threadgroupIndex         [[threadgroup_position_in_grid]],
    uint simdgroupIndex           [[simdgroup_index_in_threadgroup]],
    uint lane                     [[thread_index_in_simdgroup]])
{
    constexpr uint rowsPerThreadgroup = 8;
    const uint row = threadgroupIndex * rowsPerThreadgroup + simdgroupIndex;
    if (row >= rows) return;

    const uint groupsPerRow = columns / kMXFP4GroupSize;
    device const uint8_t* rowWeights = weights + row * (columns / 2u);
    device const uint8_t* rowScales = scales + row * groupsPerRow;
    float sum = 0.0f;
    for (uint group = 0; group < groupsPerRow; ++group) {
        const uint byteIndex = group * (kMXFP4GroupSize / 2u) + (lane >> 1u);
        const uint packed = uint(rowWeights[byteIndex]);
        const uint code = (lane & 1u) == 0u ? packed & 0x0Fu : packed >> 4u;
        // Production GPT-OSS exponent bytes are finite nonzero UE8M0 values.
        // Reinterpreting them as the FP32 exponent is exactly 2^(e - 127),
        // matching the official Metal reference without an approximation.
        const float scale = as_type<float>(uint(rowScales[group]) << 23u);
        const float weight = mxfp4_e2m1(code) * scale;
        sum = fma(weight, float(input[group * kMXFP4GroupSize + lane]), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) {
        output[row] = half(sum + (has_bias != 0u ? float(bias[row]) : 0.0f));
    }
}

// GPT-OSS leaves embeddings, attention projections, routers, and the output
// head in BF16. One SIMD group reduces one output row while eight groups share
// a threadgroup. Activations remain FP16, matching the rest of the runtime.
static inline float bf16_gemv_row(
    device const bfloat* weights,
    device const half* input,
    uint row,
    uint columns,
    uint lane
) {
    device const bfloat* rowWeights = weights + row * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32u) {
        sum = fma(float(rowWeights[column]), float(input[column]), sum);
    }
    return simd_sum(sum);
}

kernel void bf16_gemv_half_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint threadgroupIndex [[threadgroup_position_in_grid]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rowsPerThreadgroup = 8;
    const uint row = threadgroupIndex * rowsPerThreadgroup + simdgroupIndex;
    if (row >= rows) return;
    const float sum = bf16_gemv_row(weights, input, row, columns, lane);
    if (lane == 0u) {
        output[row] = half(sum + (has_bias != 0u ? float(bias[row]) : 0.0f));
    }
}

kernel void bf16_gemv_float_simd(
    device const bfloat* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const bfloat* bias [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    uint threadgroupIndex [[threadgroup_position_in_grid]],
    uint simdgroupIndex [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rowsPerThreadgroup = 8;
    const uint row = threadgroupIndex * rowsPerThreadgroup + simdgroupIndex;
    if (row >= rows) return;
    const float sum = bf16_gemv_row(weights, input, row, columns, lane);
    if (lane == 0u) {
        output[row] = sum + (has_bias != 0u ? float(bias[row]) : 0.0f);
    }
}

// Batched BF16 output-head argmax. The x grid partitions vocabulary rows and
// the y grid partitions candidate hidden rows. This keeps the target verifier
// to one head dispatch plus one reduction dispatch, rather than one full
// vocabulary GEMV and argmax pair per candidate row.
constant constexpr uint kBF16ArgmaxRowsPerTG = 8;
constant constexpr uint kBF16ArgmaxSummaryStride = 2;
constant constexpr uint kBF16ArgmaxMaxSimdGroups = 8;

[[kernel, max_total_threads_per_threadgroup(256)]]
void bf16_gemv_argmax_rows(
    device const bfloat* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device float* summaries [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& has_bias [[buffer(6)]],
    constant uint& input_stride [[buffer(7)]],
    constant uint& batch_count [[buffer(8)]],
    uint3 tg_idx [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float partial_value[kBF16ArgmaxRowsPerTG];
    threadgroup uint partial_index[kBF16ArgmaxRowsPerTG];

    const uint batch = tg_idx.y;
    const uint row = tg_idx.x * kBF16ArgmaxRowsPerTG + simd_group_id;
    float best_value = -INFINITY;
    uint best_index = 0xFFFFFFFFu;
    if (batch < batch_count && row < rows) {
        device const half* row_input = input + batch * input_stride;
        const float sum = bf16_gemv_row(weights, row_input, row,
                                        columns, simd_lane_id);
        // `encodeHalf` stores this intermediate as FP16 before the existing
        // argmax reads it. Round here as well so close logits do not change
        // greedy output when this batched path replaces the scalar path.
        const half rounded = half(sum +
            (has_bias != 0u ? float(bias[row]) : 0.0f));
        const float value = float(rounded);
        if (simd_lane_id == 0u && isfinite(value)) {
            best_value = value;
            best_index = row;
        }
    }

    if (simd_lane_id == 0u) {
        partial_value[simd_group_id] = best_value;
        partial_index[simd_group_id] = best_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0u) {
        const bool active = simd_lane_id < simdgroups;
        const float value = active ? partial_value[simd_lane_id] : -INFINITY;
        const uint index = active ? partial_index[simd_lane_id] : 0xFFFFFFFFu;
        const float all_value = simd_max(value);
        const uint all_index = (value == all_value)
            ? index : 0xFFFFFFFFu;
        const uint all_index_min = simd_min(all_index);
        if (simd_lane_id == 0u) {
            device float* slot = summaries
                + (batch * ((rows + kBF16ArgmaxRowsPerTG - 1u)
                            / kBF16ArgmaxRowsPerTG)
                   + tg_idx.x) * kBF16ArgmaxSummaryStride;
            slot[0] = all_value;
            slot[1] = as_type<float>(all_index_min);
        }
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void bf16_gemv_argmax_rows_reduce(
    device const float* summaries [[buffer(0)]],
    device uint* out_tokens [[buffer(1)]],
    constant uint& row_groups [[buffer(2)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]) {
    threadgroup float partial_value[kBF16ArgmaxMaxSimdGroups];
    threadgroup uint partial_index[kBF16ArgmaxMaxSimdGroups];
    const device float* base = summaries
        + batch * row_groups * kBF16ArgmaxSummaryStride;

    float best_value = -INFINITY;
    uint best_index = 0xFFFFFFFFu;
    for (uint i = lid; i < row_groups; i += lsize) {
        const device float* slot = base + i * kBF16ArgmaxSummaryStride;
        const float value = slot[0];
        const uint index = as_type<uint>(slot[1]);
        if (value > best_value
            || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }

    const float simd_value = simd_max(best_value);
    const uint simd_index = (best_value == simd_value)
        ? best_index : 0xFFFFFFFFu;
    const uint simd_index_min = simd_min(simd_index);
    if (simd_lane_id == 0u) {
        partial_value[simd_group_id] = simd_value;
        partial_index[simd_group_id] = simd_index_min;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0u) {
        const bool active = simd_lane_id < simdgroups;
        const float value = active ? partial_value[simd_lane_id] : -INFINITY;
        const uint index = active ? partial_index[simd_lane_id] : 0xFFFFFFFFu;
        const float all_value = simd_max(value);
        const uint all_index = (value == all_value)
            ? index : 0xFFFFFFFFu;
        const uint all_index_min = simd_min(all_index);
        if (simd_lane_id == 0u) {
            out_tokens[batch] = all_index_min;
        }
    }
}

kernel void bf16_embedding_lookup_half(
    device const bfloat* table [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= hidden_size) return;
    output[gid] = half(table[token * hidden_size + gid]);
}

kernel void bf16_embedding_lookup_float(
    device const bfloat* table [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= hidden_size) return;
    output[gid] = float(table[token * hidden_size + gid]);
}
