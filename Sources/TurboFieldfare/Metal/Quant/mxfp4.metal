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
