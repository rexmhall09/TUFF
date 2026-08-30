#include <metal_stdlib>
using namespace metal;

constant constexpr uint kVisionHeadDim = 72;
constant constexpr uint kVisionPaddedHeadDim = 80;
constant constexpr uint kVisionMaxQueryTile = 16;
constant constexpr uint kVisionKeyTile = 32;
constant constexpr uint kVisionThreads = 128;

kernel void vision_pad_heads_72_to_80(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* paddedQ [[buffer(3)]],
    device bfloat* paddedK [[buffer(4)]],
    device bfloat* paddedV [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * heads * 80u) return;
    const uint dimension = index % 80u;
    const uint vector = index / 80u;
    const uint row = vector % rows;
    const uint head = vector / rows;
    const uint source = (row * heads + head) * 72u + dimension;
    paddedQ[index] = dimension < 72u ? q[source] : bfloat(0.0f);
    paddedK[index] = dimension < 72u ? k[source] : bfloat(0.0f);
    paddedV[index] = dimension < 72u ? v[source] : bfloat(0.0f);
}

kernel void vision_unpad_heads_80_to_72(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * heads * 72u) return;
    const uint dimension = index % 72u;
    const uint vector = index / 72u;
    const uint row = vector / heads;
    const uint head = vector % heads;
    output[index] = input[(head * rows + row) * 80u + dimension];
}

template <uint LogicalDim, uint ComputeDim, uint QueryTile>
inline void vision_attention_online_impl(
    device const bfloat* q,
    device const bfloat* k,
    device const bfloat* v,
    device bfloat* output,
    constant uint& sequenceLength,
    constant uint& numHeads,
    constant float& scoreScale,
    uint2 group,
    uint tid,
    threadgroup bfloat* qTile,
    threadgroup bfloat* kTile,
    threadgroup bfloat* vTile,
    threadgroup float* probabilities,
    threadgroup float* accumulator,
    threadgroup float* oldScale,
    threadgroup float* rowSum,
    threadgroup float* rowMax) {

    const uint head = group.y;
    const uint queryBase = group.x * QueryTile;
    if (head >= numHeads) return;

    for (uint index = tid; index < QueryTile * LogicalDim;
         index += kVisionThreads) {
        const uint localQuery = index / LogicalDim;
        const uint dimension = index % LogicalDim;
        const uint query = queryBase + localQuery;
        qTile[index] = query < sequenceLength
            ? q[(query * numHeads + head) * LogicalDim + dimension]
            : bfloat(0.0f);
    }
    for (uint index = tid; index < QueryTile * ComputeDim;
         index += kVisionThreads) {
        accumulator[index] = 0.0f;
    }
    if (tid < QueryTile) {
        rowMax[tid] = -INFINITY;
        rowSum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint keyBase = 0; keyBase < sequenceLength; keyBase += kVisionKeyTile) {
        for (uint index = tid; index < kVisionKeyTile * LogicalDim;
             index += kVisionThreads) {
            const uint localKey = index / LogicalDim;
            const uint dimension = index % LogicalDim;
            const uint keyIndex = keyBase + localKey;
            const uint source = (keyIndex * numHeads + head) * LogicalDim + dimension;
            kTile[index] = keyIndex < sequenceLength ? k[source] : bfloat(0.0f);
            vTile[index] = keyIndex < sequenceLength ? v[source] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pair = tid; pair < QueryTile * kVisionKeyTile;
             pair += kVisionThreads) {
            const uint localQuery = pair / kVisionKeyTile;
            const uint localKey = pair % kVisionKeyTile;
            const uint query = queryBase + localQuery;
            const uint keyIndex = keyBase + localKey;
            float score = -INFINITY;
            if (query < sequenceLength && keyIndex < sequenceLength) {
                score = 0.0f;
                for (uint dimension = 0; dimension < ComputeDim; ++dimension) {
                    const float qValue = dimension < LogicalDim
                        ? float(qTile[localQuery * LogicalDim + dimension]) : 0.0f;
                    const float kValue = dimension < LogicalDim
                        ? float(kTile[localKey * LogicalDim + dimension]) : 0.0f;
                    score = fma(qValue, kValue, score);
                }
            }
            probabilities[pair] = score * scoreScale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < QueryTile) {
            float tileMax = -INFINITY;
            for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                tileMax = max(tileMax,
                              probabilities[tid * kVisionKeyTile + localKey]);
            }
            const float nextMax = max(rowMax[tid], tileMax);
            const float scale = rowMax[tid] == -INFINITY
                ? 0.0f : exp(rowMax[tid] - nextMax);
            float tileSum = 0.0f;
            for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                const uint keyIndex = keyBase + localKey;
                const float probability = keyIndex < sequenceLength
                    ? exp(probabilities[tid * kVisionKeyTile + localKey] - nextMax)
                    : 0.0f;
                probabilities[tid * kVisionKeyTile + localKey] = probability;
                tileSum += probability;
            }
            oldScale[tid] = scale;
            rowSum[tid] = rowSum[tid] * scale + tileSum;
            rowMax[tid] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = tid; index < QueryTile * ComputeDim;
             index += kVisionThreads) {
            const uint localQuery = index / ComputeDim;
            const uint dimension = index % ComputeDim;
            float value = accumulator[index] * oldScale[localQuery];
            if (dimension < LogicalDim) {
                for (uint localKey = 0; localKey < kVisionKeyTile; ++localKey) {
                    value = fma(
                        probabilities[localQuery * kVisionKeyTile + localKey],
                        float(vTile[localKey * LogicalDim + dimension]), value);
                }
            }
            accumulator[index] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = tid; index < QueryTile * LogicalDim;
         index += kVisionThreads) {
        const uint localQuery = index / LogicalDim;
        const uint dimension = index % LogicalDim;
        const uint query = queryBase + localQuery;
        if (query < sequenceLength) {
            output[(query * numHeads + head) * LogicalDim + dimension] =
                bfloat(accumulator[localQuery * ComputeDim + dimension]
                       / rowSum[localQuery]);
        }
    }
}

kernel void vision_attention_online_72(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant float& scoreScale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionHeadDim, kVisionHeadDim, 8>(
        q, k, v, output, sequenceLength, numHeads, scoreScale,
        group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_attention_online_64(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant float& scoreScale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<64, 64, 8>(
        q, k, v, output, sequenceLength, numHeads, scoreScale,
        group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_attention_online_80(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant float& scoreScale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionHeadDim, kVisionPaddedHeadDim, 8>(
        q, k, v, output, sequenceLength, numHeads, scoreScale,
        group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_attention_online_72_q16(
    device const bfloat* q [[buffer(0)]],
    device const bfloat* k [[buffer(1)]],
    device const bfloat* v [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant float& scoreScale [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threadPosition [[thread_position_in_threadgroup]]) {
    threadgroup bfloat qTile[kVisionMaxQueryTile * kVisionHeadDim];
    threadgroup bfloat kTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup bfloat vTile[kVisionKeyTile * kVisionHeadDim];
    threadgroup float probabilities[kVisionMaxQueryTile * kVisionKeyTile];
    threadgroup float accumulator[kVisionMaxQueryTile * kVisionPaddedHeadDim];
    threadgroup float oldScale[kVisionMaxQueryTile];
    threadgroup float rowSum[kVisionMaxQueryTile];
    threadgroup float rowMax[kVisionMaxQueryTile];
    vision_attention_online_impl<kVisionHeadDim, kVisionHeadDim, 16>(
        q, k, v, output, sequenceLength, numHeads, scoreScale,
        group.xy, threadPosition.x,
        qTile, kTile, vTile, probabilities, accumulator,
        oldScale, rowSum, rowMax);
}

kernel void vision_normalize_patches(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& realRows [[buffer(2)]],
    constant uint& paddedRows [[buffer(3)]],
    constant uint& patchDimension [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = paddedRows * patchDimension;
    if (index >= count) return;
    output[index] = index < realRows * patchDimension
        ? bfloat(2.0f * (float(input[index]) - 0.5f))
        : bfloat(0.0f);
}

kernel void vision_add_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& positionSize [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint dimension = index % width;
    const int2 position = positions[row];
    const uint xIndex = uint(position.x) * width + dimension;
    const uint yIndex = (positionSize + uint(position.y)) * width + dimension;
    hidden[index] = bfloat(float(hidden[index])
                           + float(table[xIndex]) + float(table[yIndex]));
}

inline float vision_simdgroup_sum(float value,
                                  uint lane,
                                  uint simdgroup,
                                  threadgroup float* partials) {
    const float subtotal = simd_sum(value);
    if (lane == 0) partials[simdgroup] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        float merged = lane < 8u ? partials[lane] : 0.0f;
        merged = simd_sum(merged);
        if (lane == 0) partials[0] = merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

kernel void vision_rmsnorm_rows(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(input[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        if (dimension < width) {
            const bfloat normalized = bfloat(values[index] * inverseRMS[0]);
            output[row * width + dimension] = bfloat(
                float(weight[dimension]) * float(normalized));
        }
    }
}

kernel void vision_postnorm_residual(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float values[4];
    float sum = 0.0f;
    const uint base = tid * 4u;
    for (uint index = 0; index < 4u; ++index) {
        const uint dimension = base + index;
        values[index] = dimension < width
            ? float(branch[row * width + dimension]) : 0.0f;
        sum += values[index] * values[index];
    }
    sum = simd_sum(sum);
    if (simdgroup == 0) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        sum = simd_sum(partials[lane]);
        if (lane == 0) {
            inverseRMS[0] = metal::precise::rsqrt(sum / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint element = 0; element < 4u; ++element) {
        const uint dimension = base + element;
        if (dimension < width) {
            const uint index = row * width + dimension;
            const bfloat scaled = bfloat(values[element] * inverseRMS[0]);
            const bfloat normalized = bfloat(
                float(weight[dimension]) * float(scaled));
            output[index] = bfloat(float(residual[index]) + float(normalized));
        }
    }
}

inline float2 vision_rope_pair(float2 value, int position, uint pairInAxis) {
    const float exponent = (2.0f / 36.0f) * float(pairInAxis);
    const float angle = float(position) / powr(100.0f, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    return float2(value.x * cosine - value.y * sine,
                  value.y * cosine + value.x * sine);
}

kernel void vision_qkv_norm_rope(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint base = vectorIndex * kVisionHeadDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / 72.0f + 1e-6f);
    const float kInverse = rsqrt(kSquares / 72.0f + 1e-6f);
    const float vInverse = rsqrt(vSquares / 72.0f + 1e-6f);
    const int2 position = positions[vectorIndex / heads];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * 36u;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < 18u; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + 18u;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + 18u]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + 18u]));
            qPair = vision_rope_pair(qPair, coordinate, pair);
            kPair = vision_rope_pair(kPair, coordinate, pair);
            q[first] = bfloat(qPair.x);
            q[second] = bfloat(qPair.y);
            k[first] = bfloat(kPair.x);
            k[second] = bfloat(kPair.y);
        }
    }
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

kernel void vision_qkv_norm_rope_64(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    constexpr uint headDim = 64u;
    constexpr uint axisDim = 32u;
    constexpr uint pairsPerAxis = 16u;
    const uint base = vectorIndex * headDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < headDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / float(headDim) + 1e-6f);
    const float kInverse = rsqrt(kSquares / float(headDim) + 1e-6f);
    const float vInverse = rsqrt(vSquares / float(headDim) + 1e-6f);
    const int2 position = positions[vectorIndex / heads];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * axisDim;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < pairsPerAxis; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + pairsPerAxis;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + pairsPerAxis]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + pairsPerAxis]));
            const float exponent = (2.0f / float(axisDim)) * float(pair);
            const float angle = float(coordinate) / powr(100.0f, exponent);
            const float cosine = cos(angle);
            const float sine = sin(angle);
            q[first] = bfloat(qPair.x * cosine - qPair.y * sine);
            q[second] = bfloat(qPair.y * cosine + qPair.x * sine);
            k[first] = bfloat(kPair.x * cosine - kPair.y * sine);
            k[second] = bfloat(kPair.y * cosine + kPair.x * sine);
        }
    }
    for (uint dimension = 0; dimension < headDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

/// Same arithmetic as `vision_qkv_norm_rope` on the head-major padded layout the
/// QKV projections now write: vector `(head, row)` starts at
/// `(head * rowStride + row) * 80`. Threads walk rows within a head so the
/// strided-by-80 accesses stay coalesced.
kernel void vision_qkv_norm_rope_padded(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qWeight [[buffer(3)]],
    device const bfloat* kWeight [[buffer(4)]],
    device const int2* positions [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& rowStride [[buffer(8)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint head = vectorIndex / rows;
    const uint row = vectorIndex % rows;
    const uint base = (head * rowStride + row) * kVisionPaddedHeadDim;
    float qSquares = 0.0f;
    float kSquares = 0.0f;
    float vSquares = 0.0f;
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        const float qValue = float(q[base + dimension]);
        const float kValue = float(k[base + dimension]);
        const float vValue = float(v[base + dimension]);
        qSquares = fma(qValue, qValue, qSquares);
        kSquares = fma(kValue, kValue, kSquares);
        vSquares = fma(vValue, vValue, vSquares);
    }
    const float qInverse = rsqrt(qSquares / 72.0f + 1e-6f);
    const float kInverse = rsqrt(kSquares / 72.0f + 1e-6f);
    const float vInverse = rsqrt(vSquares / 72.0f + 1e-6f);
    const int2 position = positions[row];
    for (uint axis = 0; axis < 2u; ++axis) {
        const uint axisBase = axis * 36u;
        const int coordinate = axis == 0u ? position.x : position.y;
        for (uint pair = 0; pair < 18u; ++pair) {
            const uint first = base + axisBase + pair;
            const uint second = first + 18u;
            float2 qPair = float2(float(q[first]) * qInverse * float(qWeight[axisBase + pair]),
                                  float(q[second]) * qInverse * float(qWeight[axisBase + pair + 18u]));
            float2 kPair = float2(float(k[first]) * kInverse * float(kWeight[axisBase + pair]),
                                  float(k[second]) * kInverse * float(kWeight[axisBase + pair + 18u]));
            qPair = vision_rope_pair(qPair, coordinate, pair);
            kPair = vision_rope_pair(kPair, coordinate, pair);
            q[first] = bfloat(qPair.x);
            q[second] = bfloat(qPair.y);
            k[first] = bfloat(kPair.x);
            k[second] = bfloat(kPair.y);
        }
    }
    for (uint dimension = 0; dimension < kVisionHeadDim; ++dimension) {
        v[base + dimension] = bfloat(float(v[base + dimension]) * vInverse);
    }
}

kernel void vision_geglu(
    device bfloat* gate [[buffer(0)]],
    device const bfloat* up [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(gate[index]);
    const float inner = clamp(
        0.7978845608f * (x + 0.044715f * x * x * x), -20.0f, 20.0f);
    const float gelu = 0.5f * x * (1.0f + tanh(inner));
    gate[index] = bfloat(gelu * float(up[index]));
}

kernel void vision_pool_standardize_norm(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* stdBias [[buffer(1)]],
    device const bfloat* stdScale [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& patchWidth [[buffer(4)]],
    constant uint& patchHeight [[buffer(5)]],
    uint outputRow [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint outputWidth = patchWidth / 3u;
    const uint outputHeight = patchHeight / 3u;
    if (outputRow >= outputWidth * outputHeight) return;
    const uint outputX = outputRow % outputWidth;
    const uint outputY = outputRow / outputWidth;
    threadgroup float partials[8];
    float squares = 0.0f;
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        float pooled = 0.0f;
        for (uint dy = 0; dy < 3u; ++dy) {
            for (uint dx = 0; dx < 3u; ++dx) {
                const uint inputRow = (outputY * 3u + dy) * patchWidth
                    + outputX * 3u + dx;
                pooled += float(input[inputRow * 1152u + dimension]);
            }
        }
        pooled *= sqrt(1152.0f) / 9.0f;
        const float standardized = (pooled - float(stdBias[dimension]))
            * float(stdScale[dimension]);
        output[outputRow * 1152u + dimension] = bfloat(standardized);
        squares = fma(standardized, standardized, squares);
    }
    const float total = vision_simdgroup_sum(squares, lane, simdgroup, partials);
    const float inverse = rsqrt(total / 1152.0f + 1e-6f);
    for (uint dimension = tid; dimension < 1152u; dimension += 256u) {
        const uint index = outputRow * 1152u + dimension;
        output[index] = bfloat(float(output[index]) * inverse);
    }
}

kernel void vision_pool_norm(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& patchWidth [[buffer(2)]],
    constant uint& patchHeight [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& poolSize [[buffer(5)]],
    uint outputRow [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint outputWidth = patchWidth / poolSize;
    const uint outputHeight = patchHeight / poolSize;
    if (outputRow >= outputWidth * outputHeight) return;
    const uint outputX = outputRow % outputWidth;
    const uint outputY = outputRow / outputWidth;
    threadgroup float partials[8];
    float squares = 0.0f;
    const float factor = sqrt(float(width)) / float(poolSize * poolSize);
    for (uint dimension = tid; dimension < width; dimension += 256u) {
        float pooled = 0.0f;
        for (uint dy = 0; dy < poolSize; ++dy) {
            for (uint dx = 0; dx < poolSize; ++dx) {
                const uint inputRow = (outputY * poolSize + dy) * patchWidth
                    + outputX * poolSize + dx;
                pooled += float(input[inputRow * width + dimension]);
            }
        }
        pooled *= factor;
        output[outputRow * width + dimension] = bfloat(pooled);
        squares = fma(pooled, pooled, squares);
    }
    const float total = vision_simdgroup_sum(squares, lane, simdgroup, partials);
    const float inverse = rsqrt(total / float(width) + 1e-6f);
    for (uint dimension = tid; dimension < width; dimension += 256u) {
        const uint index = outputRow * width + dimension;
        output[index] = bfloat(float(output[index]) * inverse);
    }
}

kernel void vision_bfloat_to_half(
    device const bfloat* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = half(input[index]);
}

kernel void vision_half_to_bfloat(
    device const half* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = bfloat(input[index]);
}

kernel void vision_unified_add_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * width) return;
    const uint row = index / width;
    const uint dimension = index % width;
    const int2 position = positions[row];
    const uint xIndex = (uint(position.x) * 2u) * width + dimension;
    const uint yIndex = (uint(position.y) * 2u + 1u) * width + dimension;
    hidden[index] = bfloat(float(hidden[index])
                           + float(table[xIndex]) + float(table[yIndex]));
}

kernel void vision_rmsnorm_no_scale_rows(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float inverseRMS[1];
    float sum = 0.0f;
    for (uint d = tid; d < width; d += 288u) {
        const float value = float(input[row * width + d]);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (simdgroup == 0u) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const float total = simd_sum(partials[lane]);
        if (lane == 0u) inverseRMS[0] = rsqrt(total / float(width) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < width; d += 288u) {
        output[row * width + d] = bfloat(float(input[row * width + d]) * inverseRMS[0]);
    }
}

// MARK: - Qwen3.6 vision

kernel void vision_qwen_copy_padded(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& realRows [[buffer(2)]],
    constant uint& paddedRows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= paddedRows * width) return;
    output[index] = index < realRows * width ? input[index] : bfloat(0.0f);
}

kernel void vision_qwen_add_interpolated_position(
    device bfloat* hidden [[buffer(0)]],
    device const bfloat* table [[buffer(1)]],
    device const int2* positions [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& gridHeight [[buffer(4)]],
    constant uint& gridWidth [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * 1152u) return;
    const uint row = index / 1152u;
    const uint dimension = index % 1152u;
    const int2 position = positions[row]; // (height, width)
    const float sourceY = gridHeight > 1u
        ? float(position.x) * 47.0f / float(gridHeight - 1u) : 0.0f;
    const float sourceX = gridWidth > 1u
        ? float(position.y) * 47.0f / float(gridWidth - 1u) : 0.0f;
    const uint y0 = uint(floor(sourceY));
    const uint x0 = uint(floor(sourceX));
    const uint y1 = min(y0 + 1u, 47u);
    const uint x1 = min(x0 + 1u, 47u);
    const float fy = sourceY - float(y0);
    const float fx = sourceX - float(x0);
    const float p00 = float(table[(y0 * 48u + x0) * 1152u + dimension]);
    const float p01 = float(table[(y0 * 48u + x1) * 1152u + dimension]);
    const float p10 = float(table[(y1 * 48u + x0) * 1152u + dimension]);
    const float p11 = float(table[(y1 * 48u + x1) * 1152u + dimension]);
    const float top = mix(p00, p01, fx);
    const float bottom = mix(p10, p11, fx);
    hidden[index] = bfloat(float(hidden[index]) + mix(top, bottom, fy));
}

kernel void vision_qwen_layernorm_rows(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    if (row >= rows) return;
    threadgroup float partials[32];
    threadgroup float sharedMean[1];
    threadgroup float sharedInverse[1];
    float sum = 0.0f;
    for (uint d = tid; d < width; d += 288u) {
        sum += float(input[row * width + d]);
    }
    sum = simd_sum(sum);
    if (simdgroup == 0u) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const float total = simd_sum(partials[lane]);
        if (lane == 0u) sharedMean[0] = total / float(width);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float squares = 0.0f;
    for (uint d = tid; d < width; d += 288u) {
        const float centered = float(input[row * width + d]) - sharedMean[0];
        squares = fma(centered, centered, squares);
    }
    squares = simd_sum(squares);
    if (simdgroup == 0u) partials[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) partials[simdgroup] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        const float total = simd_sum(partials[lane]);
        if (lane == 0u) {
            sharedInverse[0] = metal::precise::rsqrt(total / float(width) + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < width; d += 288u) {
        const float normalized = (float(input[row * width + d]) - sharedMean[0])
            * sharedInverse[0];
        output[row * width + d] = bfloat(
            normalized * float(weight[d]) + float(bias[d]));
    }
}

inline float2 vision_qwen_rope_pair(float2 value, int position, uint pair) {
    const float exponent = (2.0f / 36.0f) * float(pair);
    const float angle = float(position) / powr(10000.0f, exponent);
    const float c = cos(angle);
    const float s = sin(angle);
    return float2(value.x * c - value.y * s,
                  value.y * c + value.x * s);
}

kernel void vision_qwen_qkv_bias_rope(
    device bfloat* q [[buffer(0)]],
    device bfloat* k [[buffer(1)]],
    device bfloat* v [[buffer(2)]],
    device const bfloat* qBias [[buffer(3)]],
    device const bfloat* kBias [[buffer(4)]],
    device const bfloat* vBias [[buffer(5)]],
    device const int2* positions [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    uint vectorIndex [[thread_position_in_grid]]) {
    if (vectorIndex >= rows * heads) return;
    const uint head = vectorIndex % heads;
    const uint row = vectorIndex / heads;
    const uint base = vectorIndex * 72u;
    const uint biasBase = head * 72u;
    const int2 position = positions[row];
    for (uint pair = 0; pair < 36u; ++pair) {
        const uint first = base + pair;
        const uint second = first + 36u;
        float2 qPair = float2(
            float(q[first]) + float(qBias[biasBase + pair]),
            float(q[second]) + float(qBias[biasBase + pair + 36u]));
        float2 kPair = float2(
            float(k[first]) + float(kBias[biasBase + pair]),
            float(k[second]) + float(kBias[biasBase + pair + 36u]));
        const int coordinate = pair < 18u ? position.x : position.y;
        qPair = vision_qwen_rope_pair(qPair, coordinate, pair % 18u);
        kPair = vision_qwen_rope_pair(kPair, coordinate, pair % 18u);
        q[first] = bfloat(qPair.x);
        q[second] = bfloat(qPair.y);
        k[first] = bfloat(kPair.x);
        k[second] = bfloat(kPair.y);
    }
    for (uint d = 0; d < 72u; ++d) {
        const uint index = base + d;
        v[index] = bfloat(float(v[index]) + float(vBias[biasBase + d]));
    }
}

kernel void vision_qwen_residual_bias(
    device const bfloat* residual [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = bfloat(float(residual[index]) + float(branch[index])
                           + float(bias[index % width]));
}

kernel void vision_qwen_add_bias(
    device bfloat* values [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        values[index] = bfloat(float(values[index]) + float(bias[index % width]));
    }
}

kernel void vision_qwen_gelu_tanh_bias(
    device bfloat* values [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(values[index]) + float(bias[index % width]);
    const float inner = clamp(
        0.7978845608f * (x + 0.044715f * x * x * x), -20.0f, 20.0f);
    values[index] = bfloat(0.5f * x * (1.0f + tanh(inner)));
}

// MSL does not expose erf on every supported compiler. This Abramowitz-Stegun
// polynomial has maximum absolute error below 1.5e-7, well under one BF16 ULP
// throughout the merger's useful range, while preserving Qwen's exact-GELU
// branch rather than substituting the tower MLP's tanh approximation.
inline float vision_qwen_erf(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float x = abs(value);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t
        + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t);
    return sign * (1.0f - polynomial * exp(-x * x));
}

kernel void vision_qwen_gelu_erf_bias(
    device bfloat* values [[buffer(0)]],
    device const bfloat* bias [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = float(values[index]) + float(bias[index % width]);
    values[index] = bfloat(
        0.5f * x * (1.0f + vision_qwen_erf(x * 0.7071067811865475f)));
}
