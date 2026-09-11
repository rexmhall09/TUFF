#include <metal_stdlib>
using namespace metal;

struct QSAParams {
    uint start, tokens, dim, heads, ratio, topBlocks, blocks, rotary;
    uint attentionDim, attentionHeads, kvHeads, partitions;
    float theta, scale;
};
inline float qsa_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
inline int qsa_axis(uint i, int3 p) {
    if (i % 3 == 1 && i < 33) return p.y;
    if (i % 3 == 2 && i < 30) return p.z;
    return p.x;
}
// Index Q/K projection is packed [Q heads, raw K]. Keys are pooled BEFORE
// normalization and RoPE. Queries use the attention module's rotary width.
kernel void qsa_prepare(device const half* projected [[buffer(0)]],
    device const ushort* norm [[buffer(1)]], device half* queries [[buffer(2)]],
    device half* rawKeys [[buffer(3)]], device packed_int3* positions [[buffer(4)]],
    device const packed_int3* currentPositions [[buffer(5)]], constant QSAParams& p [[buffer(6)]],
    uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint t = g.y, h = g.x;
    device const half* x = projected + (t * (p.heads + 1) + h) * p.dim;
    if (h == p.heads) {
        if (d < p.dim) rawKeys[(p.start + t) * p.dim + d] = x[d];
        if (d == 0) positions[p.start + t] = currentPositions[t];
        return;
    }
    threadgroup float values[128];
    threadgroup float sums[4];
    float v = d < p.dim ? float(x[d]) : 0;
    float sum = simd_sum(v * v);
    if (d % 32 == 0) sums[d / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((sums[0] + sums[1] + sums[2] + sums[3]) / p.dim + 1e-6f);
    // Preserve the FP16 intermediate at the norm/rotary boundary.
    values[d] = d < p.dim ? float(half(v * inv * (1 + qsa_bf16(norm[d])))) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= p.dim) return;
    v = values[d];
    if (d < p.rotary) {
        uint i = d % (p.rotary / 2);
        float angle = float(qsa_axis(i, int3(currentPositions[t]))) * pow(p.theta, -2.f * i / p.rotary);
        float other = values[(d + p.rotary / 2) % p.rotary];
        v = v * cos(angle) + (d < p.rotary / 2 ? -other : other) * sin(angle);
    }
    queries[(t * p.heads + h) * p.dim + d] = half(v);
}
kernel void qsa_pool(device const half* rawKeys [[buffer(0)]],
    device const ushort* norm [[buffer(1)]], device const packed_int3* positions [[buffer(2)]],
    device half* pooled [[buffer(3)]], constant QSAParams& p [[buffer(4)]],
    uint blockLocal [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint b = p.start / p.ratio + blockLocal;
    if (b >= p.blocks) return;
    float v = 0;
    if (d < p.dim) for (uint i = 0; i < p.ratio; ++i) v += float(rawKeys[(b * p.ratio + i) * p.dim + d]);
    v = float(half(v / p.ratio));
    threadgroup float values[128];
    threadgroup float sums[4];
    float sum = simd_sum(v * v);
    if (d % 32 == 0) sums[d / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((sums[0] + sums[1] + sums[2] + sums[3]) / p.dim + 1e-6f);
    values[d] = d < p.dim ? float(half(v * inv * (1 + qsa_bf16(norm[d])))) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d >= p.dim) return;
    v = values[d];
    if (d < p.rotary) {
        uint i = d % (p.rotary / 2);
        float angle = float(qsa_axis(i, int3(positions[b * p.ratio]))) * pow(p.theta, -2.f * i / p.rotary);
        float other = values[(d + p.rotary / 2) % p.rotary];
        v = v * cos(angle) + (d < p.rotary / 2 ? -other : other) * sin(angle);
    }
    pooled[b * p.dim + d] = half(v);
}
// One SIMD group per compressed block/query pair. Float32 scoring determines
// selection; using half products can change blocks near the cutoff.
kernel void qsa_scores(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device float* scores [[buffer(2)]], constant QSAParams& p [[buffer(3)]],
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint b = g.x, t = g.y;
    float score = 0;
    for (uint h = 0; h < p.heads; ++h) {
        float dot = 0;
        for (uint d = lane; d < p.dim; d += 32) dot += float(q[(t * p.heads + h) * p.dim + d]) * float(k[b * p.dim + d]);
        score += max(simd_sum(dot), 0.f);
    }
    if (lane == 0) scores[t * p.blocks + b] = b < (p.start + t + 1) / p.ratio ? score * rsqrt(float(p.dim)) : -INFINITY;
}
// Four radix histogram passes select a cutoff without sorting the entire
// context. Ties are resolved by block index, so repeated runs are deterministic.
kernel void qsa_select(device const float* scores [[buffer(0)]], device uint* selected [[buffer(1)]],
    constant QSAParams& p [[buffer(2)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint count = (p.start + t + 1) / p.ratio;
    if (count <= p.topBlocks) {
        for (uint i = tid; i < p.topBlocks; i += 256) selected[t * p.topBlocks + i] = i < count ? i : 0xffffffffu;
        return;
    }
    threadgroup atomic_uint hist[256];
    threadgroup uint prefix, mask, rank;
    threadgroup uint greater[256], equal[256], greaterTotal;
    if (tid == 0) { prefix = 0; mask = 0; rank = p.topBlocks; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(&hist[tid], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < count; i += 256) {
            uint key = as_type<uint>(scores[t * p.blocks + i]);
            if ((key & mask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (int b = 255; b >= 0; --b) {
                uint n = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (rank > n) rank -= n;
                else { prefix |= uint(b) << shift; break; }
            }
            mask |= 255u << shift;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Contiguous ranges plus prefix sums preserve block-index order for ties.
    uint lo = count * tid / 256, hi = count * (tid + 1) / 256;
    uint ng = 0, ne = 0;
    for (uint i = lo; i < hi; ++i) {
        uint key = as_type<uint>(scores[t * p.blocks + i]);
        ng += key > prefix; ne += key == prefix;
    }
    greater[tid] = ng; equal[tid] = ne;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint a = 0, b = 0;
        for (uint i = 0; i < 256; ++i) { uint x = greater[i], y = equal[i]; greater[i] = a; equal[i] = b; a += x; b += y; }
        greaterTotal = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint a = greater[tid], b = equal[tid] + greaterTotal;
    for (uint i = lo; i < hi; ++i) {
        uint key = as_type<uint>(scores[t * p.blocks + i]);
        if (key > prefix) selected[t * p.topBlocks + a++] = i;
        else if (key == prefix) { if (b < p.topBlocks) selected[t * p.topBlocks + b] = i; ++b; }
    }
}
// Split online softmax over selected blocks and the causal incomplete tail.
// No dense [query, context] attention mask or copied KV cache is needed.
kernel void qsa_attention_partial(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
    device const half* v [[buffer(2)]], device const uint* selected [[buffer(3)]],
    device float* partial [[buffer(4)]], constant QSAParams& p [[buffer(5)]],
    uint3 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint h = g.x, t = g.y, part = g.z, end = p.start + t + 1;
    uint complete = end / p.ratio, blocks = min(complete, p.topBlocks);
    uint n = blocks * p.ratio + end % p.ratio, kvh = h / (p.attentionHeads / p.kvHeads);
    float acc[8] = {0}, m = -INFINITY, sum = 0;
    for (uint i = part; i < n; i += p.partitions) {
        uint token = i < blocks * p.ratio ? selected[t * p.topBlocks + i / p.ratio] * p.ratio + i % p.ratio : complete * p.ratio + i - blocks * p.ratio;
        float dot = 0;
        for (uint d = lane; d < p.attentionDim; d += 32) dot += float(q[(t * p.attentionHeads + h) * p.attentionDim + d]) * float(k[(token * p.kvHeads + kvh) * p.attentionDim + d]);
        float score = simd_sum(dot) * p.scale, next = max(m, score);
        float a = exp(m - next), b = exp(score - next);
        for (uint d = lane; d < p.attentionDim; d += 32) acc[d / 32] = acc[d / 32] * a + b * float(v[(token * p.kvHeads + kvh) * p.attentionDim + d]);
        sum = sum * a + b; m = next;
    }
    uint base = ((t * p.attentionHeads + h) * p.partitions + part) * (p.attentionDim + 2);
    for (uint d = lane; d < p.attentionDim; d += 32) partial[base + d] = acc[d / 32];
    if (lane == 0) { partial[base + p.attentionDim] = m; partial[base + p.attentionDim + 1] = sum; }
}
kernel void qsa_attention_combine(device const float* partial [[buffer(0)]], device half* out [[buffer(1)]],
    constant QSAParams& p [[buffer(2)]], uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint base = (g.y * p.attentionHeads + g.x) * p.partitions * (p.attentionDim + 2);
    float m = -INFINITY;
    for (uint i = 0; i < p.partitions; ++i) m = max(m, partial[base + i * (p.attentionDim + 2) + p.attentionDim]);
    float total = 0, acc[8] = {0};
    for (uint i = 0; i < p.partitions; ++i) {
        uint b = base + i * (p.attentionDim + 2);
        float w = exp(partial[b + p.attentionDim] - m);
        total += w * partial[b + p.attentionDim + 1];
        for (uint d = lane; d < p.attentionDim; d += 32) acc[d / 32] += w * partial[b + d];
    }
    for (uint d = lane; d < p.attentionDim; d += 32) out[(g.y * p.attentionHeads + g.x) * p.attentionDim + d] = half(acc[d / 32] / total);
}
