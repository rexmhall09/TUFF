#include <metal_stdlib>
using namespace metal;

constant uint FC_ROPE_HEAD_DIM [[function_constant(50)]];
constant uint FC_ROPE_NUM_HEADS [[function_constant(51)]];
constant uint FC_ROPE_ROTATED_PAIRS [[function_constant(52)]];
constant bool FC_ROPE_USE_FC [[function_constant(53)]];

static inline uint rope_head_dim(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_HEAD_DIM))
        ? FC_ROPE_HEAD_DIM
        : runtime_value;
}
static inline uint rope_num_heads(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_NUM_HEADS))
        ? FC_ROPE_NUM_HEADS
        : runtime_value;
}

static inline uint rope_rotated_pairs(constant uint& runtime_value) {
    return (is_function_constant_defined(FC_ROPE_USE_FC) &&
            FC_ROPE_USE_FC &&
            is_function_constant_defined(FC_ROPE_ROTATED_PAIRS))
        ? FC_ROPE_ROTATED_PAIRS
        : runtime_value;
}

static inline void apply_neox_pair(
    device half* head,
    uint pair,
    uint half_dim,
    uint frequency_divisor,
    float position,
    float theta
) {
    const float exponent = -float(2u * pair) / float(frequency_divisor);
    const float angle = position * pow(theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const uint lower = pair;
    const uint upper = half_dim + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_default_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_dimension = dimension / 2u;
    if (pair >= half_dimension || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}

// Qwen-style partial RoPE: rotation confined to the first `rotary_dim`
// elements of each head, NeoX pairing (i, rotary_dim/2 + i) inside that
// window, frequency divisor = rotary_dim. Elements >= rotary_dim pass
// through untouched. (Gemma's proportional variant instead pairs across the
// full head and divides frequencies by head_dim — a different element set.)
kernel void rope_neox_subdim(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotary_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint half_rotary = rotary_dim / 2u;
    if (pair >= half_rotary || head_index >= heads) return;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_rotary, rotary_dim,
                    float(position), theta);
}

// YaRN frequency blending and concentration used by GPT-OSS. The low
// dimensions extrapolate their original frequencies while high dimensions
// interpolate by the context scaling factor, with a linear NTK-by-parts ramp.
kernel void rope_yarn_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& original_context_length [[buffer(5)]],
    constant float& scaling_factor [[buffer(6)]],
    constant float& beta_fast [[buffer(7)]],
    constant float& beta_slow [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = head_dim;
    const uint heads = num_heads;
    const uint half_dimension = dimension / 2u;
    if (pair >= half_dimension || head_index >= heads) return;

    const float log_theta = log(theta);
    const float low = float(half_dimension)
        * log(float(original_context_length) / (beta_fast * 2.0f * M_PI_F))
        / log_theta;
    const float high = float(half_dimension)
        * log(float(original_context_length) / (beta_slow * 2.0f * M_PI_F))
        / log_theta;
    const float ramp = clamp((float(pair) - low) / (high - low), 0.0f, 1.0f);
    const float inv_extrapolation = pow(theta, -float(2u * pair) / float(dimension));
    const float inv_interpolation = inv_extrapolation / scaling_factor;
    const float inv_frequency = mix(inv_extrapolation, inv_interpolation, ramp);
    const float angle = float(position + token_index) * inv_frequency;
    const float concentration = 0.1f * log(scaling_factor) + 1.0f;
    const float cosine = cos(angle) * concentration;
    const float sine = sin(angle) * concentration;

    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    const uint lower = pair;
    const uint upper = half_dimension + pair;
    const float x0 = float(head[lower]);
    const float x1 = float(head[upper]);
    head[lower] = half(x0 * cosine - x1 * sine);
    head[upper] = half(x0 * sine + x1 * cosine);
}

kernel void rope_proportional_neox(
    device half* data [[buffer(0)]],
    constant uint& position [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant float& theta [[buffer(4)]],
    constant uint& rotated_pairs [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint pair = gid.x;
    const uint head_index = gid.y;
    const uint token_index = gid.z;
    const uint dimension = rope_head_dim(head_dim);
    const uint heads = rope_num_heads(num_heads);
    const uint active_pairs = rope_rotated_pairs(rotated_pairs);
    if (pair >= active_pairs || head_index >= heads) return;

    const uint half_dimension = dimension / 2u;
    device half* head = data
        + token_index * heads * dimension
        + head_index * dimension;
    apply_neox_pair(head, pair, half_dimension, dimension,
                    float(position), theta);
}
