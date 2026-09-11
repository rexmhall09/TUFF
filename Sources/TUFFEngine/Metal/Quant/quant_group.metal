#include <metal_stdlib>
using namespace metal;

// ============================================================================
// quant_group — affine quantization constants shared across shader modules.
//
// The runtime library is one concatenated translation unit, so a function
// constant can only be declared once. This module is compiled first and owns
// the declarations; dequant_int4.metal and moe.metal read them.
//
// Group size is 64 for every checkpoint in the lineup except Qwen3.8 Flash
// Next, which is quantized at 32 — a requirement rather than a preference,
// since its n-gram PLE rows are 160 values wide and 160 is not divisible by
// 64. Leaving the constant undefined selects the historical 64, so the
// shipping models compile to exactly what they did before it became variable.
// ============================================================================

constant constexpr uint kQuantGroupSizeDefault = 64;
constant uint FC_QUANT_GROUP [[function_constant(100)]];

static inline uint quant_group_size() {
    return is_function_constant_defined(FC_QUANT_GROUP) ? FC_QUANT_GROUP
                                                        : kQuantGroupSizeDefault;
}

// Routed experts one token can select. Eight until Qwen3.8 Flash Next, which
// routes ten of its 512. The blob array is sized for the largest supported;
// the active count is specialized so the eight-expert path keeps its
// constant-folded loop bounds.
constant constexpr uint kMaxStreamedExperts = 16;
constant constexpr uint kStreamedExpertsDefault = 8;
constant uint FC_STREAMED_EXPERTS [[function_constant(101)]];

static inline uint streamed_expert_count() {
    return is_function_constant_defined(FC_STREAMED_EXPERTS)
        ? FC_STREAMED_EXPERTS : kStreamedExpertsDefault;
}
