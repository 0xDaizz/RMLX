//! Softmax: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//!
//! MLX-style online softmax with SIMD reductions.
//!
//! Two kernel variants:
//! - `softmax_single_row_*`: small rows (axis_size <= N_READS * threadgroup_size).
//!   Loads data once into registers, computes max+normalizer, writes normalized output.
//! - `softmax_looped_*`: large rows. Online softmax: single pass computes running
//!   max and normalizer with correction factor.
//!
//! SIMD reductions replace the old `threadgroup float shared_data[1024]` tree
//! reduction with `simd_sum()`/`simd_max()` + small shared memory (one per simdgroup).

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use objc2_metal::MTLDevice as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLResourceOptions;
use rmlx_metal::MTLSize;

/// N_READS: each thread processes 4 consecutive elements per iteration for
/// better memory coalescing.
const N_READS: usize = 4;

pub const SOFTMAX_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ---------------------------------------------------------------------------
// SIMD cross-simdgroup reduction helpers
// ---------------------------------------------------------------------------

constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint N_READS   = 4;

// Reduce max across all simdgroups in the threadgroup.
// After this call every thread has the same `max_val` and `normalizer`.
inline void cross_simdgroup_reduce(
    thread float& max_val,
    thread float& normalizer,
    threadgroup float* local_max,
    threadgroup float* local_normalizer,
    uint simd_lane_id,
    uint simd_group_id)
{
    // -- SIMD-level reduction (within one simdgroup) --
    float simd_max_val   = simd_max(max_val);
    normalizer          *= fast::exp(max_val - simd_max_val);
    max_val              = simd_max_val;
    float simd_norm      = simd_sum(normalizer);

    // -- Cross-simdgroup reduction via shared memory --
    if (simd_group_id == 0) {
        local_max[simd_lane_id]        = -INFINITY;
        local_normalizer[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        local_max[simd_group_id]        = max_val;
        local_normalizer[simd_group_id] = simd_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup does the final reduction
    if (simd_group_id == 0) {
        float m = local_max[simd_lane_id];
        float n = local_normalizer[simd_lane_id];
        float final_max = simd_max(m);
        n *= fast::exp(m - final_max);
        float final_norm = simd_sum(n);
        if (simd_lane_id == 0) {
            local_max[0]        = final_max;
            local_normalizer[0] = 1.0f / final_norm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_val    = local_max[0];
    normalizer = local_normalizer[0];  // this is actually 1/normalizer now
}

// ===========================================================================
// softmax_single_row: small rows that fit in registers (axis_size <= N_READS * tgsize)
// ===========================================================================

// ---- float32 ----
kernel void softmax_single_row_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    // Load N_READS elements per thread into registers
    float vals[N_READS];
    float max_val = -INFINITY;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = input[base + idx];
            max_val = max(max_val, vals[r]);
        } else {
            vals[r] = -INFINITY;
        }
    }

    // Reduce max across threadgroup using cross-simdgroup reduction.
    // We pass normalizer=0 since we haven't computed exp values yet;
    // the reduction gives us the global max (normalizer result is unused).
    float normalizer = 0.0f;
    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);
    float row_max = max_val;

    normalizer = 0.0f;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = fast::exp(vals[r] - row_max);
            normalizer += vals[r];
        }
    }

    // Reduce normalizer across threadgroup
    float sum_simd = simd_sum(normalizer);
    if (simd_group_id == 0) {
        local_normalizer[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_normalizer[simd_group_id] = sum_simd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float n = local_normalizer[simd_lane_id];
        float total = simd_sum(n);
        if (simd_lane_id == 0) {
            local_normalizer[0] = 1.0f / total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_normalizer[0];

    // Write from registers -- no re-read from global memory
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            output[base + idx] = vals[r] * inv_norm;
        }
    }
}

// ---- float16 (accumulate in float32) ----
kernel void softmax_single_row_f16(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float vals[N_READS];
    float max_val = -INFINITY;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = float(input[base + idx]);
            max_val = max(max_val, vals[r]);
        } else {
            vals[r] = -INFINITY;
        }
    }

    float normalizer = 0.0f;
    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);
    float row_max = max_val;

    normalizer = 0.0f;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = fast::exp(vals[r] - row_max);
            normalizer += vals[r];
        }
    }

    float sum_simd = simd_sum(normalizer);
    if (simd_group_id == 0) { local_normalizer[simd_lane_id] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) { local_normalizer[simd_group_id] = sum_simd; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float n = local_normalizer[simd_lane_id];
        float total = simd_sum(n);
        if (simd_lane_id == 0) { local_normalizer[0] = 1.0f / total; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_normalizer[0];

    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            output[base + idx] = half(vals[r] * inv_norm);
        }
    }
}

// ---- bfloat16 (accumulate in float32) ----
kernel void softmax_single_row_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint&       axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float vals[N_READS];
    float max_val = -INFINITY;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = float(input[base + idx]);
            max_val = max(max_val, vals[r]);
        } else {
            vals[r] = -INFINITY;
        }
    }

    float normalizer = 0.0f;
    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);
    float row_max = max_val;

    normalizer = 0.0f;
    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            vals[r] = fast::exp(vals[r] - row_max);
            normalizer += vals[r];
        }
    }

    float sum_simd = simd_sum(normalizer);
    if (simd_group_id == 0) { local_normalizer[simd_lane_id] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) { local_normalizer[simd_group_id] = sum_simd; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float n = local_normalizer[simd_lane_id];
        float total = simd_sum(n);
        if (simd_lane_id == 0) { local_normalizer[0] = 1.0f / total; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_norm = local_normalizer[0];

    for (uint r = 0; r < N_READS; r++) {
        uint idx = tid + r * tgsize;
        if (idx < axis_size) {
            output[base + idx] = bfloat(vals[r] * inv_norm);
        }
    }
}

// ===========================================================================
// softmax_looped: large rows — online softmax (single pass for max+normalizer)
// ===========================================================================

// ---- float32 ----
kernel void softmax_looped_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    // Online pass: compute running max and normalizer simultaneously
    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = input[base + i];
            float v1 = input[base + i + 1];
            float v2 = input[base + i + 2];
            float v3 = input[base + i + 3];
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = input[base + j];
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    // Cross-threadgroup reduction
    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;  // cross_simdgroup_reduce stores 1/sum

    // Write pass
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = fast::exp(input[base + i]     - row_max) * inv_norm;
            output[base + i + 1] = fast::exp(input[base + i + 1] - row_max) * inv_norm;
            output[base + i + 2] = fast::exp(input[base + i + 2] - row_max) * inv_norm;
            output[base + i + 3] = fast::exp(input[base + i + 3] - row_max) * inv_norm;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = fast::exp(input[base + j] - row_max) * inv_norm;
            }
        }
    }
}

// ---- float16 (accumulate in float32) ----
kernel void softmax_looped_f16(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = float(input[base + j]);
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = half(fast::exp(float(input[base + i])     - row_max) * inv_norm);
            output[base + i + 1] = half(fast::exp(float(input[base + i + 1]) - row_max) * inv_norm);
            output[base + i + 2] = half(fast::exp(float(input[base + i + 2]) - row_max) * inv_norm);
            output[base + i + 3] = half(fast::exp(float(input[base + i + 3]) - row_max) * inv_norm);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = half(fast::exp(float(input[base + j]) - row_max) * inv_norm);
            }
        }
    }
}

// ---- bfloat16 (accumulate in float32) ----
kernel void softmax_looped_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint&       axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = float(input[base + j]);
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = bfloat(fast::exp(float(input[base + i])     - row_max) * inv_norm);
            output[base + i + 1] = bfloat(fast::exp(float(input[base + i + 1]) - row_max) * inv_norm);
            output[base + i + 2] = bfloat(fast::exp(float(input[base + i + 2]) - row_max) * inv_norm);
            output[base + i + 3] = bfloat(fast::exp(float(input[base + i + 3]) - row_max) * inv_norm);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = bfloat(fast::exp(float(input[base + j]) - row_max) * inv_norm);
            }
        }
    }
}

// ===========================================================================
// Compatibility aliases: softmax_f32 maps to softmax_looped_f32.
// This preserves backward compat for any code that looks up "softmax_f32".
// ===========================================================================
kernel void softmax_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    // Online pass: compute running max and normalizer simultaneously
    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = input[base + i];
            float v1 = input[base + i + 1];
            float v2 = input[base + i + 2];
            float v3 = input[base + i + 3];
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = input[base + j];
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    // Cross-threadgroup reduction
    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;  // cross_simdgroup_reduce stores 1/sum

    // Write pass
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = fast::exp(input[base + i]     - row_max) * inv_norm;
            output[base + i + 1] = fast::exp(input[base + i + 1] - row_max) * inv_norm;
            output[base + i + 2] = fast::exp(input[base + i + 2] - row_max) * inv_norm;
            output[base + i + 3] = fast::exp(input[base + i + 3] - row_max) * inv_norm;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = fast::exp(input[base + j] - row_max) * inv_norm;
            }
        }
    }
}

kernel void softmax_f16(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = float(input[base + j]);
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = half(fast::exp(float(input[base + i])     - row_max) * inv_norm);
            output[base + i + 1] = half(fast::exp(float(input[base + i + 1]) - row_max) * inv_norm);
            output[base + i + 2] = half(fast::exp(float(input[base + i + 2]) - row_max) * inv_norm);
            output[base + i + 3] = half(fast::exp(float(input[base + i + 3]) - row_max) * inv_norm);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = half(fast::exp(float(input[base + j]) - row_max) * inv_norm);
            }
        }
    }
}

kernel void softmax_bf16(
    device const bfloat* input  [[buffer(0)]],
    device bfloat*       output [[buffer(1)]],
    constant uint&       axis_size [[buffer(2)]],
    uint row          [[threadgroup_position_in_grid]],
    uint tid          [[thread_position_in_threadgroup]],
    uint tgsize       [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_max[SIMD_SIZE];
    threadgroup float local_normalizer[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float max_val    = -INFINITY;
    float normalizer = 0.0f;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            float prev_max;
            prev_max = max_val; max_val = max(max_val, v0);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v0 - max_val);
            prev_max = max_val; max_val = max(max_val, v1);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v1 - max_val);
            prev_max = max_val; max_val = max(max_val, v2);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v2 - max_val);
            prev_max = max_val; max_val = max(max_val, v3);
            normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(v3 - max_val);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float val = float(input[base + j]);
                float prev_max = max_val;
                max_val = max(max_val, val);
                normalizer = normalizer * fast::exp(prev_max - max_val) + fast::exp(val - max_val);
            }
        }
    }

    cross_simdgroup_reduce(max_val, normalizer, local_max, local_normalizer,
                           simd_lane_id, simd_group_id);

    float row_max  = max_val;
    float inv_norm = normalizer;

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            output[base + i]     = bfloat(fast::exp(float(input[base + i])     - row_max) * inv_norm);
            output[base + i + 1] = bfloat(fast::exp(float(input[base + i + 1]) - row_max) * inv_norm);
            output[base + i + 2] = bfloat(fast::exp(float(input[base + i + 2]) - row_max) * inv_norm);
            output[base + i + 3] = bfloat(fast::exp(float(input[base + i + 3]) - row_max) * inv_norm);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                output[base + j] = bfloat(fast::exp(float(input[base + j]) - row_max) * inv_norm);
            }
        }
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("softmax", SOFTMAX_SHADER_SOURCE)
}

/// Apply softmax over the last axis of an input array.
///
/// Supports arbitrary-dimensional input: all leading dimensions are flattened
/// into batch rows, and softmax is applied along the last axis. For example,
/// an input of shape `[2, 3, 4]` is treated as 6 rows of 4 elements each.
///
/// The output has the same shape as the input.
///
/// Supported dtypes: `Float32`, `Float16`, `Bfloat16`. The `f16`/`bf16`
/// variants accumulate reductions in `f32` for numerical stability.
pub fn softmax(
    registry: &KernelRegistry,
    input: &Array,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if input.ndim() == 0 {
        return Err(KernelError::InvalidShape(
            "softmax requires at least 1D input, got 0D".to_string(),
        ));
    }

    // Handle zero-element tensors: return empty output without GPU dispatch.
    let total_numel: usize = input.shape().iter().product();
    if total_numel == 0 {
        return Ok(Array::zeros(
            registry.device().raw(),
            input.shape(),
            input.dtype(),
        ));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let shape = input.shape();
    let axis_size = *shape
        .last()
        .ok_or_else(|| KernelError::InvalidShape("softmax: empty shape (0D tensor)".to_string()))?;
    let num_rows: usize = shape.iter().rev().skip(1).product();
    // For 1-D input there is exactly one row.
    let num_rows = if num_rows == 0 { 1 } else { num_rows };

    let axis_size_u32 = super::checked_u32(axis_size, "axis_size")?;

    // Choose kernel variant based on dtype and row size.
    let tg_size_hint: usize = 1024; // will be clamped below
    let use_single_row = axis_size <= N_READS * tg_size_hint;

    let kernel_name = match (input.dtype(), use_single_row) {
        (DType::Float32, true) => "softmax_single_row_f32",
        (DType::Float32, false) => "softmax_looped_f32",
        (DType::Float16, true) => "softmax_single_row_f16",
        (DType::Float16, false) => "softmax_looped_f16",
        (DType::Bfloat16, true) => "softmax_single_row_bf16",
        (DType::Bfloat16, false) => "softmax_looped_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "softmax not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let out = Array::zeros(registry.device().raw(), shape, input.dtype());

    let axis_buf = unsafe {
        registry
            .device()
            .raw()
            .newBufferWithBytes_length_options(
                std::ptr::NonNull::new(
                    &axis_size_u32 as *const u32 as *const std::ffi::c_void
                        as *mut std::ffi::c_void,
                )
                .unwrap(),
                4_usize,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    };

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset());
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&axis_buf), 0);
    let tg_size = std::cmp::min(tg_size_hint, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: num_rows as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
    encoder.end();
    super::commit_with_mode(&command_buffer, super::ExecMode::Sync);

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_zero_rank_returns_error() {
        let gpu_dev = crate::test_utils::require_gpu!();
        let queue = gpu_dev.new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        super::register(&registry).expect("register softmax kernels");
        let device = registry.device().raw();

        // Create a 0D (scalar) array
        let scalar = Array::zeros(device, &[], DType::Float32);
        assert_eq!(scalar.ndim(), 0);

        let result = softmax(&registry, &scalar, &queue);
        assert!(result.is_err(), "softmax on 0D tensor should return error");
        match result {
            Err(KernelError::InvalidShape(msg)) => {
                assert!(msg.contains("0D"), "error should mention 0D: {msg}");
            }
            Err(other) => panic!("expected InvalidShape, got {other:?}"),
            Ok(_) => unreachable!(),
        }
    }
}
