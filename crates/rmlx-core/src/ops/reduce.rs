//! Reduction operations: sum, max, min, mean, argmax.
//!
//! All kernels use SIMD-based reductions (`simd_sum` / `simd_max` / `simd_min`)
//! with a small `threadgroup float local[32]` cross-simdgroup gather, matching
//! the pattern used in `rms_norm`. Each thread processes N_READS=4 elements per
//! loop iteration for coalesced memory access.
//!
//! f16 and bf16 variants accumulate in f32 for precision, then write back in the
//! original type. Min/max kernels propagate NaN.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Threshold for switching from single-pass to two-pass reduction.
/// For arrays with <= this many elements, single threadgroup is used.
const TWO_PASS_THRESHOLD: usize = 1024;

/// Metal shader source for reduction kernels.
pub const REDUCE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

constexpr uint SIMD_SIZE = 32;
constexpr uint N_READS  = 4;

// NaN-propagating min/max helpers
static inline float nan_max(float a, float b) {
    if (isnan(a) || isnan(b)) return NAN;
    return max(a, b);
}
static inline float nan_min(float a, float b) {
    if (isnan(a) || isnan(b)) return NAN;
    return min(a, b);
}

// -----------------------------------------------------------------------
// f32 global reductions  (single threadgroup — small arrays or pass 2)
// -----------------------------------------------------------------------

kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    // N_READS coalescing
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc += input[i + r];
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = acc;
    }
}

kernel void reduce_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_max(acc, input[i + r]);
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = acc;
    }
}

kernel void reduce_min_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_min(acc, input[i + r]);
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = acc;
    }
}

// Argmax: still uses threadgroup shared mem for index tracking
kernel void reduce_argmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_vals[1024];
    threadgroup uint shared_idxs[1024];

    float max_val = -INFINITY;
    uint max_idx = 0;
    for (uint i = tid; i < size; i += tgsize) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    shared_vals[tid] = max_val;
    shared_idxs[tid] = max_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idxs[tid] = shared_idxs[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = (float)shared_idxs[0];
    }
}

// -----------------------------------------------------------------------
// f32 multi-threadgroup pass 1 (each threadgroup reduces a chunk)
// -----------------------------------------------------------------------

kernel void reduce_sum_pass1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& chunk_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    uint chunk_start = gid * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, size);

    float acc = 0;
    uint base = chunk_start + tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < chunk_end; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < chunk_end; ++r) {
            acc += input[i + r];
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[gid] = acc;
    }
}

kernel void reduce_max_pass1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& chunk_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    uint chunk_start = gid * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, size);

    float acc = -INFINITY;
    uint base = chunk_start + tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < chunk_end; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < chunk_end; ++r) {
            acc = nan_max(acc, input[i + r]);
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[gid] = acc;
    }
}

kernel void reduce_min_pass1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& chunk_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    uint chunk_start = gid * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, size);

    float acc = INFINITY;
    uint base = chunk_start + tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < chunk_end; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < chunk_end; ++r) {
            acc = nan_min(acc, input[i + r]);
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[gid] = acc;
    }
}

// -----------------------------------------------------------------------
// f32 row-wise reductions (one threadgroup per row)
// -----------------------------------------------------------------------

kernel void reduce_sum_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc += input[row_base + i + r];
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = acc;
    }
}

kernel void reduce_max_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_max(acc, input[row_base + i + r]);
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = acc;
    }
}

kernel void reduce_min_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_min(acc, input[row_base + i + r]);
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = acc;
    }
}

// -----------------------------------------------------------------------
// f32 column-wise reductions (one threadgroup per column)
// Reduce along axis 0: input [rows, cols] -> output [cols]
// -----------------------------------------------------------------------

kernel void reduce_sum_col_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    for (uint r = tid; r < rows; r += tgsize) {
        acc += input[r * cols + col];
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = acc;
    }
}

kernel void reduce_max_col_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_max(acc, input[r * cols + col]);
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = acc;
    }
}

kernel void reduce_min_col_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_min(acc, input[r * cols + col]);
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = acc;
    }
}

// -----------------------------------------------------------------------
// f16 global reductions (accumulate in f32)
// -----------------------------------------------------------------------

kernel void reduce_sum_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc += float(input[i + r]);
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = half(acc);
    }
}

kernel void reduce_max_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_max(acc, float(input[i + r]));
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = half(acc);
    }
}

kernel void reduce_min_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_min(acc, float(input[i + r]));
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = half(acc);
    }
}

// -----------------------------------------------------------------------
// bf16 global reductions (accumulate in f32)
// -----------------------------------------------------------------------

kernel void reduce_sum_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc += float(input[i + r]);
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = bfloat(acc);
    }
}

kernel void reduce_max_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_max(acc, float(input[i + r]));
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = bfloat(acc);
    }
}

kernel void reduce_min_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < size; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < size; ++r) {
            acc = nan_min(acc, float(input[i + r]));
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[0] = bfloat(acc);
    }
}

// -----------------------------------------------------------------------
// f16 row-wise reductions
// -----------------------------------------------------------------------

kernel void reduce_sum_row_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc += float(input[row_base + i + r]);
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = half(acc);
    }
}

kernel void reduce_max_row_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_max(acc, float(input[row_base + i + r]));
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = half(acc);
    }
}

kernel void reduce_min_row_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_min(acc, float(input[row_base + i + r]));
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = half(acc);
    }
}

// -----------------------------------------------------------------------
// bf16 row-wise reductions
// -----------------------------------------------------------------------

kernel void reduce_sum_row_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc += float(input[row_base + i + r]);
        }
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = bfloat(acc);
    }
}

kernel void reduce_max_row_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_max(acc, float(input[row_base + i + r]));
        }
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = bfloat(acc);
    }
}

kernel void reduce_min_row_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    uint row_base = row * cols;
    uint base = tid * N_READS;
    uint stride = tgsize * N_READS;
    for (uint i = base; i < cols; i += stride) {
        for (uint r = 0; r < N_READS && (i + r) < cols; ++r) {
            acc = nan_min(acc, float(input[row_base + i + r]));
        }
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[row] = bfloat(acc);
    }
}

// -----------------------------------------------------------------------
// f16 column-wise reductions
// -----------------------------------------------------------------------

kernel void reduce_sum_col_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    for (uint r = tid; r < rows; r += tgsize) {
        acc += float(input[r * cols + col]);
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = half(acc);
    }
}

kernel void reduce_max_col_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_max(acc, float(input[r * cols + col]));
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = half(acc);
    }
}

kernel void reduce_min_col_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_min(acc, float(input[r * cols + col]));
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = half(acc);
    }
}

// -----------------------------------------------------------------------
// bf16 column-wise reductions
// -----------------------------------------------------------------------

kernel void reduce_sum_col_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[SIMD_SIZE];

    float acc = 0;
    for (uint r = tid; r < rows; r += tgsize) {
        acc += float(input[r * cols + col]);
    }

    acc = simd_sum(acc);

    if (simd_group_id == 0) local_sums[simd_lane_id] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = bfloat(acc);
    }
}

kernel void reduce_max_col_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = -INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_max(acc, float(input[r * cols + col]));
    }

    acc = simd_max(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_max(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = bfloat(acc);
    }
}

kernel void reduce_min_col_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_vals[SIMD_SIZE];

    float acc = INFINITY;
    for (uint r = tid; r < rows; r += tgsize) {
        acc = nan_min(acc, float(input[r * cols + col]));
    }

    acc = simd_min(acc);

    if (simd_group_id == 0) local_vals[simd_lane_id] = INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) local_vals[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_min(local_vals[simd_lane_id]);
        if (simd_lane_id == 0) output[col] = bfloat(acc);
    }
}

// -----------------------------------------------------------------------
// Scalar divide (for mean = sum / count, applied after reduce)
// -----------------------------------------------------------------------

kernel void scalar_div_f32(
    device float* data [[buffer(0)]],
    constant float& divisor [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    data[id] = data[id] / divisor;
}

kernel void scalar_div_f16(
    device half* data [[buffer(0)]],
    constant float& divisor [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    data[id] = half(float(data[id]) / divisor);
}

kernel void scalar_div_bf16(
    device bfloat* data [[buffer(0)]],
    constant float& divisor [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    data[id] = bfloat(float(data[id]) / divisor);
}
"#;

/// Reduction type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
    ArgMax,
}

/// Reduction axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceAxis {
    /// Reduce all elements to a scalar.
    All,
    /// Reduce along the last axis (rows): `[rows, cols] -> [rows]`.
    Row,
    /// Reduce along axis 0 (columns): `[rows, cols] -> [cols]`.
    Col,
    /// Reduce along an arbitrary axis.
    Axis(usize),
}

/// Register reduce kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("reduce", REDUCE_SHADER_SOURCE)
}

// -----------------------------------------------------------------------
// Kernel-name lookup helpers
// -----------------------------------------------------------------------

/// Return (kernel_name, output_dtype) for a single-pass all-reduce.
fn global_kernel_name(op: ReduceOp, dtype: DType) -> Result<(&'static str, DType), KernelError> {
    // Mean reuses the sum kernel; the Rust side divides afterwards.
    match (op, dtype) {
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float32) => Ok(("reduce_sum_f32", DType::Float32)),
        (ReduceOp::Max, DType::Float32) => Ok(("reduce_max_f32", DType::Float32)),
        (ReduceOp::Min, DType::Float32) => Ok(("reduce_min_f32", DType::Float32)),
        (ReduceOp::ArgMax, DType::Float32) => Ok(("reduce_argmax_f32", DType::Float32)),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float16) => Ok(("reduce_sum_f16", DType::Float16)),
        (ReduceOp::Max, DType::Float16) => Ok(("reduce_max_f16", DType::Float16)),
        (ReduceOp::Min, DType::Float16) => Ok(("reduce_min_f16", DType::Float16)),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Bfloat16) => {
            Ok(("reduce_sum_bf16", DType::Bfloat16))
        }
        (ReduceOp::Max, DType::Bfloat16) => Ok(("reduce_max_bf16", DType::Bfloat16)),
        (ReduceOp::Min, DType::Bfloat16) => Ok(("reduce_min_bf16", DType::Bfloat16)),
        _ => Err(KernelError::NotFound(format!(
            "reduce {:?} not supported for {:?}",
            op, dtype
        ))),
    }
}

/// Return (pass1_kernel, pass2_kernel) for two-pass all-reduce.
fn two_pass_kernel_names(
    op: ReduceOp,
    dtype: DType,
) -> Result<(&'static str, &'static str), KernelError> {
    match (op, dtype) {
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float32) => {
            Ok(("reduce_sum_pass1_f32", "reduce_sum_f32"))
        }
        (ReduceOp::Max, DType::Float32) => Ok(("reduce_max_pass1_f32", "reduce_max_f32")),
        (ReduceOp::Min, DType::Float32) => Ok(("reduce_min_pass1_f32", "reduce_min_f32")),
        _ => Err(KernelError::NotFound(format!(
            "two-pass reduce {:?} not supported for {:?}",
            op, dtype
        ))),
    }
}

/// Return the kernel name for row-wise reduction.
fn row_kernel_name(op: ReduceOp, dtype: DType) -> Result<&'static str, KernelError> {
    match (op, dtype) {
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float32) => Ok("reduce_sum_row_f32"),
        (ReduceOp::Max, DType::Float32) => Ok("reduce_max_row_f32"),
        (ReduceOp::Min, DType::Float32) => Ok("reduce_min_row_f32"),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float16) => Ok("reduce_sum_row_f16"),
        (ReduceOp::Max, DType::Float16) => Ok("reduce_max_row_f16"),
        (ReduceOp::Min, DType::Float16) => Ok("reduce_min_row_f16"),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Bfloat16) => Ok("reduce_sum_row_bf16"),
        (ReduceOp::Max, DType::Bfloat16) => Ok("reduce_max_row_bf16"),
        (ReduceOp::Min, DType::Bfloat16) => Ok("reduce_min_row_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "reduce_row {:?} not supported for {:?}",
            op, dtype
        ))),
    }
}

/// Return the kernel name for column-wise reduction.
fn col_kernel_name(op: ReduceOp, dtype: DType) -> Result<&'static str, KernelError> {
    match (op, dtype) {
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float32) => Ok("reduce_sum_col_f32"),
        (ReduceOp::Max, DType::Float32) => Ok("reduce_max_col_f32"),
        (ReduceOp::Min, DType::Float32) => Ok("reduce_min_col_f32"),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Float16) => Ok("reduce_sum_col_f16"),
        (ReduceOp::Max, DType::Float16) => Ok("reduce_max_col_f16"),
        (ReduceOp::Min, DType::Float16) => Ok("reduce_min_col_f16"),
        (ReduceOp::Sum | ReduceOp::Mean, DType::Bfloat16) => Ok("reduce_sum_col_bf16"),
        (ReduceOp::Max, DType::Bfloat16) => Ok("reduce_max_col_bf16"),
        (ReduceOp::Min, DType::Bfloat16) => Ok("reduce_min_col_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "reduce_col {:?} not supported for {:?}",
            op, dtype
        ))),
    }
}

/// Return the scalar-divide kernel name for the given dtype.
fn scalar_div_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("scalar_div_f32"),
        DType::Float16 => Ok("scalar_div_f16"),
        DType::Bfloat16 => Ok("scalar_div_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "scalar_div not supported for {:?}",
            dtype
        ))),
    }
}

// -----------------------------------------------------------------------
// In-place scalar divide (for Mean)
// -----------------------------------------------------------------------

/// Divide every element of `output` by `count` in-place on the GPU.
fn apply_mean_divisor(
    registry: &KernelRegistry,
    output: &Array,
    count: usize,
    queue: &metal::CommandQueue,
) -> Result<(), KernelError> {
    let kernel_name = scalar_div_kernel_name(output.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, output.dtype())?;
    let divisor = count as f32;

    let div_buf = registry.device().raw().new_buffer_with_data(
        &divisor as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let numel = output.numel() as u64;
    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(output.metal_buffer()), 0);
    encoder.set_buffer(1, Some(&div_buf), 0);

    let tg_size = std::cmp::min(256u64, pipeline.max_total_threads_per_threadgroup());
    let grid = numel.div_ceil(tg_size);
    encoder.dispatch_thread_groups(MTLSize::new(grid, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}

// -----------------------------------------------------------------------
// Public API: reduce_all
// -----------------------------------------------------------------------

/// Reduce all elements of the array.
pub fn reduce_all(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all_with_mode(registry, input, op, queue, super::ExecMode::Sync)
}

/// Reduce all elements with explicit execution mode.
///
/// For arrays with > 1024 elements, uses a two-pass reduction:
/// - Pass 1: Multiple threadgroups each reduce a chunk to a partial result
/// - Pass 2: A single threadgroup reduces the partial results to the final scalar
///
/// For small arrays (<= 1024 elements), uses a single-pass reduction.
///
/// `ReduceOp::Mean` computes sum then divides by element count on the GPU.
pub fn reduce_all_with_mode(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let numel = input.numel();
    let is_mean = matches!(op, ReduceOp::Mean);

    // ArgMax always uses single-pass (needs index tracking)
    let result = if matches!(op, ReduceOp::ArgMax) || numel <= TWO_PASS_THRESHOLD {
        reduce_all_single_pass(registry, input, op, queue, mode)?
    } else {
        reduce_all_two_pass(registry, input, op, queue, mode)?
    };

    if is_mean {
        apply_mean_divisor(registry, &result, numel, queue)?;
    }

    Ok(result)
}

/// Single-pass reduction (for small arrays and argmax).
fn reduce_all_single_pass(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    let (kernel_name, out_dtype) = global_kernel_name(op, input.dtype())?;

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = super::checked_u32(input.numel(), "numel")?;

    let out = Array::zeros(registry.device().raw(), &[1], out_dtype);

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&size_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Two-pass reduction for large arrays (Sum, Max, Min, Mean).
fn reduce_all_two_pass(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    let numel = input.numel();
    let numel_u32 = super::checked_u32(numel, "numel")?;

    let (pass1_kernel, pass2_kernel) = two_pass_kernel_names(op, input.dtype())?;

    let pass1_pipeline = registry.get_pipeline(pass1_kernel, input.dtype())?;
    let tg_size = std::cmp::min(1024u64, pass1_pipeline.max_total_threads_per_threadgroup());

    // Number of threadgroups for pass 1: enough to cover all elements with good occupancy.
    // Must be power of 2 because pass 2 uses tree reduction over the partial results.
    let raw_tg_count = (numel as u64).div_ceil(tg_size).min(256);
    let num_threadgroups = raw_tg_count.next_power_of_two().min(256) as u32;

    // Allocate intermediate buffer for partial results
    let partial = Array::zeros(
        registry.device().raw(),
        &[num_threadgroups as usize],
        DType::Float32,
    );

    // Each threadgroup handles a contiguous chunk
    let chunk_size = (numel as u64).div_ceil(num_threadgroups as u64) as u32;

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let chunk_buf = registry.device().raw().new_buffer_with_data(
        &chunk_size as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // --- Pass 1: reduce input -> partial results ---
    let cb1 = queue.new_command_buffer();
    let enc1 = cb1.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pass1_pipeline);
    enc1.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    enc1.set_buffer(1, Some(partial.metal_buffer()), 0);
    enc1.set_buffer(2, Some(&size_buf), 0);
    enc1.set_buffer(3, Some(&chunk_buf), 0);

    enc1.dispatch_thread_groups(
        MTLSize::new(num_threadgroups as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc1.end_encoding();
    cb1.commit();
    cb1.wait_until_completed();

    // --- Pass 2: reduce partial results -> scalar ---
    let pass2_pipeline = registry.get_pipeline(pass2_kernel, DType::Float32)?;
    let out = Array::zeros(registry.device().raw(), &[1], DType::Float32);

    let partial_size_buf = registry.device().raw().new_buffer_with_data(
        &num_threadgroups as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pass2_pipeline);
    enc2.set_buffer(0, Some(partial.metal_buffer()), 0);
    enc2.set_buffer(1, Some(out.metal_buffer()), 0);
    enc2.set_buffer(2, Some(&partial_size_buf), 0);

    // Threadgroup size for pass 2 must be a power of 2 for the tree reduction
    let tg2_size = (num_threadgroups as u64)
        .next_power_of_two()
        .min(pass2_pipeline.max_total_threads_per_threadgroup());
    enc2.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg2_size.max(1), 1, 1));
    enc2.end_encoding();
    super::commit_with_mode(cb2, mode);

    Ok(out)
}

// -----------------------------------------------------------------------
// Public API: reduce_row
// -----------------------------------------------------------------------

/// Reduce along rows (for 2D arrays).
/// `[rows, cols] -> [rows]` by reducing each row of `cols` elements.
pub fn reduce_row(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "reduce_row requires 2D array, got {}D",
            input.ndim()
        )));
    }
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let cols_u32 = super::checked_u32(cols, "cols")?;
    let is_mean = matches!(op, ReduceOp::Mean);

    let kernel_name = row_kernel_name(op, input.dtype())?;

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let out = Array::zeros(registry.device().raw(), &[rows], input.dtype());

    let cols_buf = registry.device().raw().new_buffer_with_data(
        &cols_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&cols_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if is_mean {
        apply_mean_divisor(registry, &out, cols, queue)?;
    }

    Ok(out)
}

// -----------------------------------------------------------------------
// Public API: reduce_col
// -----------------------------------------------------------------------

/// Reduce along columns (axis 0) for 2D arrays.
/// `[rows, cols] -> [cols]` by reducing each column of `rows` elements.
pub fn reduce_col(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "reduce_col requires 2D array, got {}D",
            input.ndim()
        )));
    }
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let rows_u32 = super::checked_u32(rows, "rows")?;
    let cols_u32 = super::checked_u32(cols, "cols")?;
    let is_mean = matches!(op, ReduceOp::Mean);

    let kernel_name = col_kernel_name(op, input.dtype())?;

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let out = Array::zeros(registry.device().raw(), &[cols], input.dtype());

    let rows_buf = registry.device().raw().new_buffer_with_data(
        &rows_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let cols_buf = registry.device().raw().new_buffer_with_data(
        &cols_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&rows_buf), 0);
    encoder.set_buffer(3, Some(&cols_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(cols as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if is_mean {
        apply_mean_divisor(registry, &out, rows, queue)?;
    }

    Ok(out)
}

// -----------------------------------------------------------------------
// Public API: reduce_axis (arbitrary axis for N-D arrays)
// -----------------------------------------------------------------------

/// Reduce along an arbitrary axis of an N-D array.
///
/// Strategy: reshape/transpose to make the target axis the last dimension,
/// then use `reduce_row` to reduce along it. The result is reshaped to
/// remove the reduced axis from the original shape.
///
/// For the last axis this is equivalent to `reduce_row`.
/// For axis 0 on a 2D array this is equivalent to `reduce_col`.
pub fn reduce_axis(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    axis: usize,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(KernelError::InvalidShape(format!(
            "reduce_axis: axis {} out of range for {}D array",
            axis, ndim
        )));
    }
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }

    let shape = input.shape();

    // Special case: if ndim <= 2, dispatch to optimized paths directly.
    if ndim == 1 {
        // Reducing the only axis is a global reduce.
        return reduce_all(registry, input, op, queue);
    }
    if ndim == 2 && axis == 1 {
        return reduce_row(registry, input, op, queue);
    }
    if ndim == 2 && axis == 0 {
        return reduce_col(registry, input, op, queue);
    }

    // General N-D case: flatten to 2D with the target axis as the last dim,
    // then use reduce_row. The outer dimensions are the product of all other
    // dimensions.
    //
    // If axis == last, just reshape to [outer, axis_size].
    // Otherwise, we must make the data contiguous with the target axis last
    // by performing a copy through a transposed view.

    let axis_size = shape[axis];
    let outer_size: usize = shape.iter().product::<usize>() / axis_size;

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let work_2d = if axis == ndim - 1 {
        // Target axis is already last -- just reshape.
        input.reshape(vec![outer_size, axis_size])?
    } else {
        // Build a permuted view: move `axis` to the end.
        // perm = [0, 1, ..., axis-1, axis+1, ..., ndim-1, axis]
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(axis);
        perm.push(axis);

        // Build the transposed shape and strides.
        let src_strides = input.strides();
        let new_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let new_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
        let transposed = input.view(new_shape.clone(), new_strides, input.offset());

        // The transposed view is non-contiguous; copy to make it contiguous.
        let contiguous = super::copy::copy(registry, &transposed, queue)?;
        contiguous.reshape(vec![outer_size, axis_size])?
    };

    let reduced_2d = reduce_row(registry, &work_2d, op, queue)?;

    // Build the output shape: original shape with the reduced axis removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    reduced_2d.reshape(out_shape)
}

// -----------------------------------------------------------------------
// Public API: unified reduce dispatcher
// -----------------------------------------------------------------------

/// Reduce an array along the given axis with the given operation.
///
/// This is the main entry point that dispatches to `reduce_all`, `reduce_row`,
/// `reduce_col`, or `reduce_axis` as appropriate.
pub fn reduce(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    axis: ReduceAxis,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    match axis {
        ReduceAxis::All => reduce_all(registry, input, op, queue),
        ReduceAxis::Row => reduce_row(registry, input, op, queue),
        ReduceAxis::Col => reduce_col(registry, input, op, queue),
        ReduceAxis::Axis(i) => reduce_axis(registry, input, op, i, queue),
    }
}

// -----------------------------------------------------------------------
// Public API: reduce_all_async
// -----------------------------------------------------------------------

/// Reduce all elements asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
///
/// Note: `ReduceOp::Mean` is not supported in async mode; use sync mode instead.
pub fn reduce_all_async(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<super::LaunchResult, KernelError> {
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }
    if matches!(op, ReduceOp::Mean) {
        return Err(KernelError::NotFound(
            "reduce_all_async does not support Mean; use reduce_all instead".into(),
        ));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let (kernel_name, out_dtype) = global_kernel_name(op, input.dtype())?;

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel_u32 = super::checked_u32(input.numel(), "numel")?;

    let out = Array::zeros(registry.device().raw(), &[1], out_dtype);

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&size_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();

    let handle = super::commit_with_mode(command_buffer, super::ExecMode::Async)
        .expect("async mode always returns a handle");

    Ok(super::LaunchResult::new(out, handle))
}

// -----------------------------------------------------------------------
// Convenience functions
// -----------------------------------------------------------------------

/// Convenience: sum all elements.
pub fn sum(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Sum, queue)
}

/// Convenience: max of all elements.
pub fn max(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Max, queue)
}

/// Convenience: min of all elements.
pub fn min(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Min, queue)
}

/// Convenience: mean of all elements.
pub fn mean(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Mean, queue)
}

/// Convenience: argmax of all elements.
pub fn argmax(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::ArgMax, queue)
}
