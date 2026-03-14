//! RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
//!
//! ## Improvements over the baseline kernel
//!
//! - **N_READS=4 coalescing**: each thread processes 4 consecutive elements per
//!   iteration for better memory throughput.
//! - **Register caching**: input values read in Phase 1 are cached in registers
//!   so Phase 2 reads from cache instead of device memory (eliminates 2nd pass).
//! - **Weight stride support** (`w_stride`): handles non-contiguous weight tensors.
//! - **Optional weight** (`has_w`): when the weight pointer is null the
//!   multiplication is skipped (pure RMS normalisation).
//! - **uint32 overflow fix**: uses `size_t` for row-base addressing so that
//!   `row * axis_size` does not overflow 32 bits on large tensors.
//! - **f16 / bf16 support**: accumulation in f32, read/write in half / bfloat.
//! - **Single-row vs looped variants**: for `axis_size <= 1024` (i.e. <= tgsize)
//!   a simpler single-row kernel is selected; for larger sizes the looped variant is used.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use crate::ops::buffer_slots::{inv_rms, rms_norm as rms_slots, rms_norm_residual};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer as _;
use objc2_metal::MTLCommandQueue as _;
use objc2_metal::MTLComputePipelineState as _;
use rmlx_metal::ComputePass;
use rmlx_metal::MTLSize;

pub const RMS_NORM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ─── N_READS = 4 coalescing constant ──────────────────────────────────────
constant constexpr uint N_READS = 4;

// ─── Max elements cached per thread in registers ─────────────────────────
constant constexpr uint MAX_PER_THREAD = 64;

// ─── Shared reduction helper ──────────────────────────────────────────────
// Reduces per-thread `acc` across the threadgroup and writes 1/rms into
// `local_inv_rms[0]`.  Returns the inverse-rms value.
inline float reduce_sum_of_squares(
    float acc,
    uint axis_size,
    float eps,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup float* local_sums,
    threadgroup float* local_inv_rms)
{
    constexpr int SIMD_SIZE = 32;

    // Simdgroup reduction
    acc = simd_sum(acc);

    // Initialize shared memory
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write per-simdgroup partial sums
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroups
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_rms[0] = metal::precise::rsqrt(acc / float(axis_size) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return local_inv_rms[0];
}

// ═══════════════════════════════════════════════════════════════════════════
// f32 kernels
// ═══════════════════════════════════════════════════════════════════════════

// ─── Looped variant (arbitrary axis_size) ─────────────────────────────────
kernel void rms_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  axis_size [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    constant     uint&  w_stride [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares with N_READS coalescing + register caching ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = input[base + i];
            float v1 = input[base + i + 1];
            float v2 = input[base + i + 2];
            float v3 = input[base + i + 3];
            cached[n_cached++] = v0;
            cached[n_cached++] = v1;
            cached[n_cached++] = v2;
            cached[n_cached++] = v3;
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = input[base + j];
                cached[n_cached++] = v;
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= weight[i       * w_stride];
                o1 *= weight[(i + 1) * w_stride];
                o2 *= weight[(i + 2) * w_stride];
                o3 *= weight[(i + 3) * w_stride];
            }
            output[base + i]     = o0;
            output[base + i + 1] = o1;
            output[base + i + 2] = o2;
            output[base + i + 3] = o3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= weight[j * w_stride]; }
                output[base + j] = o;
            }
        }
    }
}

// ─── Single-row variant (axis_size <= 4096, no loop overhead) ─────────────
kernel void rms_norm_single_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint&  axis_size [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    constant     uint&  w_stride [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares (single pass, no outer loop) ──
    float acc = 0.0;
    float cached_val = 0.0;
    uint i = tid;
    if (i < axis_size) {
        float v = input[base + i];
        cached_val = v;
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached_val) ──
    if (i < axis_size) {
        float o = cached_val * rms;
        if (has_w) { o *= weight[i * w_stride]; }
        output[base + i] = o;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// f16 kernels (accumulate in f32, read/write half)
// ═══════════════════════════════════════════════════════════════════════════

kernel void rms_norm_f16(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device       half*  output  [[buffer(2)]],
    constant     uint&  axis_size [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    constant     uint&  w_stride [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares with register caching (cache as f32) ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            cached[n_cached++] = v0;
            cached[n_cached++] = v1;
            cached[n_cached++] = v2;
            cached[n_cached++] = v3;
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = float(input[base + j]);
                cached[n_cached++] = v;
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= float(weight[i       * w_stride]);
                o1 *= float(weight[(i + 1) * w_stride]);
                o2 *= float(weight[(i + 2) * w_stride]);
                o3 *= float(weight[(i + 3) * w_stride]);
            }
            output[base + i]     = half(o0);
            output[base + i + 1] = half(o1);
            output[base + i + 2] = half(o2);
            output[base + i + 3] = half(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= float(weight[j * w_stride]); }
                output[base + j] = half(o);
            }
        }
    }
}

kernel void rms_norm_single_f16(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device       half*  output  [[buffer(2)]],
    constant     uint&  axis_size [[buffer(3)]],
    constant     float& eps     [[buffer(4)]],
    constant     uint&  w_stride [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares (single pass, no outer loop) ──
    float acc = 0.0;
    float cached_val = 0.0;
    uint i = tid;
    if (i < axis_size) {
        float v = float(input[base + i]);
        cached_val = v;
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached_val) ──
    if (i < axis_size) {
        float o = cached_val * rms;
        if (has_w) { o *= float(weight[i * w_stride]); }
        output[base + i] = half(o);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// bf16 kernels (accumulate in f32, read/write bfloat)
// ═══════════════════════════════════════════════════════════════════════════

kernel void rms_norm_bf16(
    device const bfloat* input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device       bfloat* output  [[buffer(2)]],
    constant     uint&   axis_size [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    constant     uint&   w_stride [[buffer(5)]],
    constant     uint&   has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares with register caching (cache as f32) ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            cached[n_cached++] = v0;
            cached[n_cached++] = v1;
            cached[n_cached++] = v2;
            cached[n_cached++] = v3;
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = float(input[base + j]);
                cached[n_cached++] = v;
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= float(weight[i       * w_stride]);
                o1 *= float(weight[(i + 1) * w_stride]);
                o2 *= float(weight[(i + 2) * w_stride]);
                o3 *= float(weight[(i + 3) * w_stride]);
            }
            output[base + i]     = bfloat(o0);
            output[base + i + 1] = bfloat(o1);
            output[base + i + 2] = bfloat(o2);
            output[base + i + 3] = bfloat(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= float(weight[j * w_stride]); }
                output[base + j] = bfloat(o);
            }
        }
    }
}

kernel void rms_norm_single_bf16(
    device const bfloat* input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device       bfloat* output  [[buffer(2)]],
    constant     uint&   axis_size [[buffer(3)]],
    constant     float&  eps     [[buffer(4)]],
    constant     uint&   w_stride [[buffer(5)]],
    constant     uint&   has_w   [[buffer(6)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: sum of squares (single pass, no outer loop) ──
    float acc = 0.0;
    float cached_val = 0.0;
    uint i = tid;
    if (i < axis_size) {
        float v = float(input[base + i]);
        cached_val = v;
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached_val) ──
    if (i < axis_size) {
        float o = cached_val * rms;
        if (has_w) { o *= float(weight[i * w_stride]); }
        output[base + i] = bfloat(o);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// inv_rms kernels: compute only inv_rms[row] = 1/rms without normalization.
// Output: float32 inv_rms vector [rows], one value per row.
// Used for fused RMSNorm+GEMM where GEMM applies norm on-the-fly.
// ═══════════════════════════════════════════════════════════════════════════

kernel void inv_rms_f32(
    device const float* input    [[buffer(0)]],
    device       float* inv_out  [[buffer(1)]],
    constant     uint&  axis_size [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = input[base + i];
            float v1 = input[base + i + 1];
            float v2 = input[base + i + 2];
            float v3 = input[base + i + 3];
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = input[base + j];
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    if (tid == 0) {
        inv_out[row] = rms;
    }
}

kernel void inv_rms_f16(
    device const half*  input    [[buffer(0)]],
    device       float* inv_out  [[buffer(1)]],
    constant     uint&  axis_size [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = float(input[base + j]);
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    if (tid == 0) {
        inv_out[row] = rms;
    }
}

kernel void inv_rms_bf16(
    device const bfloat* input    [[buffer(0)]],
    device       float*  inv_out  [[buffer(1)]],
    constant     uint&   axis_size [[buffer(2)]],
    constant     float&  eps      [[buffer(3)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float v0 = float(input[base + i]);
            float v1 = float(input[base + i + 1]);
            float v2 = float(input[base + i + 2]);
            float v3 = float(input[base + i + 3]);
            acc += v0*v0 + v1*v1 + v2*v2 + v3*v3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float v = float(input[base + j]);
                acc += v * v;
            }
        }
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    if (tid == 0) {
        inv_out[row] = rms;
    }
}
"#;

/// Metal shader source for the fused RMSNorm + residual add kernel.
///
/// Computes `x = input[i] + residual[i]`, writes `x` back to `residual`
/// (updated skip connection), then writes `RMSNorm(x, weight, eps)` to `output`.
pub const RMS_NORM_RESIDUAL_ADD_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint N_READS = 4;

// ─── Max elements cached per thread in registers ─────────────────────────
constant constexpr uint MAX_PER_THREAD = 64;

// ─── Shared reduction helper (same as rms_norm) ─────────────────────────
inline float reduce_sum_of_squares_fused(
    float acc,
    uint axis_size,
    float eps,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup float* local_sums,
    threadgroup float* local_inv_rms)
{
    constexpr int SIMD_SIZE = 32;

    acc = simd_sum(acc);

    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_rms[0] = metal::precise::rsqrt(acc / float(axis_size) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return local_inv_rms[0];
}

// ═══════════════════════════════════════════════════════════════════════════
// f32 fused RMSNorm + residual add
// ═══════════════════════════════════════════════════════════════════════════

kernel void rms_norm_residual_add_f32(
    device const float* input    [[buffer(0)]],
    device       float* residual [[buffer(1)]],
    device const float* weight   [[buffer(2)]],
    device       float* output   [[buffer(3)]],
    constant     uint&  axis_size [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    constant     uint&  w_stride [[buffer(6)]],
    constant     uint&  has_w    [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: add residual + sum of squares + register caching ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float x0 = input[base + i]     + residual[base + i];
            float x1 = input[base + i + 1] + residual[base + i + 1];
            float x2 = input[base + i + 2] + residual[base + i + 2];
            float x3 = input[base + i + 3] + residual[base + i + 3];
            residual[base + i]     = x0;
            residual[base + i + 1] = x1;
            residual[base + i + 2] = x2;
            residual[base + i + 3] = x3;
            cached[n_cached++] = x0;
            cached[n_cached++] = x1;
            cached[n_cached++] = x2;
            cached[n_cached++] = x3;
            acc += x0*x0 + x1*x1 + x2*x2 + x3*x3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float x = input[base + j] + residual[base + j];
                residual[base + j] = x;
                cached[n_cached++] = x;
                acc += x * x;
            }
        }
    }

    float rms = reduce_sum_of_squares_fused(acc, axis_size, eps,
                                             simd_lane_id, simd_group_id,
                                             local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= weight[i       * w_stride];
                o1 *= weight[(i + 1) * w_stride];
                o2 *= weight[(i + 2) * w_stride];
                o3 *= weight[(i + 3) * w_stride];
            }
            output[base + i]     = o0;
            output[base + i + 1] = o1;
            output[base + i + 2] = o2;
            output[base + i + 3] = o3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= weight[j * w_stride]; }
                output[base + j] = o;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// f16 fused RMSNorm + residual add
// ═══════════════════════════════════════════════════════════════════════════

kernel void rms_norm_residual_add_f16(
    device const half*  input    [[buffer(0)]],
    device       half*  residual [[buffer(1)]],
    device const half*  weight   [[buffer(2)]],
    device       half*  output   [[buffer(3)]],
    constant     uint&  axis_size [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    constant     uint&  w_stride [[buffer(6)]],
    constant     uint&  has_w    [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: add residual + sum of squares + register caching (cache as f32) ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float x0 = float(input[base + i])     + float(residual[base + i]);
            float x1 = float(input[base + i + 1]) + float(residual[base + i + 1]);
            float x2 = float(input[base + i + 2]) + float(residual[base + i + 2]);
            float x3 = float(input[base + i + 3]) + float(residual[base + i + 3]);
            residual[base + i]     = half(x0);
            residual[base + i + 1] = half(x1);
            residual[base + i + 2] = half(x2);
            residual[base + i + 3] = half(x3);
            cached[n_cached++] = x0;
            cached[n_cached++] = x1;
            cached[n_cached++] = x2;
            cached[n_cached++] = x3;
            acc += x0*x0 + x1*x1 + x2*x2 + x3*x3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float x = float(input[base + j]) + float(residual[base + j]);
                residual[base + j] = half(x);
                cached[n_cached++] = x;
                acc += x * x;
            }
        }
    }

    float rms = reduce_sum_of_squares_fused(acc, axis_size, eps,
                                             simd_lane_id, simd_group_id,
                                             local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= float(weight[i       * w_stride]);
                o1 *= float(weight[(i + 1) * w_stride]);
                o2 *= float(weight[(i + 2) * w_stride]);
                o3 *= float(weight[(i + 3) * w_stride]);
            }
            output[base + i]     = half(o0);
            output[base + i + 1] = half(o1);
            output[base + i + 2] = half(o2);
            output[base + i + 3] = half(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= float(weight[j * w_stride]); }
                output[base + j] = half(o);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// bf16 fused RMSNorm + residual add
// ═══════════════════════════════════════════════════════════════════════════

kernel void rms_norm_residual_add_bf16(
    device const bfloat* input    [[buffer(0)]],
    device       bfloat* residual [[buffer(1)]],
    device const bfloat* weight   [[buffer(2)]],
    device       bfloat* output   [[buffer(3)]],
    constant     uint&   axis_size [[buffer(4)]],
    constant     float&  eps      [[buffer(5)]],
    constant     uint&   w_stride [[buffer(6)]],
    constant     uint&   has_w    [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Phase 1: add residual + sum of squares + register caching (cache as f32) ──
    float cached[MAX_PER_THREAD];
    uint n_cached = 0;
    float acc = 0.0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float x0 = float(input[base + i])     + float(residual[base + i]);
            float x1 = float(input[base + i + 1]) + float(residual[base + i + 1]);
            float x2 = float(input[base + i + 2]) + float(residual[base + i + 2]);
            float x3 = float(input[base + i + 3]) + float(residual[base + i + 3]);
            residual[base + i]     = bfloat(x0);
            residual[base + i + 1] = bfloat(x1);
            residual[base + i + 2] = bfloat(x2);
            residual[base + i + 3] = bfloat(x3);
            cached[n_cached++] = x0;
            cached[n_cached++] = x1;
            cached[n_cached++] = x2;
            cached[n_cached++] = x3;
            acc += x0*x0 + x1*x1 + x2*x2 + x3*x3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float x = float(input[base + j]) + float(residual[base + j]);
                residual[base + j] = bfloat(x);
                cached[n_cached++] = x;
                acc += x * x;
            }
        }
    }

    float rms = reduce_sum_of_squares_fused(acc, axis_size, eps,
                                             simd_lane_id, simd_group_id,
                                             local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight (read from cached[]) ──
    n_cached = 0;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = cached[n_cached++] * rms;
            float o1 = cached[n_cached++] * rms;
            float o2 = cached[n_cached++] * rms;
            float o3 = cached[n_cached++] * rms;
            if (has_w) {
                o0 *= float(weight[i       * w_stride]);
                o1 *= float(weight[(i + 1) * w_stride]);
                o2 *= float(weight[(i + 2) * w_stride]);
                o3 *= float(weight[(i + 3) * w_stride]);
            }
            output[base + i]     = bfloat(o0);
            output[base + i + 1] = bfloat(o1);
            output[base + i + 2] = bfloat(o2);
            output[base + i + 3] = bfloat(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = cached[n_cached++] * rms;
                if (has_w) { o *= float(weight[j * w_stride]); }
                output[base + j] = bfloat(o);
            }
        }
    }
}
"#;

/// Threshold: axis_size <= this uses the simpler single-row kernel.
const SINGLE_ROW_THRESHOLD: usize = 1024;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("rms_norm", RMS_NORM_SHADER_SOURCE)?;
    registry.register_jit_source("rms_norm_residual_add", RMS_NORM_RESIDUAL_ADD_SHADER_SOURCE)
}

/// Return the kernel name for the given dtype and axis_size.
fn rms_kernel_name(dtype: DType, axis_size: usize) -> Result<&'static str, KernelError> {
    let single = axis_size <= SINGLE_ROW_THRESHOLD;
    match (dtype, single) {
        (DType::Float32, false) => Ok("rms_norm_f32"),
        (DType::Float32, true) => Ok("rms_norm_single_f32"),
        (DType::Float16, false) => Ok("rms_norm_f16"),
        (DType::Float16, true) => Ok("rms_norm_single_f16"),
        (DType::Bfloat16, false) => Ok("rms_norm_bf16"),
        (DType::Bfloat16, true) => Ok("rms_norm_single_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rms_norm not supported for {:?}",
            dtype
        ))),
    }
}

/// Apply RMS normalization: y = x * rsqrt(mean(x^2) + eps) * weight.
///
/// - `input` shape: `[rows, axis_size]` (2-D).
/// - `weight` shape: `[axis_size]` (1-D), or `None` for weight-free normalisation.
/// - `eps`: small constant for numerical stability.
pub fn rms_norm(
    registry: &KernelRegistry,
    input: &Array,
    weight: &Array,
    eps: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    rms_norm_opt(registry, input, Some(weight), eps, queue)
}

/// Apply RMS normalization with an optional weight tensor.
///
/// When `weight` is `None`, the kernel skips the weight multiplication
/// (pure RMS normalisation).
pub fn rms_norm_opt(
    registry: &KernelRegistry,
    input: &Array,
    weight: Option<&Array>,
    eps: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let axis_size_usize = input.shape()[1];

    // Validate weight if present
    let (has_w, w_stride): (u32, u32) = if let Some(w) = weight {
        if w.ndim() != 1 {
            return Err(KernelError::InvalidShape(format!(
                "rms_norm requires 1D weight, got {}D",
                w.ndim()
            )));
        }
        if w.shape()[0] != axis_size_usize {
            return Err(KernelError::InvalidShape(format!(
                "axis size mismatch: input[1]={} vs weight[0]={}",
                axis_size_usize,
                w.shape()[0]
            )));
        }
        // Compute the stride of the weight's single dimension.
        // For a contiguous 1-D tensor this is 1.
        let ws = w.strides()[0] as u32;
        (1, ws)
    } else {
        (0, 1)
    };

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    // Weight: make contiguous only if stride != 1 AND weight is present.
    // When w_stride != 1 we keep the original buffer and pass the stride.
    // When stride == 1 we can still make_contiguous for safety.
    let weight_contig = weight
        .map(|w| super::make_contiguous(w, registry, queue))
        .transpose()?;
    let weight_resolved: Option<&Array> = match (&weight_contig, weight) {
        (Some(Some(c)), _) => Some(c),
        (_, Some(w)) => Some(w),
        _ => None,
    };

    // Re-derive w_stride from the resolved weight (may have been copied to contiguous).
    let w_stride: u32 = if let Some(w) = weight_resolved {
        w.strides()[0] as u32
    } else {
        w_stride
    };

    let kernel_name = rms_kernel_name(input.dtype(), axis_size_usize)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset());
    // For weight buffer: when has_w == 0 we still bind the input buffer as a
    // dummy (the kernel will never read from it).
    if let Some(w) = weight_resolved {
        encoder.set_buffer(1, Some(w.metal_buffer()), w.offset());
    } else {
        encoder.set_buffer(1, Some(input.metal_buffer()), 0);
    }
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_val(3, &axis_size);
    encoder.set_val(4, &eps);
    encoder.set_val(5, &w_stride);
    encoder.set_val(6, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows as usize,
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

// ---------------------------------------------------------------------------
// Into-CB variant (encode into existing command buffer, no commit/wait)
// ---------------------------------------------------------------------------

/// Encode RMS norm into an existing command buffer (no commit/wait).
pub fn rms_norm_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    weight: Option<&Array>,
    eps: f32,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let axis_size_usize = input.shape()[1];

    let (has_w, w_stride): (u32, u32) = if let Some(w) = weight {
        if w.ndim() != 1 {
            return Err(KernelError::InvalidShape(format!(
                "rms_norm requires 1D weight, got {}D",
                w.ndim()
            )));
        }
        if w.shape()[0] != axis_size_usize {
            return Err(KernelError::InvalidShape(format!(
                "axis size mismatch: input[1]={} vs weight[0]={}",
                axis_size_usize,
                w.shape()[0]
            )));
        }
        let ws = w.strides()[0] as u32;
        (1, ws)
    } else {
        (0, 1)
    };

    let kernel_name = rms_kernel_name(input.dtype(), axis_size_usize)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        rms_slots::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    if let Some(w) = weight {
        encoder.set_buffer(rms_slots::WEIGHT as u32, Some(w.metal_buffer()), w.offset());
    } else {
        encoder.set_buffer(rms_slots::WEIGHT as u32, Some(input.metal_buffer()), 0);
    }
    encoder.set_buffer(rms_slots::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(rms_slots::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_slots::EPS as u32, &eps);
    encoder.set_val(rms_slots::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_slots::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows,
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

    Ok(out)
}

/// Encode RMS normalization into an existing compute command encoder (no encoder create/end).
/// Caller is responsible for creating and ending the encoder.
pub fn rms_norm_into_encoder(
    registry: &KernelRegistry,
    input: &Array,
    weight: Option<&Array>,
    eps: f32,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let axis_size_usize = input.shape()[1];

    let (has_w, w_stride): (u32, u32) = if let Some(w) = weight {
        if w.ndim() != 1 {
            return Err(KernelError::InvalidShape(format!(
                "rms_norm requires 1D weight, got {}D",
                w.ndim()
            )));
        }
        if w.shape()[0] != axis_size_usize {
            return Err(KernelError::InvalidShape(format!(
                "axis size mismatch: input[1]={} vs weight[0]={}",
                axis_size_usize,
                w.shape()[0]
            )));
        }
        let ws = w.strides()[0] as u32;
        (1, ws)
    } else {
        (0, 1)
    };

    let kernel_name = rms_kernel_name(input.dtype(), axis_size_usize)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        rms_slots::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    if let Some(w) = weight {
        encoder.set_buffer(rms_slots::WEIGHT as u32, Some(w.metal_buffer()), w.offset());
    } else {
        encoder.set_buffer(rms_slots::WEIGHT as u32, Some(input.metal_buffer()), 0);
    }
    encoder.set_buffer(rms_slots::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(rms_slots::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_slots::EPS as u32, &eps);
    encoder.set_val(rms_slots::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_slots::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );

    Ok(out)
}

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode RMS norm using a pre-resolved PSO and pre-allocated output buffer.
/// Skips all validation — caller must ensure correctness.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_preresolved_into_encoder(
    pso: &rmlx_metal::MtlPipeline,
    input_buf: &rmlx_metal::MtlBuffer,
    input_offset: usize,
    weight_buf: &rmlx_metal::MtlBuffer,
    weight_offset: usize,
    out_buf: &rmlx_metal::MtlBuffer,
    out_offset: usize,
    axis_size: u32,
    eps: f32,
    w_stride: u32,
    has_w: u32,
    rows: u64,
    encoder: ComputePass<'_>,
) {
    encoder.set_pipeline(pso);
    encoder.set_buffer(rms_slots::INPUT as u32, Some(input_buf), input_offset);
    encoder.set_buffer(rms_slots::WEIGHT as u32, Some(weight_buf), weight_offset);
    encoder.set_buffer(rms_slots::OUT as u32, Some(out_buf), out_offset);
    encoder.set_val(rms_slots::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_slots::EPS as u32, &eps);
    encoder.set_val(rms_slots::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_slots::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pso.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
}

/// Get the RMS norm kernel name for a given dtype and axis_size.
pub fn rms_norm_kernel_name_for(
    dtype: DType,
    axis_size: usize,
) -> Result<&'static str, KernelError> {
    rms_kernel_name(dtype, axis_size)
}

// ---------------------------------------------------------------------------
// inv_rms: compute only the inverse-RMS per row (no normalization output)
// ---------------------------------------------------------------------------

fn inv_rms_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("inv_rms_f32"),
        DType::Float16 => Ok("inv_rms_f16"),
        DType::Bfloat16 => Ok("inv_rms_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "inv_rms not supported for {:?}",
            dtype
        ))),
    }
}

/// Compute per-row inverse-RMS: `inv_rms[row] = rsqrt(mean(x[row,:]^2) + eps)`.
///
/// Returns a 1-D f32 array of shape `[rows]`.
/// Used internally for fused RMSNorm+GEMM (the GEMM kernel applies norm on-the-fly).
pub fn compute_inv_rms(
    registry: &KernelRegistry,
    input: &Array,
    eps: f32,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "compute_inv_rms requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(input.shape()[1], "axis_size")?;

    let kernel_name = inv_rms_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let dev = registry.device().raw();
    let out = Array::uninit(dev, &[rows], DType::Float32);

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let enc = ComputePass::new(&raw_enc);
    enc.set_pipeline(&pipeline);
    enc.set_buffer(
        inv_rms::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    enc.set_buffer(inv_rms::OUT as u32, Some(out.metal_buffer()), 0);
    enc.set_val(inv_rms::AXIS_SIZE as u32, &axis_size);
    enc.set_val(inv_rms::EPS as u32, &eps);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    enc.dispatch_threadgroups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
    enc.end();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Fused RMSNorm + residual add
// ---------------------------------------------------------------------------

/// Return the fused kernel name for the given dtype.
fn rms_residual_add_kernel_name(dtype: DType) -> Result<&'static str, KernelError> {
    match dtype {
        DType::Float32 => Ok("rms_norm_residual_add_f32"),
        DType::Float16 => Ok("rms_norm_residual_add_f16"),
        DType::Bfloat16 => Ok("rms_norm_residual_add_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "rms_norm_residual_add not supported for {:?}",
            dtype
        ))),
    }
}

/// Fused RMSNorm + residual add:
///   x = input + residual
///   output = RMSNorm(x, weight, eps)
///   residual (updated in-place) = x
///
/// Returns `(normalized_output, updated_residual)`.
///
/// - `input` shape: `[rows, axis_size]` (2-D).
/// - `residual` shape: must match `input` exactly.
/// - `weight` shape: `[axis_size]` (1-D).
/// - `eps`: small constant for numerical stability.
///
/// The residual buffer is **mutated in-place** and returned as the second
/// element of the tuple for convenience.
pub fn rms_norm_residual_add(
    registry: &KernelRegistry,
    input: &Array,
    residual: &Array,
    weight: &Array,
    eps: f32,
    queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
) -> Result<(Array, Array), KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 2D input, got {}D",
            input.ndim()
        )));
    }
    if residual.shape() != input.shape() {
        return Err(KernelError::InvalidShape(format!(
            "residual shape {:?} does not match input shape {:?}",
            residual.shape(),
            input.shape()
        )));
    }
    if weight.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 1D weight, got {}D",
            weight.ndim()
        )));
    }
    let axis_size_usize = input.shape()[1];
    if weight.shape()[0] != axis_size_usize {
        return Err(KernelError::InvalidShape(format!(
            "axis size mismatch: input[1]={} vs weight[0]={}",
            axis_size_usize,
            weight.shape()[0]
        )));
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtype mismatch: input={:?}, residual={:?}, weight={:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        )));
    }

    // Ensure contiguous layouts
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    // We always need a writable copy of the residual because the kernel
    // mutates it in-place (stores input + residual back).  `copy::copy`
    // produces a fresh contiguous buffer regardless.
    let residual_buf = super::copy::copy(registry, residual, queue)?;

    let weight_contig = super::make_contiguous(weight, registry, queue)?;
    let weight = weight_contig.as_ref().unwrap_or(weight);

    let w_stride: u32 = weight.strides()[0] as u32;
    let has_w: u32 = 1;

    let kernel_name = rms_residual_add_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let command_buffer = queue.commandBuffer().unwrap();
    let raw_enc = command_buffer.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        rms_norm_residual::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::RESIDUAL as u32,
        Some(residual_buf.metal_buffer()),
        residual_buf.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::WEIGHT as u32,
        Some(weight.metal_buffer()),
        weight.offset(),
    );
    encoder.set_buffer(rms_norm_residual::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(rms_norm_residual::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_norm_residual::EPS as u32, &eps);
    encoder.set_val(rms_norm_residual::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_norm_residual::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows as usize,
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

    Ok((out, residual_buf))
}

/// CB-based fused RMSNorm + residual add for pipeline use.
///
/// Like [`rms_norm_residual_add`] but encodes into an existing command buffer
/// instead of creating and committing its own.  The caller manages the CB
/// lifecycle.
///
/// **IMPORTANT**: The residual is copied before mutation.  The returned
/// `residual_buf` contains `input + residual` (the updated hidden state *h*).
///
/// All inputs must be contiguous (which they are in the prefill pipeline where
/// every array comes from GEMM / attention output).
pub fn rms_norm_residual_add_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    residual: &Array,
    weight: &Array,
    eps: f32,
    cb: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
) -> Result<(Array, Array), KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 2D input, got {}D",
            input.ndim()
        )));
    }
    if residual.shape() != input.shape() {
        return Err(KernelError::InvalidShape(format!(
            "residual shape {:?} does not match input shape {:?}",
            residual.shape(),
            input.shape()
        )));
    }
    if weight.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 1D weight, got {}D",
            weight.ndim()
        )));
    }
    let axis_size_usize = input.shape()[1];
    if weight.shape()[0] != axis_size_usize {
        return Err(KernelError::InvalidShape(format!(
            "axis size mismatch: input[1]={} vs weight[0]={}",
            axis_size_usize,
            weight.shape()[0]
        )));
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtype mismatch: input={:?}, residual={:?}, weight={:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        )));
    }

    // In the prefill pipeline all arrays are contiguous (GEMM/attention output).
    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_into_cb: input must be contiguous".into(),
        ));
    }
    if !residual.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_into_cb: residual must be contiguous".into(),
        ));
    }
    if !weight.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_into_cb: weight must be contiguous".into(),
        ));
    }

    // Fresh copy of residual — the kernel mutates it in-place.
    let residual_buf = super::copy::copy_into_cb(registry, residual, cb)?;

    let w_stride: u32 = weight.strides()[0] as u32;
    let has_w: u32 = 1;

    let kernel_name = rms_residual_add_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    let raw_enc = cb.computeCommandEncoder().unwrap();
    let encoder = ComputePass::new(&raw_enc);
    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        rms_norm_residual::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::RESIDUAL as u32,
        Some(residual_buf.metal_buffer()),
        residual_buf.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::WEIGHT as u32,
        Some(weight.metal_buffer()),
        weight.offset(),
    );
    encoder.set_buffer(rms_norm_residual::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(rms_norm_residual::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_norm_residual::EPS as u32, &eps);
    encoder.set_val(rms_norm_residual::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_norm_residual::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows,
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

    Ok((out, residual_buf))
}

// ---------------------------------------------------------------------------
// _encode variants — accept &ComputeCommandEncoderRef instead of &CommandBufferRef
// ---------------------------------------------------------------------------

/// Alias for [`rms_norm_into_encoder`] — encode RMS norm into an existing
/// compute command encoder (no encoder create/end).
pub fn rms_norm_encode(
    registry: &KernelRegistry,
    input: &Array,
    weight: Option<&Array>,
    eps: f32,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    rms_norm_into_encoder(registry, input, weight, eps, encoder)
}

/// Encode fused residual-add + RMS norm into an existing compute command encoder.
///
/// Unlike [`rms_norm_residual_add_into_cb`], this variant dispatches the copy
/// and the norm kernel using the *same* encoder the caller provides.
///
/// Returns `(normed, updated_residual)`.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_residual_add_encode(
    registry: &KernelRegistry,
    input: &Array,
    residual: &Array,
    weight: &Array,
    eps: f32,
    encoder: ComputePass<'_>,
) -> Result<(Array, Array), KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 2D input, got {}D",
            input.ndim()
        )));
    }
    if residual.shape() != input.shape() {
        return Err(KernelError::InvalidShape(format!(
            "residual shape {:?} does not match input shape {:?}",
            residual.shape(),
            input.shape()
        )));
    }
    if weight.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm_residual_add requires 1D weight, got {}D",
            weight.ndim()
        )));
    }
    let axis_size_usize = input.shape()[1];
    if weight.shape()[0] != axis_size_usize {
        return Err(KernelError::InvalidShape(format!(
            "axis size mismatch: input[1]={} vs weight[0]={}",
            axis_size_usize,
            weight.shape()[0]
        )));
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "dtype mismatch: input={:?}, residual={:?}, weight={:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        )));
    }
    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_encode: input must be contiguous".into(),
        ));
    }
    if !residual.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_encode: residual must be contiguous".into(),
        ));
    }
    if !weight.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "rms_norm_residual_add_encode: weight must be contiguous".into(),
        ));
    }

    // Fresh copy of residual — the kernel mutates it in-place.
    let residual_buf = super::copy::copy_encode(registry, residual, encoder)?;

    let w_stride: u32 = weight.strides()[0] as u32;
    let has_w: u32 = 1;

    let kernel_name = rms_residual_add_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());

    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        rms_norm_residual::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::RESIDUAL as u32,
        Some(residual_buf.metal_buffer()),
        residual_buf.offset(),
    );
    encoder.set_buffer(
        rms_norm_residual::WEIGHT as u32,
        Some(weight.metal_buffer()),
        weight.offset(),
    );
    encoder.set_buffer(rms_norm_residual::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(rms_norm_residual::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(rms_norm_residual::EPS as u32, &eps);
    encoder.set_val(rms_norm_residual::W_STRIDE as u32, &w_stride);
    encoder.set_val(rms_norm_residual::HAS_W as u32, &has_w);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );

    Ok((out, residual_buf))
}

/// Encode inv_rms computation into an existing compute command encoder.
///
/// Same as [`compute_inv_rms`] but dispatches into a caller-provided encoder
/// instead of creating its own.  The caller is responsible for memory barriers
/// and encoder lifecycle.
///
/// Returns a 1-D f32 array of shape `[rows]`.
pub fn compute_inv_rms_encode(
    registry: &KernelRegistry,
    input: &Array,
    eps: f32,
    encoder: ComputePass<'_>,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "compute_inv_rms_encode requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(input.shape()[1], "axis_size")?;

    let kernel_name = inv_rms_kernel_name(input.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;

    let dev = registry.device().raw();
    let out = Array::uninit(dev, &[rows], DType::Float32);

    encoder.set_pipeline(&pipeline);
    encoder.set_buffer(
        inv_rms::INPUT as u32,
        Some(input.metal_buffer()),
        input.offset(),
    );
    encoder.set_buffer(inv_rms::OUT as u32, Some(out.metal_buffer()), 0);
    encoder.set_val(inv_rms::AXIS_SIZE as u32, &axis_size);
    encoder.set_val(inv_rms::EPS as u32, &eps);
    let tg_size = std::cmp::min(1024, pipeline.maxTotalThreadsPerThreadgroup());
    encoder.dispatch_threadgroups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );

    Ok(out)
}
