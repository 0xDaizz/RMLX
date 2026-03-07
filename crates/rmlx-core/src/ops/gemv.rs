//! General Matrix-Vector multiplication: y = A * x
//!
//! Optimised Metal kernels using SIMD shuffle reduction (`simd_sum`) and
//! multi-row (TM=4) / vectorised-load (TN=4) processing.
//!
//! Variants:
//! - `gemv_f32`  -- f32 matrix & vector, f32 accumulation
//! - `gemv_f16`  -- f16 matrix & vector, f32 accumulation, f16 output
//! - `gemv_bf16` -- bf16 matrix & vector, f32 accumulation, bf16 output
//! - `gemv_bm8_*` -- BM=8 variants: each simdgroup processes TM=4 rows independently
//! - `gemv_t_f32` / `gemv_t_f16` / `gemv_t_bf16` -- transposed: y = A^T @ x
//! - Fused bias variants: `gemv_bias_f32`, `gemv_bias_f16`, `gemv_bias_bf16`
//! - Fused bias BM=8 variants: `gemv_bias_bm8_f32`, `gemv_bias_bm8_f16`, `gemv_bias_bm8_bf16`

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// GEMV tuning — dimension-adaptive parameter selection
// ---------------------------------------------------------------------------

/// GEMV tuning parameters derived from matrix dimensions (MLX pattern).
#[derive(Debug, Clone, Copy)]
pub struct GemvTuning {
    pub sm: u64,
    pub sn: u64,
    pub bn: u64,
}

impl GemvTuning {
    /// Select GEMV tuning based on input/output vector dimensions.
    ///
    /// Follows the MLX dimension-adaptive pattern for optimal throughput.
    pub fn select(in_vec: usize, out_vec: usize) -> Self {
        if in_vec >= 8192 && out_vec >= 2048 {
            Self {
                sm: 4,
                sn: 8,
                bn: 16,
            }
        } else if out_vec >= 2048 {
            Self {
                sm: 8,
                sn: 4,
                bn: 16,
            }
        } else if out_vec >= 512 {
            Self {
                sm: 8,
                sn: 4,
                bn: 4,
            }
        } else {
            Self {
                sm: 8,
                sn: 4,
                bn: 2,
            }
        }
    }
}

/// Threadgroup size used for GEMV dispatch.
const GEMV_THREADGROUP_SIZE: u64 = 256;

/// Number of rows processed per threadgroup (tile-M).
const TM: u64 = 4;

/// Number of simdgroups per threadgroup for the BM=8 variant.
const BM8: u64 = 8;
/// Rows per threadgroup in BM=8 mode: BM8 * TM = 32.
const BM8_ROWS: u64 = BM8 * TM;
/// Minimum M to use BM=8 variant (need enough rows to fill threadgroups).
const BM8_THRESHOLD: u64 = 256;

pub const GEMV_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constant constexpr uint SIMD_SIZE = 32;
constant constexpr uint TM = 4;   // rows per threadgroup
constant constexpr uint BM8 = 8;  // simdgroups per threadgroup for bm8 variant

// ---------------------------------------------------------------------------
// gemv_f32:  y = A * x    (A: [M, K], x: [K], y: [M])
// ---------------------------------------------------------------------------
kernel void gemv_f32(
    device const float* mat  [[buffer(0)]],
    device const float* vec  [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  M    [[buffer(3)]],
    constant     uint&  K    [[buffer(4)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Vectorised loads (4 elements per iteration when possible)
    uint k4 = K / 4;
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
                acc[r] += dot(m4, v4);
            }
        }
    }
    // Handle remaining elements
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = vec[i];
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    // Intra-simdgroup reduction via simd_sum
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    // Cross-simdgroup reduction using shared memory
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gemv_f16:  y = A * x   (f16 storage, f32 accumulation, f16 output)
// ---------------------------------------------------------------------------
kernel void gemv_f16(
    device const half*  mat    [[buffer(0)]],
    device const half*  vec    [[buffer(1)]],
    device       half*  output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
                float4 m4 = float4(m4h);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = half(val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// gemv_bf16:  y = A * x   (bf16 storage, f32 accumulation, bf16 output)
// ---------------------------------------------------------------------------
kernel void gemv_bf16(
    device const bfloat*  mat    [[buffer(0)]],
    device const bfloat*  vec    [[buffer(1)]],
    device       bfloat*  output [[buffer(2)]],
    constant     uint&    M      [[buffer(3)]],
    constant     uint&    K      [[buffer(4)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // bfloat does not support vector loads wider than scalar on all HW,
    // so we fall back to scalar loads with f32 accumulation.
    for (uint i = tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = bfloat(val);
            }
        }
    }
}

// ===========================================================================
// BM=8 variants: each simdgroup processes TM=4 rows independently
// No cross-simdgroup barriers needed!
// ===========================================================================

kernel void gemv_bm8_f32(
    device const float* mat  [[buffer(0)]],
    device const float* vec  [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  M    [[buffer(3)]],
    constant     uint&  K    [[buffer(4)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup handles TM=4 rows independently
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = simd_lane_id; i < k4; i += SIMD_SIZE) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
                acc[r] += dot(m4, v4);
            }
        }
    }
    // Handle remaining elements
    for (uint i = k4 * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    // Intra-simdgroup reduction only — no barriers needed!
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = acc[r];
        }
    }
}

kernel void gemv_bm8_f16(
    device const half*  mat    [[buffer(0)]],
    device const half*  vec    [[buffer(1)]],
    device       half*  output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = simd_lane_id; i < k4; i += SIMD_SIZE) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
                float4 m4 = float4(m4h);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = half(acc[r]);
        }
    }
}

kernel void gemv_bm8_bf16(
    device const bfloat*  mat    [[buffer(0)]],
    device const bfloat*  vec    [[buffer(1)]],
    device       bfloat*  output [[buffer(2)]],
    constant     uint&    M      [[buffer(3)]],
    constant     uint&    K      [[buffer(4)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = bfloat(acc[r]);
        }
    }
}

// ===========================================================================
// Fused bias variants:  y = A * x + bias
// ===========================================================================

kernel void gemv_bias_f32(
    device const float* mat    [[buffer(0)]],
    device const float* vec    [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    device const float* bias   [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = vec[i];
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = val + bias[row_base + r];
            }
        }
    }
}

kernel void gemv_bias_f16(
    device const half*  mat    [[buffer(0)]],
    device const half*  vec    [[buffer(1)]],
    device       half*  output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    device const half*  bias   [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
                float4 m4 = float4(m4h);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = half(val + float(bias[row_base + r]));
            }
        }
    }
}

kernel void gemv_bias_bf16(
    device const bfloat*  mat    [[buffer(0)]],
    device const bfloat*  vec    [[buffer(1)]],
    device       bfloat*  output [[buffer(2)]],
    constant     uint&    M      [[buffer(3)]],
    constant     uint&    K      [[buffer(4)]],
    device const bfloat*  bias   [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint row_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (row_base + r) < M) {
                output[row_base + r] = bfloat(val + float(bias[row_base + r]));
            }
        }
    }
}

// ===========================================================================
// Fused bias BM=8 variants:  y = A * x + bias  (no cross-simdgroup barriers)
// ===========================================================================

kernel void gemv_bias_bm8_f32(
    device const float* mat    [[buffer(0)]],
    device const float* vec    [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    device const float* bias   [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = simd_lane_id; i < k4; i += SIMD_SIZE) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += mat[(row_base + r) * K + i] * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = acc[r] + bias[row_base + r];
        }
    }
}

kernel void gemv_bias_bm8_f16(
    device const half*  mat    [[buffer(0)]],
    device const half*  vec    [[buffer(1)]],
    device       half*  output [[buffer(2)]],
    constant     uint&  M      [[buffer(3)]],
    constant     uint&  K      [[buffer(4)]],
    device const half*  bias   [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = K / 4;
    for (uint i = simd_lane_id; i < k4; i += SIMD_SIZE) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
                float4 m4 = float4(m4h);
                acc[r] += dot(m4, v4);
            }
        }
    }
    for (uint i = k4 * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = half(acc[r] + float(bias[row_base + r]));
        }
    }
}

kernel void gemv_bias_bm8_bf16(
    device const bfloat*  mat    [[buffer(0)]],
    device const bfloat*  vec    [[buffer(1)]],
    device       bfloat*  output [[buffer(2)]],
    constant     uint&    M      [[buffer(3)]],
    constant     uint&    K      [[buffer(4)]],
    device const bfloat*  bias   [[buffer(5)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (row_base + r < M) {
                acc[r] += float(mat[(row_base + r) * K + i]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0 && (row_base + r) < M) {
            output[row_base + r] = bfloat(acc[r] + float(bias[row_base + r]));
        }
    }
}

// ===========================================================================
// Transposed variants:  y = A^T * x
// A: [M_a, K_a] stored row-major.  y[j] = sum_i A[i,j] * x[i]
// Here M = K_a (output length = number of columns of A)
//      K = M_a (reduction length = number of rows of A)
// Grid: ceil(N_out / TM) threadgroups, each handles TM output elements.
// ===========================================================================

kernel void gemv_t_f32(
    device const float* mat    [[buffer(0)]],
    device const float* vec    [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  N_out  [[buffer(3)]],   // number of columns of A (output length)
    constant     uint&  K_red  [[buffer(4)]],   // number of rows of A (reduction length)
    constant     uint&  lda    [[buffer(5)]],   // leading dimension (= N_out for row-major A)
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint col_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread strides across the reduction dimension (rows of A)
    for (uint i = tid; i < K_red; i += tgsize) {
        float v = vec[i];
        for (uint r = 0; r < TM; r++) {
            if (col_base + r < N_out) {
                acc[r] += mat[i * lda + col_base + r] * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (col_base + r) < N_out) {
                output[col_base + r] = val;
            }
        }
    }
}

kernel void gemv_t_f16(
    device const half*  mat    [[buffer(0)]],
    device const half*  vec    [[buffer(1)]],
    device       half*  output [[buffer(2)]],
    constant     uint&  N_out  [[buffer(3)]],
    constant     uint&  K_red  [[buffer(4)]],
    constant     uint&  lda    [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint col_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K_red; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (col_base + r < N_out) {
                acc[r] += float(mat[i * lda + col_base + r]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (col_base + r) < N_out) {
                output[col_base + r] = half(val);
            }
        }
    }
}

kernel void gemv_t_bf16(
    device const bfloat*  mat    [[buffer(0)]],
    device const bfloat*  vec    [[buffer(1)]],
    device       bfloat*  output [[buffer(2)]],
    constant     uint&    N_out  [[buffer(3)]],
    constant     uint&    K_red  [[buffer(4)]],
    constant     uint&    lda    [[buffer(5)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint tgsize        [[threads_per_threadgroup]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float local_sums[TM * SIMD_SIZE];

    uint col_base = tg_id * TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K_red; i += tgsize) {
        float v = float(vec[i]);
        for (uint r = 0; r < TM; r++) {
            if (col_base + r < N_out) {
                acc[r] += float(mat[i * lda + col_base + r]) * v;
            }
        }
    }

    for (uint r = 0; r < TM; r++) acc[r] = simd_sum(acc[r]);

    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        for (uint r = 0; r < TM; r++) local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0 && (col_base + r) < N_out) {
                output[col_base + r] = bfloat(val);
            }
        }
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gemv", GEMV_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Rust dispatch helpers
// ---------------------------------------------------------------------------

/// Compute ceil(a / b) for unsigned integers.
#[allow(clippy::manual_div_ceil)]
fn ceil_div(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

/// Matrix-vector multiply: y = A * x
/// - mat: [M, K]
/// - vec: [K]
/// - output: [M]
///
/// Supports Float32, Float16, and Bfloat16. For f16/bf16 the kernel
/// accumulates in f32 internally and writes the result in the original dtype.
pub fn gemv(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }

    let mat_contig = super::make_contiguous(mat, registry, queue)?;
    let mat = mat_contig.as_ref().unwrap_or(mat);
    let vec_contig = super::make_contiguous(vec, registry, queue)?;
    let vec = vec_contig.as_ref().unwrap_or(vec);

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bm8_f32",
        (DType::Float32, false) => "gemv_f32",
        (DType::Float16, true) => "gemv_bm8_f16",
        (DType::Float16, false) => "gemv_f16",
        (DType::Bfloat16, true) => "gemv_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::zeros(registry.device().raw(), &[m as usize], mat.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let m_buf = dev.new_buffer_with_data(&m as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&m_buf), 0);
    encoder.set_buffer(4, Some(&k_buf), 0);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, super::ExecMode::Sync);

    Ok(out)
}

/// Encode GEMV into an existing command buffer (no commit/wait).
/// mat: [M, K], vec: [K] → output: [M]
pub fn gemv_into_cb(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bm8_f32",
        (DType::Float32, false) => "gemv_f32",
        (DType::Float16, true) => "gemv_bm8_f16",
        (DType::Float16, false) => "gemv_f16",
        (DType::Bfloat16, true) => "gemv_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::uninit(registry.device().raw(), &[m as usize], mat.dtype());

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// Encode GEMV into an existing compute command encoder (no encoder create/end).
/// mat: [M, K], vec: [K] → output: [M]
/// Caller is responsible for creating and ending the encoder.
pub fn gemv_into_encoder(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bm8_f32",
        (DType::Float32, false) => "gemv_f32",
        (DType::Float16, true) => "gemv_bm8_f16",
        (DType::Float16, false) => "gemv_f16",
        (DType::Bfloat16, true) => "gemv_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::uninit(registry.device().raw(), &[m as usize], mat.dtype());

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

/// Encode GEMV with fused bias into an existing command buffer (no commit/wait).
/// mat: [M, K], vec: [K], bias: [M] → output: [M]
/// Inputs must already be contiguous.
pub fn gemv_bias_into_cb(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    bias: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if bias.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D bias, got {}D",
            bias.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }
    if mat.shape()[0] != bias.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "bias length must match M: {} vs {}",
            bias.shape()[0],
            mat.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }
    if bias.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, bias={:?}",
            mat.dtype(),
            bias.dtype()
        )));
    }

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bias_bm8_f32",
        (DType::Float32, false) => "gemv_bias_f32",
        (DType::Float16, true) => "gemv_bias_bm8_f16",
        (DType::Float16, false) => "gemv_bias_f16",
        (DType::Bfloat16, true) => "gemv_bias_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bias_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv_bias not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::uninit(registry.device().raw(), &[m as usize], mat.dtype());

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.set_buffer(5, Some(bias.metal_buffer()), bias.offset() as u64);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// Encode GEMV with fused bias into an existing compute command encoder (no encoder create/end).
/// mat: [M, K], vec: [K], bias: [M] → output: [M]
/// Caller is responsible for creating and ending the encoder.
pub fn gemv_bias_into_encoder(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    bias: &Array,
    encoder: &metal::ComputeCommandEncoderRef,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if bias.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D bias, got {}D",
            bias.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }
    if mat.shape()[0] != bias.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "bias length must match M: {} vs {}",
            bias.shape()[0],
            mat.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }
    if bias.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, bias={:?}",
            mat.dtype(),
            bias.dtype()
        )));
    }

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bias_bm8_f32",
        (DType::Float32, false) => "gemv_bias_f32",
        (DType::Float16, true) => "gemv_bias_bm8_f16",
        (DType::Float16, false) => "gemv_bias_f16",
        (DType::Bfloat16, true) => "gemv_bias_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bias_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv_bias not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::uninit(registry.device().raw(), &[m as usize], mat.dtype());

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.set_buffer(5, Some(bias.metal_buffer()), bias.offset() as u64);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );

    Ok(out)
}

/// Matrix-vector multiply with fused bias: y = A * x + bias
/// - mat: [M, K]
/// - vec: [K]
/// - bias: [M]
/// - output: [M]
pub fn gemv_bias(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    bias: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    if bias.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias requires 1D bias, got {}D",
            bias.ndim()
        )));
    }
    if mat.shape()[1] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "inner dimensions must match: {} vs {}",
            mat.shape()[1],
            vec.shape()[0]
        )));
    }
    if mat.shape()[0] != bias.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "bias length must match M: {} vs {}",
            bias.shape()[0],
            mat.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }
    if bias.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_bias: dtype mismatch: mat={:?}, bias={:?}",
            mat.dtype(),
            bias.dtype()
        )));
    }

    let mat_contig = super::make_contiguous(mat, registry, queue)?;
    let mat = mat_contig.as_ref().unwrap_or(mat);
    let vec_contig = super::make_contiguous(vec, registry, queue)?;
    let vec = vec_contig.as_ref().unwrap_or(vec);
    let bias_contig = super::make_contiguous(bias, registry, queue)?;
    let bias = bias_contig.as_ref().unwrap_or(bias);

    let m = super::checked_u32(mat.shape()[0], "M")?;
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;

    let kernel_name = match (mat.dtype(), use_bm8) {
        (DType::Float32, true) => "gemv_bias_bm8_f32",
        (DType::Float32, false) => "gemv_bias_f32",
        (DType::Float16, true) => "gemv_bias_bm8_f16",
        (DType::Float16, false) => "gemv_bias_f16",
        (DType::Bfloat16, true) => "gemv_bias_bm8_bf16",
        (DType::Bfloat16, false) => "gemv_bias_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv_bias not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let k = super::checked_u32(mat.shape()[1], "K")?;

    let out = Array::zeros(registry.device().raw(), &[m as usize], mat.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let m_buf = dev.new_buffer_with_data(&m as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&m_buf), 0);
    encoder.set_buffer(4, Some(&k_buf), 0);
    encoder.set_buffer(5, Some(bias.metal_buffer()), bias.offset() as u64);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, super::ExecMode::Sync);

    Ok(out)
}

/// Transposed matrix-vector multiply: y = A^T * x
/// - mat: [rows, cols] stored row-major
/// - vec: [rows]  (reduction dimension)
/// - output: [cols]
///
/// Computes y[j] = sum_i A[i, j] * x[i].
pub fn gemv_t(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    // For A^T @ x: A is [rows, cols], x must have length == rows
    if mat.shape()[0] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t: vector length must match matrix rows: {} vs {}",
            vec.shape()[0],
            mat.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }

    let mat_contig = super::make_contiguous(mat, registry, queue)?;
    let mat = mat_contig.as_ref().unwrap_or(mat);
    let vec_contig = super::make_contiguous(vec, registry, queue)?;
    let vec = vec_contig.as_ref().unwrap_or(vec);

    let kernel_name = match mat.dtype() {
        DType::Float32 => "gemv_t_f32",
        DType::Float16 => "gemv_t_f16",
        DType::Bfloat16 => "gemv_t_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv_t not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let rows = mat.shape()[0];
    let cols = mat.shape()[1];

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let n_out = super::checked_u32(cols, "N_out")?;
    let k_red = super::checked_u32(rows, "K_red")?;
    let lda = super::checked_u32(cols, "lda")?; // row-major leading dim = cols

    let out = Array::zeros(registry.device().raw(), &[cols], mat.dtype());

    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let n_buf = dev.new_buffer_with_data(&n_out as *const u32 as *const _, 4, opts);
    let k_buf = dev.new_buffer_with_data(&k_red as *const u32 as *const _, 4, opts);
    let lda_buf = dev.new_buffer_with_data(&lda as *const u32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&n_buf), 0);
    encoder.set_buffer(4, Some(&k_buf), 0);
    encoder.set_buffer(5, Some(&lda_buf), 0);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = ceil_div(n_out as u64, TM);
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, super::ExecMode::Sync);

    Ok(out)
}

/// Transposed matrix-vector multiply into existing command buffer: y = A^T * x
/// - mat: [rows, cols] stored row-major, must be contiguous
/// - vec: [rows], must be contiguous
/// - output: [cols]
pub fn gemv_t_into_cb(
    registry: &KernelRegistry,
    mat: &Array,
    vec: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    if mat.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t requires 2D matrix, got {}D",
            mat.ndim()
        )));
    }
    if vec.ndim() != 1 {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t requires 1D vector, got {}D",
            vec.ndim()
        )));
    }
    // For A^T @ x: A is [rows, cols], x must have length == rows
    if mat.shape()[0] != vec.shape()[0] {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t: vector length must match matrix rows: {} vs {}",
            vec.shape()[0],
            mat.shape()[0]
        )));
    }
    if vec.dtype() != mat.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "gemv_t: dtype mismatch: mat={:?}, vec={:?}",
            mat.dtype(),
            vec.dtype()
        )));
    }

    let kernel_name = match mat.dtype() {
        DType::Float32 => "gemv_t_f32",
        DType::Float16 => "gemv_t_f16",
        DType::Bfloat16 => "gemv_t_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gemv_t not supported for {:?}",
                mat.dtype()
            )))
        }
    };

    let rows = mat.shape()[0];
    let cols = mat.shape()[1];

    let pipeline = registry.get_pipeline(kernel_name, mat.dtype())?;
    let n_out = super::checked_u32(cols, "N_out")?;
    let k_red = super::checked_u32(rows, "K_red")?;
    let lda = super::checked_u32(cols, "lda")?; // row-major leading dim = cols

    let out = Array::uninit(registry.device().raw(), &[cols], mat.dtype());

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(mat.metal_buffer()), mat.offset() as u64);
    encoder.set_buffer(1, Some(vec.metal_buffer()), vec.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_bytes(3, 4, &n_out as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k_red as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(5, 4, &lda as *const u32 as *const std::ffi::c_void);

    let tg_size = std::cmp::min(
        GEMV_THREADGROUP_SIZE,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let num_threadgroups = ceil_div(n_out as u64, TM);
    encoder.dispatch_thread_groups(
        MTLSize::new(num_threadgroups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    encoder.end_encoding();

    Ok(out)
}

/// Build function constants for GEMV specialization.
pub fn gemv_constants(use_bias: bool) -> Vec<(u32, crate::kernels::FunctionConstantValue)> {
    use crate::kernels::FunctionConstantValue;
    vec![(200, FunctionConstantValue::Bool(use_bias))]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_tuning_large() {
        let t = GemvTuning::select(8192, 4096);
        assert_eq!(t.sm, 4);
        assert_eq!(t.sn, 8);
        assert_eq!(t.bn, 16);
    }

    #[test]
    fn test_gemv_tuning_medium() {
        let t = GemvTuning::select(4096, 2048);
        assert_eq!(t.sm, 8);
        assert_eq!(t.sn, 4);
        assert_eq!(t.bn, 16);
    }

    #[test]
    fn test_gemv_tuning_small() {
        let t = GemvTuning::select(512, 128);
        assert_eq!(t.sm, 8);
        assert_eq!(t.sn, 4);
        assert_eq!(t.bn, 2);
    }
}
