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
//! - Interleaved [M/TM, K, TM] BM=8 variants: `gemv_bm8_f32_interleaved`, `gemv_bm8_f16_interleaved`, `gemv_bm8_bf16_interleaved`
//! - Interleaved fused bias BM=8: `gemv_bias_bm8_f32_interleaved`, `gemv_bias_bm8_f16_interleaved`, `gemv_bias_bm8_bf16_interleaved`

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

// Uniform hint — on Metal 3.1+ (M3 and later), uses the real uniform<T> type
// to tell the compiler a value is warp-uniform (same across all threads).
// On older devices, falls back to a no-op.
#if __METAL_VERSION__ >= 310
template <typename T>
METAL_FUNC uniform<T> as_uniform(T val) {
    return make_uniform(val);
}
#else
template <typename T>
METAL_FUNC T as_uniform(T val) {
    return val;
}
#endif

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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Vectorised loads (4 elements per iteration when possible)
    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
            acc[r] += dot(m4, v4);
        }
    }
    // Handle remaining elements
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = vec[i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    // Intra-simdgroup reduction via simd_sum
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    // Cross-simdgroup reduction using shared memory
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
            float4 m4 = float4(m4h);
            acc[r] += dot(m4, v4);
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // bfloat does not support vector loads wider than scalar on all HW,
    // so we fall back to scalar loads with f32 accumulation.
    for (uint i = tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered: process 4×float4 (64 bytes) per iteration for bandwidth
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = *reinterpret_cast<device const float4*>(vec + idx);
        float4 v4b = *reinterpret_cast<device const float4*>(vec + idx + 4);
        float4 v4c = *reinterpret_cast<device const float4*>(vec + idx + 8);
        float4 v4d = *reinterpret_cast<device const float4*>(vec + idx + 12);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const float* row = mat + (row_base + r) * K + idx;
            float4 m4a = *reinterpret_cast<device const float4*>(row);
            float4 m4b = *reinterpret_cast<device const float4*>(row + 4);
            float4 m4c = *reinterpret_cast<device const float4*>(row + 8);
            float4 m4d = *reinterpret_cast<device const float4*>(row + 12);
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Handle remainder (K%16 > 0) in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = *reinterpret_cast<device const float4*>(vec + i);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + i);
            acc[r] += dot(m4, v4);
        }
    }
    // Handle scalar remainder (K%4 > 0)
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    // Intra-simdgroup reduction only — no barriers needed!
    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered f16: process 4×half4 (32 bytes) per iteration — matches f32 bandwidth
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = float4(*reinterpret_cast<device const half4*>(vec + idx));
        float4 v4b = float4(*reinterpret_cast<device const half4*>(vec + idx + 4));
        float4 v4c = float4(*reinterpret_cast<device const half4*>(vec + idx + 8));
        float4 v4d = float4(*reinterpret_cast<device const half4*>(vec + idx + 12));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const half* row = mat + (row_base + r) * K + idx;
            float4 m4a = float4(*reinterpret_cast<device const half4*>(row));
            float4 m4b = float4(*reinterpret_cast<device const half4*>(row + 4));
            float4 m4c = float4(*reinterpret_cast<device const half4*>(row + 8));
            float4 m4d = float4(*reinterpret_cast<device const half4*>(row + 12));
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Handle remainder (K%16 > 0) in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = float4(*reinterpret_cast<device const half4*>(vec + i));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + i));
            acc[r] += dot(m4, v4);
        }
    }
    // Handle scalar remainder (K%4 > 0)
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        float4 v4 = *reinterpret_cast<device const float4*>(vec + idx);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + idx);
            acc[r] += dot(m4, v4);
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = vec[i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint k4 = as_uniform(K / 4);
    for (uint i = tid; i < k4; i += tgsize) {
        uint idx = i * 4;
        half4 v4h = *reinterpret_cast<device const half4*>(vec + idx);
        float4 v4 = float4(v4h);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            half4 m4h = *reinterpret_cast<device const half4*>(mat + (row_base + r) * K + idx);
            float4 m4 = float4(m4h);
            acc[r] += dot(m4, v4);
        }
    }
    for (uint i = k4 * 4 + tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (row_base >= M) return;
    row_base = (row_base + TM <= M) ? row_base : M - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K; i += tgsize) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered: process 4×float4 (64 bytes) per iteration for bandwidth
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = *reinterpret_cast<device const float4*>(vec + idx);
        float4 v4b = *reinterpret_cast<device const float4*>(vec + idx + 4);
        float4 v4c = *reinterpret_cast<device const float4*>(vec + idx + 8);
        float4 v4d = *reinterpret_cast<device const float4*>(vec + idx + 12);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const float* row = mat + (row_base + r) * K + idx;
            float4 m4a = *reinterpret_cast<device const float4*>(row);
            float4 m4b = *reinterpret_cast<device const float4*>(row + 4);
            float4 m4c = *reinterpret_cast<device const float4*>(row + 8);
            float4 m4d = *reinterpret_cast<device const float4*>(row + 12);
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Handle remainder (K%16 > 0) in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = *reinterpret_cast<device const float4*>(vec + i);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = *reinterpret_cast<device const float4*>(mat + (row_base + r) * K + i);
            acc[r] += dot(m4, v4);
        }
    }
    // Handle scalar remainder (K%4 > 0)
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += mat[(row_base + r) * K + i] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Quad-buffered f16: process 4×half4 (32 bytes) per iteration — matches f32 bandwidth
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = float4(*reinterpret_cast<device const half4*>(vec + idx));
        float4 v4b = float4(*reinterpret_cast<device const half4*>(vec + idx + 4));
        float4 v4c = float4(*reinterpret_cast<device const half4*>(vec + idx + 8));
        float4 v4d = float4(*reinterpret_cast<device const half4*>(vec + idx + 12));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            device const half* row = mat + (row_base + r) * K + idx;
            float4 m4a = float4(*reinterpret_cast<device const half4*>(row));
            float4 m4b = float4(*reinterpret_cast<device const half4*>(row + 4));
            float4 m4c = float4(*reinterpret_cast<device const half4*>(row + 8));
            float4 m4d = float4(*reinterpret_cast<device const half4*>(row + 12));
            acc[r] += dot(m4a, v4a) + dot(m4b, v4b) + dot(m4c, v4c) + dot(m4d, v4d);
        }
    }
    // Handle remainder (K%16 > 0) in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = float4(*reinterpret_cast<device const half4*>(vec + i));
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float4 m4 = float4(*reinterpret_cast<device const half4*>(mat + (row_base + r) * K + i));
            acc[r] += dot(m4, v4);
        }
    }
    // Handle scalar remainder (K%4 > 0)
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    row_base = (row_base + TM <= M) ? row_base : M - TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[(row_base + r) * K + i]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r] + float(bias[row_base + r]));
        }
    }
}

// ===========================================================================
// Interleaved [M/TM, K, TM] BM=8 variants:  y = A_interleaved * x
// Weight matrix stored as mat[group * K * TM + k * TM + r] where group = row_base / TM
// 16 consecutive k-values x TM=4 rows = 64 halfs = 128 bytes = 1 GPU cache line
// ===========================================================================

kernel void gemv_bm8_f16_interleaved(
    device const half*  mat    [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
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
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // tile_base points to this group's interleaved tile
    device const half* tile_base = mat + (row_base / TM) * K * TM;

    // Quad-buffered: process 16 k-values per iteration
    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = float4(*reinterpret_cast<device const half4*>(vec + idx));
        float4 v4b = float4(*reinterpret_cast<device const half4*>(vec + idx + 4));
        float4 v4c = float4(*reinterpret_cast<device const half4*>(vec + idx + 8));
        float4 v4d = float4(*reinterpret_cast<device const half4*>(vec + idx + 12));

        // 16 k-values x TM=4 rows = 64 halfs = 128 bytes contiguous
        device const half* chunk = tile_base + idx * TM;

        #pragma clang loop unroll(full)
        for (uint sub = 0; sub < 4; sub++) {
            device const half* p = chunk + sub * 4 * TM;
            // Each half4 load gives [row0, row1, row2, row3] at one k-position
            half4 c0 = *reinterpret_cast<device const half4*>(p);
            half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
            half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
            half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
            float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
            acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
            acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
            acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
            acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
        }
    }
    // Remainder in groups of 4
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = float4(*reinterpret_cast<device const half4*>(vec + i));
        device const half* chunk = tile_base + i * TM;
        half4 c0 = *reinterpret_cast<device const half4*>(chunk);
        half4 c1 = *reinterpret_cast<device const half4*>(chunk + TM);
        half4 c2 = *reinterpret_cast<device const half4*>(chunk + 2 * TM);
        half4 c3 = *reinterpret_cast<device const half4*>(chunk + 3 * TM);
        acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
        acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
        acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
        acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
    }
    // Scalar remainder
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        device const half* p = tile_base + i * TM;
        acc[0] += float(p[0]) * v;
        acc[1] += float(p[1]) * v;
        acc[2] += float(p[2]) * v;
        acc[3] += float(p[3]) * v;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r]);
        }
    }
}

kernel void gemv_bm8_f32_interleaved(
    device const float* mat  [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
    device const float* vec  [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  M    [[buffer(3)]],
    constant     uint&  K    [[buffer(4)]],
    uint tg_id         [[threadgroup_position_in_grid]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint row_base = tg_id * (BM8 * TM) + simd_group_id * TM;
    if (row_base >= M) return;
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // tile_base points to this group's interleaved tile
    device const float* tile_base = mat + (row_base / TM) * K * TM;

    // Interleaved f32: each k-position has TM=4 floats contiguous (float4 load)
    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        float4 c = *reinterpret_cast<device const float4*>(tile_base + i * TM);
        acc[0] += c[0] * v;
        acc[1] += c[1] * v;
        acc[2] += c[2] * v;
        acc[3] += c[3] * v;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r];
        }
    }
}

kernel void gemv_bm8_bf16_interleaved(
    device const bfloat*  mat    [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
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
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // tile_base points to this group's interleaved tile
    device const bfloat* tile_base = mat + (row_base / TM) * K * TM;

    // bf16 scalar: process per-element with interleaved addressing
    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        device const bfloat* p = tile_base + i * TM;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(p[r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = bfloat(acc[r]);
        }
    }
}

// ===========================================================================
// Interleaved [M/TM, K, TM] fused bias BM=8 variants:  y = A_interleaved * x + bias
// ===========================================================================

kernel void gemv_bias_bm8_f16_interleaved(
    device const half*  mat    [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
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
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const half* tile_base = mat + (row_base / TM) * K * TM;

    uint k16 = as_uniform(K / 16);
    for (uint i = simd_lane_id; i < k16; i += SIMD_SIZE) {
        uint idx = i * 16;
        float4 v4a = float4(*reinterpret_cast<device const half4*>(vec + idx));
        float4 v4b = float4(*reinterpret_cast<device const half4*>(vec + idx + 4));
        float4 v4c = float4(*reinterpret_cast<device const half4*>(vec + idx + 8));
        float4 v4d = float4(*reinterpret_cast<device const half4*>(vec + idx + 12));

        device const half* chunk = tile_base + idx * TM;

        #pragma clang loop unroll(full)
        for (uint sub = 0; sub < 4; sub++) {
            device const half* p = chunk + sub * 4 * TM;
            half4 c0 = *reinterpret_cast<device const half4*>(p);
            half4 c1 = *reinterpret_cast<device const half4*>(p + TM);
            half4 c2 = *reinterpret_cast<device const half4*>(p + 2 * TM);
            half4 c3 = *reinterpret_cast<device const half4*>(p + 3 * TM);
            float4 v4 = (sub == 0) ? v4a : (sub == 1) ? v4b : (sub == 2) ? v4c : v4d;
            acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
            acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
            acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
            acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
        }
    }
    for (uint i = k16 * 16 + simd_lane_id * 4; i + 3 < K; i += SIMD_SIZE * 4) {
        float4 v4 = float4(*reinterpret_cast<device const half4*>(vec + i));
        device const half* chunk = tile_base + i * TM;
        half4 c0 = *reinterpret_cast<device const half4*>(chunk);
        half4 c1 = *reinterpret_cast<device const half4*>(chunk + TM);
        half4 c2 = *reinterpret_cast<device const half4*>(chunk + 2 * TM);
        half4 c3 = *reinterpret_cast<device const half4*>(chunk + 3 * TM);
        acc[0] += float(c0[0]) * v4[0] + float(c1[0]) * v4[1] + float(c2[0]) * v4[2] + float(c3[0]) * v4[3];
        acc[1] += float(c0[1]) * v4[0] + float(c1[1]) * v4[1] + float(c2[1]) * v4[2] + float(c3[1]) * v4[3];
        acc[2] += float(c0[2]) * v4[0] + float(c1[2]) * v4[1] + float(c2[2]) * v4[2] + float(c3[2]) * v4[3];
        acc[3] += float(c0[3]) * v4[0] + float(c1[3]) * v4[1] + float(c2[3]) * v4[2] + float(c3[3]) * v4[3];
    }
    for (uint i = (K / 4) * 4 + simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        device const half* p = tile_base + i * TM;
        acc[0] += float(p[0]) * v;
        acc[1] += float(p[1]) * v;
        acc[2] += float(p[2]) * v;
        acc[3] += float(p[3]) * v;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = half(acc[r] + float(bias[row_base + r]));
        }
    }
}

kernel void gemv_bias_bm8_f32_interleaved(
    device const float* mat    [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
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
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const float* tile_base = mat + (row_base / TM) * K * TM;

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = vec[i];
        float4 c = *reinterpret_cast<device const float4*>(tile_base + i * TM);
        acc[0] += c[0] * v;
        acc[1] += c[1] * v;
        acc[2] += c[2] * v;
        acc[3] += c[3] * v;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
            output[row_base + r] = acc[r] + bias[row_base + r];
        }
    }
}

kernel void gemv_bias_bm8_bf16_interleaved(
    device const bfloat*  mat    [[buffer(0)]],  // [M/TM, K, TM] interleaved layout
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
    if (row_base + TM > M) row_base = (M / TM - 1) * TM;

    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    device const bfloat* tile_base = mat + (row_base / TM) * K * TM;

    for (uint i = simd_lane_id; i < K; i += SIMD_SIZE) {
        float v = float(vec[i]);
        device const bfloat* p = tile_base + i * TM;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(p[r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
        if (simd_lane_id == 0) {
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
    if (col_base >= N_out) return;
    col_base = (col_base + TM <= N_out) ? col_base : N_out - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread strides across the reduction dimension (rows of A)
    for (uint i = tid; i < K_red; i += tgsize) {
        threadgroup_barrier(mem_flags::mem_none);
        float v = vec[i];
        float4 m4 = *reinterpret_cast<device const float4*>(mat + i * lda + col_base);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += m4[r] * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (col_base >= N_out) return;
    col_base = (col_base + TM <= N_out) ? col_base : N_out - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K_red; i += tgsize) {
        threadgroup_barrier(mem_flags::mem_none);
        float v = float(vec[i]);
        half4 m4h = *reinterpret_cast<device const half4*>(mat + i * lda + col_base);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(m4h[r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    if (col_base >= N_out) return;
    col_base = (col_base + TM <= N_out) ? col_base : N_out - TM;
    float acc[TM] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = tid; i < K_red; i += tgsize) {
        threadgroup_barrier(mem_flags::mem_none);
        float v = float(vec[i]);
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            acc[r] += float(mat[i * lda + col_base + r]) * v;
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < TM; r++) {
        acc[r] = simd_sum(acc[r]);
    }

    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_lane_id] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            local_sums[r * SIMD_SIZE + simd_group_id] = acc[r];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < TM; r++) {
            float val = simd_sum(local_sums[r * SIMD_SIZE + simd_lane_id]);
            if (simd_lane_id == 0) {
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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);
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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);
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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);

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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);
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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);

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
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        MTLSize::new(tg_size, 1, 1)
    };
    encoder.dispatch_thread_groups(MTLSize::new(num_threadgroups, 1, 1), tg_dim);
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

// ---------------------------------------------------------------------------
// Pre-resolved (zero-overhead) encoder helpers
// ---------------------------------------------------------------------------

/// Encode GEMV using a pre-resolved PSO and pre-allocated output buffer.
/// Skips all validation and allocation — caller must ensure correctness.
/// mat: [M, K], vec: [K] → writes into out_buf at out_offset
#[allow(clippy::too_many_arguments)]
pub fn gemv_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    mat_buf: &metal::BufferRef,
    mat_offset: u64,
    vec_buf: &metal::BufferRef,
    vec_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    m: u32,
    k: u32,
    grid: metal::MTLSize,
    tg: metal::MTLSize,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(mat_buf), mat_offset);
    encoder.set_buffer(1, Some(vec_buf), vec_offset);
    encoder.set_buffer(2, Some(out_buf), out_offset);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.dispatch_thread_groups(grid, tg);
}

/// Encode GEMV+bias using a pre-resolved PSO and pre-allocated output buffer.
#[allow(clippy::too_many_arguments)]
pub fn gemv_bias_preresolved_into_encoder(
    pso: &metal::ComputePipelineState,
    mat_buf: &metal::BufferRef,
    mat_offset: u64,
    vec_buf: &metal::BufferRef,
    vec_offset: u64,
    out_buf: &metal::BufferRef,
    out_offset: u64,
    m: u32,
    k: u32,
    bias_buf: &metal::BufferRef,
    bias_offset: u64,
    grid: metal::MTLSize,
    tg: metal::MTLSize,
    encoder: &metal::ComputeCommandEncoderRef,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(mat_buf), mat_offset);
    encoder.set_buffer(1, Some(vec_buf), vec_offset);
    encoder.set_buffer(2, Some(out_buf), out_offset);
    encoder.set_bytes(3, 4, &m as *const u32 as *const std::ffi::c_void);
    encoder.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
    encoder.set_buffer(5, Some(bias_buf), bias_offset);
    encoder.dispatch_thread_groups(grid, tg);
}

/// Compute dispatch grid and threadgroup sizes for GEMV with given M.
pub fn gemv_dispatch_sizes(
    m: u32,
    pso: &metal::ComputePipelineState,
) -> (metal::MTLSize, metal::MTLSize) {
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;
    let num_threadgroups = if use_bm8 {
        ceil_div(m as u64, BM8_ROWS)
    } else {
        ceil_div(m as u64, TM)
    };
    let tg_dim = if use_bm8 {
        MTLSize::new(32, 1, BM8)
    } else {
        let tg_size = std::cmp::min(
            GEMV_THREADGROUP_SIZE,
            pso.max_total_threads_per_threadgroup(),
        );
        MTLSize::new(tg_size, 1, 1)
    };
    (MTLSize::new(num_threadgroups, 1, 1), tg_dim)
}

/// Get the GEMV kernel name for a given dtype and M dimension.
pub fn gemv_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;
    match (dtype, use_bm8) {
        (DType::Float32, true) => Ok("gemv_bm8_f32"),
        (DType::Float32, false) => Ok("gemv_f32"),
        (DType::Float16, true) => Ok("gemv_bm8_f16"),
        (DType::Float16, false) => Ok("gemv_f16"),
        (DType::Bfloat16, true) => Ok("gemv_bm8_bf16"),
        (DType::Bfloat16, false) => Ok("gemv_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "gemv not supported for {:?}",
            dtype
        ))),
    }
}

/// Get the GEMV+bias kernel name for a given dtype and M dimension.
pub fn gemv_bias_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    let use_bm8 = (m as u64) >= BM8_THRESHOLD;
    match (dtype, use_bm8) {
        (DType::Float32, true) => Ok("gemv_bias_bm8_f32"),
        (DType::Float32, false) => Ok("gemv_bias_f32"),
        (DType::Float16, true) => Ok("gemv_bias_bm8_f16"),
        (DType::Float16, false) => Ok("gemv_bias_f16"),
        (DType::Bfloat16, true) => Ok("gemv_bias_bm8_bf16"),
        (DType::Bfloat16, false) => Ok("gemv_bias_bf16"),
        _ => Err(KernelError::NotFound(format!(
            "gemv_bias not supported for {:?}",
            dtype
        ))),
    }
}

/// Get the interleaved [M/TM, K, TM] GEMV kernel name (BM8 only, M >= 256).
pub fn gemv_interleaved_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    if (m as u64) < BM8_THRESHOLD {
        return Err(KernelError::NotFound(format!(
            "gemv_interleaved requires M >= {} (BM8), got {}",
            BM8_THRESHOLD, m
        )));
    }
    match dtype {
        DType::Float32 => Ok("gemv_bm8_f32_interleaved"),
        DType::Float16 => Ok("gemv_bm8_f16_interleaved"),
        DType::Bfloat16 => Ok("gemv_bm8_bf16_interleaved"),
        _ => Err(KernelError::NotFound(format!(
            "gemv_interleaved not supported for {:?}",
            dtype
        ))),
    }
}

/// Get the interleaved [M/TM, K, TM] GEMV+bias kernel name (BM8 only, M >= 256).
pub fn gemv_bias_interleaved_kernel_name(dtype: DType, m: u32) -> Result<&'static str, KernelError> {
    if (m as u64) < BM8_THRESHOLD {
        return Err(KernelError::NotFound(format!(
            "gemv_bias_interleaved requires M >= {} (BM8), got {}",
            BM8_THRESHOLD, m
        )));
    }
    match dtype {
        DType::Float32 => Ok("gemv_bias_bm8_f32_interleaved"),
        DType::Float16 => Ok("gemv_bias_bm8_f16_interleaved"),
        DType::Bfloat16 => Ok("gemv_bias_bm8_bf16_interleaved"),
        _ => Err(KernelError::NotFound(format!(
            "gemv_bias_interleaved not supported for {:?}",
            dtype
        ))),
    }
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
