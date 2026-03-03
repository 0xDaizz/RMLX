//! Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
//!
//! Two-pass approach:
//! - Pass 1: compute mean and variance per row
//! - Pass 2: normalize, scale, and shift
//!
//! Reuses the simdgroup reduction pattern from RMS norm.
//! Supports f32, f16, and bf16 (f16/bf16 accumulate in f32).

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// Metal shader for Layer Normalization.
///
/// Kernels:
/// - `layer_norm_f32`: f32 read/write
/// - `layer_norm_f16`: f16 read/write, f32 accumulation
/// - `layer_norm_bf16`: bf16 read/write, f32 accumulation
///
/// Each threadgroup processes one row. Two-pass within a single kernel launch:
/// 1. Compute mean and variance via cooperative reduction
/// 2. Normalize and apply affine transform
pub const LAYER_NORM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint N_READS = 4;

// ─── Shared reduction helper ──────────────────────────────────────────────

// Reduce a float value across the threadgroup using simdgroup primitives.
inline float tg_reduce_sum_ln(
    float val,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup float* buf)
{
    constexpr int SIMD_SIZE = 32;
    val = simd_sum(val);
    if (simd_group_id == 0) {
        buf[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        buf[simd_group_id] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        val = simd_sum(buf[simd_lane_id]);
        if (simd_lane_id == 0) buf[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return buf[0];
}

// ═══════════════════════════════════════════════════════════════════════════
// f32 LayerNorm
// ═══════════════════════════════════════════════════════════════════════════

kernel void layer_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* bias    [[buffer(2)]],
    device       float* output  [[buffer(3)]],
    constant     uint&  axis_size [[buffer(4)]],
    constant     float& eps     [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    constant     uint&  has_b   [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float reduce_buf[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    // ── Pass 1a: compute mean ──
    float sum_val = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            sum_val += input[base + i] + input[base + i + 1]
                     + input[base + i + 2] + input[base + i + 3];
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                sum_val += input[base + j];
            }
        }
    }
    float mean = tg_reduce_sum_ln(sum_val, simd_lane_id, simd_group_id, reduce_buf)
                 / float(axis_size);

    // ── Pass 1b: compute variance ──
    float var_acc = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float d0 = input[base + i] - mean;
            float d1 = input[base + i + 1] - mean;
            float d2 = input[base + i + 2] - mean;
            float d3 = input[base + i + 3] - mean;
            var_acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float d = input[base + j] - mean;
                var_acc += d * d;
            }
        }
    }
    float variance = tg_reduce_sum_ln(var_acc, simd_lane_id, simd_group_id, reduce_buf)
                     / float(axis_size);
    float inv_std = rsqrt(variance + eps);

    // ── Pass 2: normalize + affine ──
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = (input[base + i]     - mean) * inv_std;
            float o1 = (input[base + i + 1] - mean) * inv_std;
            float o2 = (input[base + i + 2] - mean) * inv_std;
            float o3 = (input[base + i + 3] - mean) * inv_std;
            if (has_w) {
                o0 *= weight[i];     o1 *= weight[i + 1];
                o2 *= weight[i + 2]; o3 *= weight[i + 3];
            }
            if (has_b) {
                o0 += bias[i];     o1 += bias[i + 1];
                o2 += bias[i + 2]; o3 += bias[i + 3];
            }
            output[base + i]     = o0;
            output[base + i + 1] = o1;
            output[base + i + 2] = o2;
            output[base + i + 3] = o3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = (input[base + j] - mean) * inv_std;
                if (has_w) { o *= weight[j]; }
                if (has_b) { o += bias[j]; }
                output[base + j] = o;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// f16 LayerNorm (accumulate in f32, read/write half)
// ═══════════════════════════════════════════════════════════════════════════

kernel void layer_norm_f16(
    device const half*  input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device const half*  bias    [[buffer(2)]],
    device       half*  output  [[buffer(3)]],
    constant     uint&  axis_size [[buffer(4)]],
    constant     float& eps     [[buffer(5)]],
    constant     uint&  has_w   [[buffer(6)]],
    constant     uint&  has_b   [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float reduce_buf[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float sum_val = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            sum_val += float(input[base + i]) + float(input[base + i + 1])
                     + float(input[base + i + 2]) + float(input[base + i + 3]);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                sum_val += float(input[base + j]);
            }
        }
    }
    float mean = tg_reduce_sum_ln(sum_val, simd_lane_id, simd_group_id, reduce_buf)
                 / float(axis_size);

    float var_acc = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float d0 = float(input[base + i]) - mean;
            float d1 = float(input[base + i + 1]) - mean;
            float d2 = float(input[base + i + 2]) - mean;
            float d3 = float(input[base + i + 3]) - mean;
            var_acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float d = float(input[base + j]) - mean;
                var_acc += d * d;
            }
        }
    }
    float variance = tg_reduce_sum_ln(var_acc, simd_lane_id, simd_group_id, reduce_buf)
                     / float(axis_size);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = (float(input[base + i])     - mean) * inv_std;
            float o1 = (float(input[base + i + 1]) - mean) * inv_std;
            float o2 = (float(input[base + i + 2]) - mean) * inv_std;
            float o3 = (float(input[base + i + 3]) - mean) * inv_std;
            if (has_w) {
                o0 *= float(weight[i]);     o1 *= float(weight[i + 1]);
                o2 *= float(weight[i + 2]); o3 *= float(weight[i + 3]);
            }
            if (has_b) {
                o0 += float(bias[i]);     o1 += float(bias[i + 1]);
                o2 += float(bias[i + 2]); o3 += float(bias[i + 3]);
            }
            output[base + i]     = half(o0);
            output[base + i + 1] = half(o1);
            output[base + i + 2] = half(o2);
            output[base + i + 3] = half(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = (float(input[base + j]) - mean) * inv_std;
                if (has_w) { o *= float(weight[j]); }
                if (has_b) { o += float(bias[j]); }
                output[base + j] = half(o);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// bf16 LayerNorm (accumulate in f32, read/write bfloat)
// ═══════════════════════════════════════════════════════════════════════════

kernel void layer_norm_bf16(
    device const bfloat* input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device       bfloat* output  [[buffer(3)]],
    constant     uint&   axis_size [[buffer(4)]],
    constant     float&  eps     [[buffer(5)]],
    constant     uint&   has_w   [[buffer(6)]],
    constant     uint&   has_b   [[buffer(7)]],
    uint row            [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tgsize         [[threads_per_threadgroup]],
    uint simd_lane_id   [[thread_index_in_simdgroup]],
    uint simd_group_id  [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float reduce_buf[SIMD_SIZE];

    size_t base = size_t(row) * size_t(axis_size);

    float sum_val = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            sum_val += float(input[base + i]) + float(input[base + i + 1])
                     + float(input[base + i + 2]) + float(input[base + i + 3]);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                sum_val += float(input[base + j]);
            }
        }
    }
    float mean = tg_reduce_sum_ln(sum_val, simd_lane_id, simd_group_id, reduce_buf)
                 / float(axis_size);

    float var_acc = 0.0f;
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float d0 = float(input[base + i]) - mean;
            float d1 = float(input[base + i + 1]) - mean;
            float d2 = float(input[base + i + 2]) - mean;
            float d3 = float(input[base + i + 3]) - mean;
            var_acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float d = float(input[base + j]) - mean;
                var_acc += d * d;
            }
        }
    }
    float variance = tg_reduce_sum_ln(var_acc, simd_lane_id, simd_group_id, reduce_buf)
                     / float(axis_size);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = (float(input[base + i])     - mean) * inv_std;
            float o1 = (float(input[base + i + 1]) - mean) * inv_std;
            float o2 = (float(input[base + i + 2]) - mean) * inv_std;
            float o3 = (float(input[base + i + 3]) - mean) * inv_std;
            if (has_w) {
                o0 *= float(weight[i]);     o1 *= float(weight[i + 1]);
                o2 *= float(weight[i + 2]); o3 *= float(weight[i + 3]);
            }
            if (has_b) {
                o0 += float(bias[i]);     o1 += float(bias[i + 1]);
                o2 += float(bias[i + 2]); o3 += float(bias[i + 3]);
            }
            output[base + i]     = bfloat(o0);
            output[base + i + 1] = bfloat(o1);
            output[base + i + 2] = bfloat(o2);
            output[base + i + 3] = bfloat(o3);
        } else {
            for (uint j = i; j < min(i + N_READS, axis_size); j++) {
                float o = (float(input[base + j]) - mean) * inv_std;
                if (has_w) { o *= float(weight[j]); }
                if (has_b) { o += float(bias[j]); }
                output[base + j] = bfloat(o);
            }
        }
    }
}
"#;

/// Register LayerNorm kernels with the registry.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("layer_norm", LAYER_NORM_SHADER_SOURCE)
}

/// Create a constant buffer on the device.
fn make_const_buf<T: Copy>(device: &metal::DeviceRef, val: T) -> metal::Buffer {
    let size = std::mem::size_of::<T>() as u64;
    device.new_buffer_with_data(
        &val as *const T as *const std::ffi::c_void,
        size,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Apply Layer Normalization: `y = (x - mean) / sqrt(var + eps) * weight + bias`.
///
/// - `input` shape: `[rows, axis_size]` (2-D).
/// - `weight` shape: `[axis_size]` (1-D), or `None` for weight-free normalisation.
/// - `bias` shape: `[axis_size]` (1-D), or `None` for bias-free normalisation.
/// - `eps`: small constant for numerical stability.
pub fn layer_norm(
    registry: &KernelRegistry,
    input: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "layer_norm requires 2D input, got {}D",
            input.ndim()
        )));
    }

    let axis_size_usize = input.shape()[1];

    // Validate weight
    if let Some(w) = weight {
        if w.ndim() != 1 || w.shape()[0] != axis_size_usize {
            return Err(KernelError::InvalidShape(format!(
                "layer_norm: weight must be [{}], got {:?}",
                axis_size_usize,
                w.shape()
            )));
        }
    }

    // Validate bias
    if let Some(b) = bias {
        if b.ndim() != 1 || b.shape()[0] != axis_size_usize {
            return Err(KernelError::InvalidShape(format!(
                "layer_norm: bias must be [{}], got {:?}",
                axis_size_usize,
                b.shape()
            )));
        }
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let weight_contig = weight
        .map(|w| super::make_contiguous(w, registry, queue))
        .transpose()?;
    let weight_resolved: Option<&Array> = match (&weight_contig, weight) {
        (Some(Some(c)), _) => Some(c),
        (_, Some(w)) => Some(w),
        _ => None,
    };

    let bias_contig = bias
        .map(|b| super::make_contiguous(b, registry, queue))
        .transpose()?;
    let bias_resolved: Option<&Array> = match (&bias_contig, bias) {
        (Some(Some(c)), _) => Some(c),
        (_, Some(b)) => Some(b),
        _ => None,
    };

    let kernel_name = match input.dtype() {
        DType::Float32 => "layer_norm_f32",
        DType::Float16 => "layer_norm_f16",
        DType::Bfloat16 => "layer_norm_bf16",
        _ => {
            return Err(KernelError::NotFound(format!(
                "layer_norm not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let dev = registry.device().raw();

    let rows = input.shape()[0];
    let axis_size = super::checked_u32(axis_size_usize, "axis_size")?;

    let out = Array::zeros(dev, input.shape(), input.dtype());

    let axis_buf = make_const_buf(dev, axis_size);
    let eps_buf = make_const_buf(dev, eps);
    let has_w: u32 = if weight_resolved.is_some() { 1 } else { 0 };
    let has_b: u32 = if bias_resolved.is_some() { 1 } else { 0 };
    let has_w_buf = make_const_buf(dev, has_w);
    let has_b_buf = make_const_buf(dev, has_b);

    let cb = queue.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);

    // Weight buffer (dummy if not present)
    if let Some(w) = weight_resolved {
        encoder.set_buffer(1, Some(w.metal_buffer()), w.offset() as u64);
    } else {
        encoder.set_buffer(1, Some(input.metal_buffer()), 0);
    }

    // Bias buffer (dummy if not present)
    if let Some(b) = bias_resolved {
        encoder.set_buffer(2, Some(b.metal_buffer()), b.offset() as u64);
    } else {
        encoder.set_buffer(2, Some(input.metal_buffer()), 0);
    }

    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(&axis_buf), 0);
    encoder.set_buffer(5, Some(&eps_buf), 0);
    encoder.set_buffer(6, Some(&has_w_buf), 0);
    encoder.set_buffer(7, Some(&has_b_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}
