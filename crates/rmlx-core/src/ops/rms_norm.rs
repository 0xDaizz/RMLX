//! RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
//!
//! ## Improvements over the baseline kernel
//!
//! - **N_READS=4 coalescing**: each thread processes 4 consecutive elements per
//!   iteration for better memory throughput.
//! - **Weight stride support** (`w_stride`): handles non-contiguous weight tensors.
//! - **Optional weight** (`has_w`): when the weight pointer is null the
//!   multiplication is skipped (pure RMS normalisation).
//! - **uint32 overflow fix**: uses `size_t` for row-base addressing so that
//!   `row * axis_size` does not overflow 32 bits on large tensors.
//! - **f16 / bf16 support**: accumulation in f32, read/write in half / bfloat.
//! - **Single-row vs looped variants**: for `axis_size <= 4096` a simpler
//!   single-row kernel is selected; for larger sizes the looped variant is used.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const RMS_NORM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ─── N_READS = 4 coalescing constant ──────────────────────────────────────
constant constexpr uint N_READS = 4;

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

    // ── Phase 1: sum of squares with N_READS coalescing ──
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

    // ── Phase 2: normalise + optional weight ──
    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = input[base + i]     * rms;
            float o1 = input[base + i + 1] * rms;
            float o2 = input[base + i + 2] * rms;
            float o3 = input[base + i + 3] * rms;
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
                float o = input[base + j] * rms;
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
    uint i = tid;
    if (i < axis_size) {
        float v = input[base + i];
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    // ── Phase 2: normalise + optional weight ──
    if (i < axis_size) {
        float o = input[base + i] * rms;
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

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = float(input[base + i])     * rms;
            float o1 = float(input[base + i + 1]) * rms;
            float o2 = float(input[base + i + 2]) * rms;
            float o3 = float(input[base + i + 3]) * rms;
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
                float o = float(input[base + j]) * rms;
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

    float acc = 0.0;
    uint i = tid;
    if (i < axis_size) {
        float v = float(input[base + i]);
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    if (i < axis_size) {
        float o = float(input[base + i]) * rms;
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

    for (uint i = tid * N_READS; i < axis_size; i += tgsize * N_READS) {
        if (i + 3 < axis_size) {
            float o0 = float(input[base + i])     * rms;
            float o1 = float(input[base + i + 1]) * rms;
            float o2 = float(input[base + i + 2]) * rms;
            float o3 = float(input[base + i + 3]) * rms;
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
                float o = float(input[base + j]) * rms;
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

    float acc = 0.0;
    uint i = tid;
    if (i < axis_size) {
        float v = float(input[base + i]);
        acc = v * v;
    }

    float rms = reduce_sum_of_squares(acc, axis_size, eps,
                                       simd_lane_id, simd_group_id,
                                       local_sums, local_inv_rms);

    if (i < axis_size) {
        float o = float(input[base + i]) * rms;
        if (has_w) { o *= float(weight[i * w_stride]); }
        output[base + i] = bfloat(o);
    }
}
"#;

/// Threshold: axis_size <= this uses the simpler single-row kernel.
const SINGLE_ROW_THRESHOLD: usize = 1024;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("rms_norm", RMS_NORM_SHADER_SOURCE)
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

/// Create a constant buffer on the device.
fn make_const_buf<T: Copy>(device: &metal::DeviceRef, val: T) -> metal::Buffer {
    let size = std::mem::size_of::<T>() as u64;
    device.new_buffer_with_data(
        &val as *const T as *const std::ffi::c_void,
        size,
        metal::MTLResourceOptions::StorageModeShared,
    )
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
    queue: &metal::CommandQueue,
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
    queue: &metal::CommandQueue,
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

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    let axis_buf = make_const_buf(registry.device().raw(), axis_size);
    let eps_buf = make_const_buf(registry.device().raw(), eps);
    let w_stride_buf = make_const_buf(registry.device().raw(), w_stride);
    let has_w_buf = make_const_buf(registry.device().raw(), has_w);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    // For weight buffer: when has_w == 0 we still bind the input buffer as a
    // dummy (the kernel will never read from it).
    if let Some(w) = weight_resolved {
        encoder.set_buffer(1, Some(w.metal_buffer()), w.offset() as u64);
    } else {
        encoder.set_buffer(1, Some(input.metal_buffer()), 0);
    }
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&axis_buf), 0);
    encoder.set_buffer(4, Some(&eps_buf), 0);
    encoder.set_buffer(5, Some(&w_stride_buf), 0);
    encoder.set_buffer(6, Some(&has_w_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

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
    cb: &metal::CommandBufferRef,
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

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    let axis_buf = make_const_buf(registry.device().raw(), axis_size);
    let eps_buf = make_const_buf(registry.device().raw(), eps);
    let w_stride_buf = make_const_buf(registry.device().raw(), w_stride);
    let has_w_buf = make_const_buf(registry.device().raw(), has_w);

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    if let Some(w) = weight {
        encoder.set_buffer(1, Some(w.metal_buffer()), w.offset() as u64);
    } else {
        encoder.set_buffer(1, Some(input.metal_buffer()), 0);
    }
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&axis_buf), 0);
    encoder.set_buffer(4, Some(&eps_buf), 0);
    encoder.set_buffer(5, Some(&w_stride_buf), 0);
    encoder.set_buffer(6, Some(&has_w_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();

    Ok(out)
}
