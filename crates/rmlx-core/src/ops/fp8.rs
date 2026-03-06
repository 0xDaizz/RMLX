//! FP8 dequantization and quantization kernels.
//!
//! Two FP8 formats are supported:
//!
//! - **E4M3** (1 sign / 4 exponent / 3 mantissa, bias=7): range +/-448, no infinity,
//!   NaN at `0x7F`/`0xFF`. Primary use: weights and activations.
//!
//! - **E5M2** (1 sign / 5 exponent / 2 mantissa, bias=15): range +/-57344, has infinity
//!   and NaN (like f16 structure). Primary use: gradients.
//!
//! FP8 tensors are stored as `DType::Float8E4M3` / `DType::Float8E5M2` backed by
//! uint8 data in the Metal buffer. The dequant functions produce `DType::Float16`
//! output. The quant functions take `DType::Float16` input and produce FP8 output.
//!
//! ## Per-token scaling
//!
//! Per-token variants compute `scale[i] = max(abs(token[i,:])) / 448.0` per row,
//! giving better precision than a single per-tensor scale. The 4-byte scale overhead
//! per token is negligible compared to the bandwidth savings.
//!
//! ## `_into_cb` variants
//!
//! Functions suffixed with `_into_cb` encode GPU work into a caller-provided
//! `CommandBufferRef` without committing. This enables batching multiple kernel
//! dispatches into a single command buffer for reduced CPU overhead.
//!
//! ## Vectorisation
//!
//! Each thread processes 4 elements for better memory throughput.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for FP8 dequant/quant kernels.
pub const FP8_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── FP8 E4M3 dequantization to f16 ────────────────────────────────────────
//
// E4M3: 1 sign, 4 exponent, 3 mantissa, bias=7, no inf, NaN=0x7F/0xFF
// Dequant formula: interpret as IEEE-like float with bias 7.
//
// Each thread processes 4 elements.

kernel void dequant_fp8e4m3_to_f16(
    device const uchar* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    constant uint&      numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    for (uint j = base; j < min(base + 4, numel); j++) {
        uchar bits = input[j];
        uint sign = (bits >> 7) & 1;
        uint exp  = (bits >> 3) & 0xF;  // 4 bits
        uint mant = bits & 0x7;          // 3 bits

        // E4M3 special cases: exp=15 && mant=7 is NaN, no infinity
        // Convert via float intermediate for accuracy
        float val;
        if (exp == 0 && mant == 0) {
            val = 0.0f;
        } else if (exp == 0) {
            // Subnormal: value = (-1)^s * 2^(1-7) * (0 + mant/8)
            val = (mant / 8.0f) * exp2(-6.0f);
        } else if (exp == 15 && mant == 7) {
            val = NAN;
        } else {
            // Normal: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
            val = (1.0f + mant / 8.0f) * exp2(float(exp) - 7.0f);
        }
        if (sign) val = -val;
        output[j] = half(val);
    }
}

// ─── FP8 E5M2 dequantization to f16 ────────────────────────────────────────
//
// E5M2: 1 sign, 5 exponent, 2 mantissa, bias=15, has inf and NaN (like f16)
//
// Each thread processes 4 elements.

kernel void dequant_fp8e5m2_to_f16(
    device const uchar* input  [[buffer(0)]],
    device half*        output [[buffer(1)]],
    constant uint&      numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    for (uint j = base; j < min(base + 4, numel); j++) {
        uchar bits = input[j];
        uint sign = (bits >> 7) & 1;
        uint exp  = (bits >> 2) & 0x1F;  // 5 bits
        uint mant = bits & 0x3;           // 2 bits

        float val;
        if (exp == 0 && mant == 0) {
            val = 0.0f;
        } else if (exp == 0) {
            // Subnormal: value = (-1)^s * 2^(1-15) * (0 + mant/4)
            val = (mant / 4.0f) * exp2(-14.0f);
        } else if (exp == 31 && mant != 0) {
            val = NAN;
        } else if (exp == 31) {
            val = INFINITY;
        } else {
            // Normal: value = (-1)^s * 2^(exp-15) * (1 + mant/4)
            val = (1.0f + mant / 4.0f) * exp2(float(exp) - 15.0f);
        }
        if (sign) val = -val;
        output[j] = half(val);
    }
}

// ─── f16 to FP8 E4M3 quantization ──────────────────────────────────────────
//
// Quantize f16 values to E4M3 with an optional per-tensor scale.
// The input is first multiplied by `scale` before conversion.
//
// Clamping: values outside [-448, 448] are clamped. NaN maps to 0x7F.
//
// Each thread processes 4 elements.

kernel void quant_f16_to_fp8e4m3(
    device const half*  input  [[buffer(0)]],
    device uchar*       output [[buffer(1)]],
    constant uint&      numel  [[buffer(2)]],
    constant float&     scale  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    for (uint j = base; j < min(base + 4, numel); j++) {
        float val = float(input[j]) * scale;

        // Handle NaN
        if (isnan(val)) {
            output[j] = 0x7F;  // E4M3 NaN
            continue;
        }

        // Extract sign
        uint sign = val < 0.0f ? 1 : 0;
        val = abs(val);

        // Clamp to E4M3 max (448.0)
        val = min(val, 448.0f);

        uchar result;
        if (val == 0.0f) {
            result = 0;
        } else {
            // Find exponent: val = 2^e * (1 + m/8) for normal numbers
            // e = floor(log2(val)), biased_exp = e + 7
            int e = int(floor(log2(val)));

            // Clamp exponent to valid range [0, 14] (biased)
            int biased_exp = e + 7;
            if (biased_exp < 0) {
                // Subnormal: exp field = 0, mant = round(val / 2^(-6) * 8)
                float subnormal_val = val * exp2(6.0f) * 8.0f;
                uint mant = uint(subnormal_val + 0.5f);
                mant = min(mant, 7u);
                result = uchar(mant);
            } else if (biased_exp > 14) {
                // Clamp to max normal: exp=14, mant=7 -> 448
                result = uchar((14 << 3) | 7);
            } else {
                // Normal
                float frac = val * exp2(float(-e)) - 1.0f;  // fractional part
                uint mant = uint(frac * 8.0f + 0.5f);       // round to 3 bits
                if (mant > 7) {
                    mant = 0;
                    biased_exp += 1;
                    if (biased_exp > 14) {
                        result = uchar((14 << 3) | 7);
                        result |= uchar(sign << 7);
                        output[j] = result;
                        continue;
                    }
                }
                result = uchar((uint(biased_exp) << 3) | mant);
            }
        }
        result |= uchar(sign << 7);
        output[j] = result;
    }
}

// ─── f16 to FP8 E5M2 quantization ──────────────────────────────────────────
//
// Quantize f16 values to E5M2 with an optional per-tensor scale.
// E5M2 has inf and NaN, same structure as f16 but with only 2 mantissa bits.
//
// Each thread processes 4 elements.

kernel void quant_f16_to_fp8e5m2(
    device const half*  input  [[buffer(0)]],
    device uchar*       output [[buffer(1)]],
    constant uint&      numel  [[buffer(2)]],
    constant float&     scale  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    for (uint j = base; j < min(base + 4, numel); j++) {
        float val = float(input[j]) * scale;

        // Handle NaN
        if (isnan(val)) {
            output[j] = 0x7F;  // E5M2 NaN (exp=31, mant=3)
            continue;
        }

        // Extract sign
        uint sign = val < 0.0f ? 1 : 0;
        val = abs(val);

        // Handle infinity
        if (isinf(val)) {
            output[j] = uchar((sign << 7) | (31 << 2));  // E5M2 inf
            continue;
        }

        // Clamp to E5M2 max (57344.0)
        val = min(val, 57344.0f);

        uchar result;
        if (val == 0.0f) {
            result = 0;
        } else {
            int e = int(floor(log2(val)));
            int biased_exp = e + 15;

            if (biased_exp < 0) {
                // Subnormal
                float subnormal_val = val * exp2(14.0f) * 4.0f;
                uint mant = uint(subnormal_val + 0.5f);
                mant = min(mant, 3u);
                result = uchar(mant);
            } else if (biased_exp > 30) {
                // Clamp to max normal: exp=30, mant=3 -> 57344
                result = uchar((30 << 2) | 3);
            } else {
                float frac = val * exp2(float(-e)) - 1.0f;
                uint mant = uint(frac * 4.0f + 0.5f);
                if (mant > 3) {
                    mant = 0;
                    biased_exp += 1;
                    if (biased_exp > 30) {
                        result = uchar((30 << 2) | 3);
                        result |= uchar(sign << 7);
                        output[j] = result;
                        continue;
                    }
                }
                result = uchar((uint(biased_exp) << 2) | mant);
            }
        }
        result |= uchar(sign << 7);
        output[j] = result;
    }
}

// ─── Per-token FP8 E4M3 quantization (two-pass, race-free) ──────────────────
//
// Pass 1: compute_scales_fp8e4m3 — one thread per row computes
//         scale = max(abs(row)) / 448.0 and writes scales[row].
//
// Pass 2: apply_quant_fp8e4m3 — all threads read the pre-computed scale
//         and quantize their element. No barrier needed because the two
//         passes are separate dispatches.
//
// This avoids the previous cross-threadgroup race where a single-pass
// design used threadgroup_barrier to synchronize scales[row] across
// threadgroups — which is undefined behaviour when a row spans more
// than one threadgroup.

kernel void compute_scales_fp8e4m3(
    device const half*  input      [[buffer(0)]],  // [N, D]
    device float*       scales     [[buffer(1)]],  // [N] per-token scales
    constant uint&      num_tokens [[buffer(2)]],
    constant uint&      hidden_dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint row = tid;
    if (row >= num_tokens) return;

    uint offset = row * hidden_dim;
    float row_max = 0.0f;
    for (uint d = 0; d < hidden_dim; d++) {
        float v = abs(float(input[offset + d]));
        row_max = max(row_max, v);
    }
    // Avoid division by zero: if row_max == 0, scale = 1.0
    float scale = (row_max > 0.0f) ? (row_max / 448.0f) : 1.0f;
    scales[row] = scale;
}

kernel void apply_quant_fp8e4m3(
    device const half*  input      [[buffer(0)]],  // [N, D]
    device uchar*       output     [[buffer(1)]],  // [N, D] FP8
    device const float* scales     [[buffer(2)]],  // [N] per-token scales
    constant uint&      num_tokens [[buffer(3)]],
    constant uint&      hidden_dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint row = tid.y;
    uint col = tid.x;
    if (row >= num_tokens || col >= hidden_dim) return;

    uint offset = row * hidden_dim;
    float scale = scales[row];
    float inv_scale = 1.0f / scale;

    float val = float(input[offset + col]) * inv_scale;

    // Handle NaN
    if (isnan(val)) {
        output[offset + col] = 0x7F;
        return;
    }

    uint sign = val < 0.0f ? 1 : 0;
    val = abs(val);
    val = min(val, 448.0f);

    uchar result;
    if (val == 0.0f) {
        result = 0;
    } else {
        int e = int(floor(log2(val)));
        int biased_exp = e + 7;
        if (biased_exp < 0) {
            float subnormal_val = val * exp2(6.0f) * 8.0f;
            uint mant = uint(subnormal_val + 0.5f);
            mant = min(mant, 7u);
            result = uchar(mant);
        } else if (biased_exp > 15) {
            // Clamp to E4M3 max normal: exp=15, mant=6 -> 448.
            // (mant=7 at exp=15 is NaN in E4M3.)
            result = uchar((15 << 3) | 6);
        } else {
            float frac = val * exp2(float(-e)) - 1.0f;
            uint mant = uint(frac * 8.0f + 0.5f);
            if (mant > 7) {
                mant = 0;
                biased_exp += 1;
            }
            // Clamp: biased_exp=15 && mant=7 is NaN, cap mant at 6.
            if (biased_exp > 15 || (biased_exp == 15 && mant > 6)) {
                result = uchar((15 << 3) | 6);
                result |= uchar(sign << 7);
                output[offset + col] = result;
                return;
            }
            result = uchar((uint(biased_exp) << 3) | mant);
        }
    }
    result |= uchar(sign << 7);
    output[offset + col] = result;
}

// ─── Per-token FP8 E4M3 dequantization ──────────────────────────────────────
//
// Dequantize FP8 E4M3 data back to f16 using per-token scale factors.
//
// 2D grid: tid.x = column, tid.y = token (row).

kernel void dequant_fp8e4m3_per_token(
    device const uchar* input      [[buffer(0)]],  // [N, D] FP8
    device half*        output     [[buffer(1)]],  // [N, D] f16
    device const float* scales     [[buffer(2)]],  // [N] per-token scales
    constant uint&      num_tokens [[buffer(3)]],
    constant uint&      hidden_dim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint row = tid.y;
    uint col = tid.x;
    if (row >= num_tokens || col >= hidden_dim) return;

    uint offset = row * hidden_dim;
    float scale = scales[row];

    uchar bits = input[offset + col];
    uint sign_bit = (bits >> 7) & 1;
    uint exp  = (bits >> 3) & 0xF;
    uint mant = bits & 0x7;

    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else if (exp == 0) {
        val = (mant / 8.0f) * exp2(-6.0f);
    } else if (exp == 15 && mant == 7) {
        val = NAN;
    } else {
        val = (1.0f + mant / 8.0f) * exp2(float(exp) - 7.0f);
    }
    if (sign_bit) val = -val;

    // Re-apply scale to recover original magnitude
    output[offset + col] = half(val * scale);
}
"#;

/// Register FP8 kernels (including per-token variants) with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("fp8", FP8_SHADER_SOURCE)
}

/// Validate that input is a 2D Float16 tensor and return (num_tokens, hidden_dim).
fn validate_2d_f16(input: &Array, fn_name: &str) -> Result<(usize, usize), KernelError> {
    if input.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected Float16 input, got {:?}",
            input.dtype()
        )));
    }
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected 2D input [N, D], got {}D",
            input.ndim()
        )));
    }
    Ok((input.shape()[0], input.shape()[1]))
}

/// Validate that input is a 2D FP8 E4M3 tensor and return (num_tokens, hidden_dim).
fn validate_2d_fp8e4m3(input: &Array, fn_name: &str) -> Result<(usize, usize), KernelError> {
    if input.dtype() != DType::Float8E4M3 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected Float8E4M3 input, got {:?}",
            input.dtype()
        )));
    }
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "{fn_name}: expected 2D input [N, D], got {}D",
            input.ndim()
        )));
    }
    Ok((input.shape()[0], input.shape()[1]))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a constant `uint` buffer on the device.
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Create a constant `float` buffer on the device.
fn make_f32_buf(device: &metal::DeviceRef, val: f32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

/// Elements per thread for the FP8 kernels.
const ELEMS_PER_THREAD: u64 = 4;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Dequantize an FP8 E4M3 tensor to Float16.
///
/// - `input`: Array with `DType::Float8E4M3` (backed by uint8 data).
/// - Returns: Array with `DType::Float16`.
pub fn dequant_fp8e4m3_to_f16(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float8E4M3 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_fp8e4m3_to_f16: expected Float8E4M3 input, got {:?}",
            input.dtype()
        )));
    }
    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "dequant_fp8e4m3_to_f16: input must be contiguous".into(),
        ));
    }

    let pipeline = registry.get_pipeline("dequant_fp8e4m3_to_f16", DType::Float8E4M3)?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), DType::Float16);
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);

    let grid_threads = (numel as u64).div_ceil(ELEMS_PER_THREAD);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Dequantize an FP8 E5M2 tensor to Float16.
///
/// - `input`: Array with `DType::Float8E5M2` (backed by uint8 data).
/// - Returns: Array with `DType::Float16`.
pub fn dequant_fp8e5m2_to_f16(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float8E5M2 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_fp8e5m2_to_f16: expected Float8E5M2 input, got {:?}",
            input.dtype()
        )));
    }
    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "dequant_fp8e5m2_to_f16: input must be contiguous".into(),
        ));
    }

    let pipeline = registry.get_pipeline("dequant_fp8e5m2_to_f16", DType::Float8E5M2)?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), DType::Float16);
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);

    let grid_threads = (numel as u64).div_ceil(ELEMS_PER_THREAD);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Quantize a Float16 tensor to FP8 E4M3.
///
/// - `input`: Array with `DType::Float16`.
/// - `scale`: Per-tensor scale factor applied before quantization.
///   Use `1.0` for no scaling.
/// - Returns: Array with `DType::Float8E4M3`.
pub fn quant_f16_to_fp8e4m3(
    registry: &KernelRegistry,
    input: &Array,
    scale: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "quant_f16_to_fp8e4m3: expected Float16 input, got {:?}",
            input.dtype()
        )));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let pipeline = registry.get_pipeline("quant_f16_to_fp8e4m3", DType::Float16)?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), DType::Float8E4M3);
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);
    let scale_buf = make_f32_buf(registry.device().raw(), scale);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);
    encoder.set_buffer(3, Some(&scale_buf), 0);

    let grid_threads = (numel as u64).div_ceil(ELEMS_PER_THREAD);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Quantize a Float16 tensor to FP8 E5M2.
///
/// - `input`: Array with `DType::Float16`.
/// - `scale`: Per-tensor scale factor applied before quantization.
///   Use `1.0` for no scaling.
/// - Returns: Array with `DType::Float8E5M2`.
pub fn quant_f16_to_fp8e5m2(
    registry: &KernelRegistry,
    input: &Array,
    scale: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.dtype() != DType::Float16 {
        return Err(KernelError::InvalidShape(format!(
            "quant_f16_to_fp8e5m2: expected Float16 input, got {:?}",
            input.dtype()
        )));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let pipeline = registry.get_pipeline("quant_f16_to_fp8e5m2", DType::Float16)?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), DType::Float8E5M2);
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);
    let scale_buf = make_f32_buf(registry.device().raw(), scale);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);
    encoder.set_buffer(3, Some(&scale_buf), 0);

    let grid_threads = (numel as u64).div_ceil(ELEMS_PER_THREAD);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Per-token FP8 E4M3 quantization / dequantization
// ---------------------------------------------------------------------------

/// Encode per-token FP8 E4M3 quantization into a command buffer.
///
/// This is the core implementation shared by `quant_per_token_fp8e4m3` and
/// `quant_per_token_fp8e4m3_into_cb`.
fn encode_quant_per_token(
    registry: &KernelRegistry,
    input: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<(Array, Array), KernelError> {
    let (num_tokens, hidden_dim) = validate_2d_f16(input, "quant_per_token_fp8e4m3")?;

    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "quant_per_token_fp8e4m3: input must be contiguous".into(),
        ));
    }

    let scales_pipeline = registry.get_pipeline("compute_scales_fp8e4m3", DType::Float16)?;
    let quant_pipeline = registry.get_pipeline("apply_quant_fp8e4m3", DType::Float16)?;
    let device = registry.device().raw();

    let fp8_out = Array::zeros(device, input.shape(), DType::Float8E4M3);
    let scales_out = Array::zeros(device, &[num_tokens], DType::Float32);
    let tokens_buf = make_u32_buf(device, num_tokens as u32);
    let dim_buf = make_u32_buf(device, hidden_dim as u32);

    // Pass 1: compute per-token scales (one thread per row).
    {
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&scales_pipeline);
        encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
        encoder.set_buffer(
            1,
            Some(scales_out.metal_buffer()),
            scales_out.offset() as u64,
        );
        encoder.set_buffer(2, Some(&tokens_buf), 0);
        encoder.set_buffer(3, Some(&dim_buf), 0);

        let grid_size = MTLSize::new(num_tokens as u64, 1, 1);
        let tg_w = std::cmp::min(
            num_tokens as u64,
            scales_pipeline.max_total_threads_per_threadgroup(),
        );
        let threadgroup_size = MTLSize::new(tg_w, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
    }

    // Pass 2: quantize using the pre-computed scales.
    // This is a separate encoder, so all writes from pass 1 are visible.
    {
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&quant_pipeline);
        encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
        encoder.set_buffer(1, Some(fp8_out.metal_buffer()), fp8_out.offset() as u64);
        encoder.set_buffer(
            2,
            Some(scales_out.metal_buffer()),
            scales_out.offset() as u64,
        );
        encoder.set_buffer(3, Some(&tokens_buf), 0);
        encoder.set_buffer(4, Some(&dim_buf), 0);

        let grid_size = MTLSize::new(hidden_dim as u64, num_tokens as u64, 1);
        let tg_w = std::cmp::min(
            hidden_dim as u64,
            quant_pipeline.max_total_threads_per_threadgroup(),
        );
        let threadgroup_size = MTLSize::new(tg_w, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
    }

    Ok((fp8_out, scales_out))
}

/// Encode per-token FP8 E4M3 dequantization into a command buffer.
///
/// Core implementation shared by `dequant_per_token_fp8e4m3` and
/// `dequant_per_token_fp8e4m3_into_cb`.
fn encode_dequant_per_token(
    registry: &KernelRegistry,
    input: &Array,
    scales: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let (num_tokens, hidden_dim) = validate_2d_fp8e4m3(input, "dequant_per_token_fp8e4m3")?;

    if !input.is_contiguous() {
        return Err(KernelError::InvalidShape(
            "dequant_per_token_fp8e4m3: input must be contiguous".into(),
        ));
    }
    if scales.dtype() != DType::Float32 {
        return Err(KernelError::InvalidShape(format!(
            "dequant_per_token_fp8e4m3: expected Float32 scales, got {:?}",
            scales.dtype()
        )));
    }
    if scales.ndim() != 1 || scales.shape()[0] != num_tokens {
        return Err(KernelError::InvalidShape(format!(
            "dequant_per_token_fp8e4m3: scales shape mismatch: expected [{}], got {:?}",
            num_tokens,
            scales.shape()
        )));
    }

    let pipeline = registry.get_pipeline("dequant_fp8e4m3_per_token", DType::Float8E4M3)?;
    let device = registry.device().raw();

    let out = Array::zeros(device, input.shape(), DType::Float16);
    let tokens_buf = make_u32_buf(device, num_tokens as u32);
    let dim_buf = make_u32_buf(device, hidden_dim as u32);

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(scales.metal_buffer()), scales.offset() as u64);
    encoder.set_buffer(3, Some(&tokens_buf), 0);
    encoder.set_buffer(4, Some(&dim_buf), 0);

    let grid_size = MTLSize::new(hidden_dim as u64, num_tokens as u64, 1);
    let tg_w = std::cmp::min(
        hidden_dim as u64,
        pipeline.max_total_threads_per_threadgroup(),
    );
    let threadgroup_size = MTLSize::new(tg_w, 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    Ok(out)
}

/// Quantize Float16 to FP8 E4M3 with per-token scaling.
///
/// Each token row gets its own scale factor: `scale[i] = max(abs(token[i,:])) / 448.0`.
/// The 4-byte scale per token is the overhead for the precision improvement.
///
/// - `input`: [N, D] Float16 tensor (must be contiguous).
/// - Returns: `(fp8_output [N, D], scales [N])`.
pub fn quant_per_token_fp8e4m3(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<(Array, Array), KernelError> {
    let command_buffer = queue.new_command_buffer();
    let result = encode_quant_per_token(registry, input, command_buffer)?;
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(result)
}

/// Dequantize FP8 E4M3 to Float16 using per-token scales.
///
/// - `input`: [N, D] FP8 E4M3 tensor (must be contiguous).
/// - `scales`: [N] per-token scale factors as Float32.
/// - Returns: [N, D] Float16 tensor.
pub fn dequant_per_token_fp8e4m3(
    registry: &KernelRegistry,
    input: &Array,
    scales: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let command_buffer = queue.new_command_buffer();
    let result = encode_dequant_per_token(registry, input, scales, command_buffer)?;
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(result)
}

/// Encode per-token FP8 E4M3 quantization into an existing command buffer.
///
/// Does not commit or wait. The caller is responsible for committing the
/// command buffer after encoding all desired work.
///
/// - `input`: [N, D] Float16 tensor (must be contiguous).
/// - `cb`: The command buffer to encode into.
/// - Returns: `(fp8_output [N, D], scales [N])`.
pub fn quant_per_token_fp8e4m3_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<(Array, Array), KernelError> {
    encode_quant_per_token(registry, input, cb)
}

/// Encode per-token FP8 E4M3 dequantization into an existing command buffer.
///
/// Does not commit or wait. The caller is responsible for committing the
/// command buffer after encoding all desired work.
///
/// - `input`: [N, D] FP8 E4M3 tensor (must be contiguous).
/// - `scales`: [N] per-token scale factors as Float32.
/// - `cb`: The command buffer to encode into.
/// - Returns: [N, D] Float16 tensor.
pub fn dequant_per_token_fp8e4m3_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    scales: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    encode_dequant_per_token(registry, input, scales, cb)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a Float16 array from f32 values by converting to IEEE 754
    /// binary16 format manually (no `half` crate dependency).
    fn f32_to_f16_bits(val: f32) -> u16 {
        // Use the standard conversion: f32 -> f16 via bit manipulation.
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mant = bits & 0x7F_FFFF;

        if exp > 15 {
            // Overflow -> infinity
            (sign | 0x7C00) as u16
        } else if exp < -14 {
            // Subnormal or zero
            if exp < -24 {
                sign as u16
            } else {
                let shift = -14 - exp;
                let subnorm = (0x400 | (mant >> 13)) >> shift;
                (sign | subnorm) as u16
            }
        } else {
            let biased = (exp + 15) as u32;
            let m16 = mant >> 13;
            (sign | (biased << 10) | m16) as u16
        }
    }

    fn make_f16_array(device: &metal::Device, data: &[f32], shape: Vec<usize>) -> Array {
        let f16_bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
            .collect();
        Array::from_bytes(device, &f16_bytes, shape, DType::Float16)
    }

    fn read_f32_array(arr: &Array) -> Vec<f32> {
        arr.to_vec_checked::<f32>()
    }

    fn setup() -> (KernelRegistry, metal::CommandQueue) {
        let gpu_dev = rmlx_metal::device::GpuDevice::system_default().unwrap();
        let device = gpu_dev.raw().clone();
        let queue = device.new_command_queue();
        let registry = KernelRegistry::new(gpu_dev);
        register(&registry).expect("register fp8 kernels");
        (registry, queue)
    }

    #[test]
    fn test_per_token_quant_small_dim() {
        // Basic correctness: small hidden_dim that fits in one threadgroup.
        let (registry, queue) = setup();
        let device = registry.device().raw();

        // 2 tokens, hidden_dim=4, values in a known range
        let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 100.0, 200.0, -300.0, 400.0];
        let input = make_f16_array(device, &data, vec![2, 4]);

        let (fp8_out, scales) =
            quant_per_token_fp8e4m3(&registry, &input, &queue).expect("quant should succeed");

        assert_eq!(fp8_out.shape(), &[2, 4]);
        assert_eq!(scales.shape(), &[2]);

        let scale_vals = read_f32_array(&scales);
        // Row 0: max abs = 4.0, scale = 4.0 / 448.0
        let expected_scale_0 = 4.0_f32 / 448.0;
        assert!(
            (scale_vals[0] - expected_scale_0).abs() < 1e-4,
            "scale[0] = {}, expected ~{}",
            scale_vals[0],
            expected_scale_0
        );
        // Row 1: max abs = 400.0, scale = 400.0 / 448.0
        let expected_scale_1 = 400.0_f32 / 448.0;
        assert!(
            (scale_vals[1] - expected_scale_1).abs() < 1e-3,
            "scale[1] = {}, expected ~{}",
            scale_vals[1],
            expected_scale_1
        );

        // Round-trip: dequantize and check values are close
        let recovered = dequant_per_token_fp8e4m3(&registry, &fp8_out, &scales, &queue)
            .expect("dequant should succeed");
        assert_eq!(recovered.shape(), &[2, 4]);

        // Read f16 output as raw bytes and convert back to f32 for comparison
        let numel = 8usize;
        let raw_ptr = recovered.metal_buffer().contents() as *const u16;
        let f16_vals: Vec<f32> = (0..numel)
            .map(|i| {
                let bits = unsafe { *raw_ptr.add(i) };
                f16_bits_to_f32(bits)
            })
            .collect();

        for (i, (&orig, &recov)) in data.iter().zip(f16_vals.iter()).enumerate() {
            // FP8 E4M3 has limited precision (3 mantissa bits), so tolerance must
            // account for ~6% quantization error plus the f16 scale imprecision.
            let tol = orig.abs() * 0.1 + 1.0;
            assert!(
                (orig - recov).abs() < tol,
                "element {}: orig={}, recovered={}, diff={}",
                i,
                orig,
                recov,
                (orig - recov).abs()
            );
        }
    }

    #[test]
    fn test_per_token_quant_large_hidden_dim() {
        // Regression test for the cross-threadgroup race on scales.
        // Use hidden_dim = 2048 which exceeds typical max_threadgroup_size (1024).
        let (registry, queue) = setup();
        let device = registry.device().raw();

        let num_tokens = 4usize;
        let hidden_dim = 2048usize;

        // Create data where each row has a known max at a specific column.
        // Row i: all values are 1.0 except column (hidden_dim - 1) which is (i+1)*100.
        let mut data = vec![1.0f32; num_tokens * hidden_dim];
        for row in 0..num_tokens {
            let peak = (row as f32 + 1.0) * 100.0;
            data[row * hidden_dim + hidden_dim - 1] = peak;
        }

        let input = make_f16_array(device, &data, vec![num_tokens, hidden_dim]);

        let (fp8_out, scales) =
            quant_per_token_fp8e4m3(&registry, &input, &queue).expect("quant should succeed");

        assert_eq!(fp8_out.shape(), &[num_tokens, hidden_dim]);
        assert_eq!(scales.shape(), &[num_tokens]);

        let scale_vals = read_f32_array(&scales);

        for (row, &scale_val) in scale_vals.iter().enumerate().take(num_tokens) {
            let peak = (row as f32 + 1.0) * 100.0;
            let expected_scale = peak / 448.0;
            assert!(
                (scale_val - expected_scale).abs() < 1e-2,
                "row {}: scale = {}, expected ~{} (peak={})",
                row,
                scale_val,
                expected_scale,
                peak
            );
        }

        // Round-trip dequant and verify the peak values survive
        let recovered = dequant_per_token_fp8e4m3(&registry, &fp8_out, &scales, &queue)
            .expect("dequant should succeed");

        let raw_ptr = recovered.metal_buffer().contents() as *const u16;
        for row in 0..num_tokens {
            let peak = (row as f32 + 1.0) * 100.0;
            let idx = row * hidden_dim + hidden_dim - 1;
            let bits = unsafe { *raw_ptr.add(idx) };
            let recov = f16_bits_to_f32(bits);
            let tol = peak * 0.05 + 1.0;
            assert!(
                (peak - recov).abs() < tol,
                "row {} peak: orig={}, recovered={}, diff={}",
                row,
                peak,
                recov,
                (peak - recov).abs()
            );
        }
    }

    #[test]
    fn test_per_token_quant_zero_row() {
        // A row of all zeros should produce scale = 1.0 and all-zero FP8 output.
        let (registry, queue) = setup();
        let device = registry.device().raw();

        let data = vec![0.0f32; 8];
        let input = make_f16_array(device, &data, vec![2, 4]);

        let (_fp8_out, scales) =
            quant_per_token_fp8e4m3(&registry, &input, &queue).expect("quant should succeed");

        let scale_vals = read_f32_array(&scales);
        for (i, &s) in scale_vals.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-6,
                "zero-row {}: scale should be 1.0, got {}",
                i,
                s
            );
        }
    }

    /// Convert f16 bits back to f32 for test comparison.
    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;

        if exp == 0 && mant == 0 {
            if sign == 1 {
                -0.0f32
            } else {
                0.0f32
            }
        } else if exp == 0 {
            // Subnormal
            let val = (mant as f32 / 1024.0) * 2.0f32.powi(-14);
            if sign == 1 {
                -val
            } else {
                val
            }
        } else if exp == 31 {
            if mant != 0 {
                f32::NAN
            } else if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    }
}
