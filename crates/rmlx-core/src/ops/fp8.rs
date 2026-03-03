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
"#;

/// Register FP8 kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("fp8", FP8_SHADER_SOURCE)
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
