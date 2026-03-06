//! GELU (Gaussian Error Linear Unit) activation functions.
//!
//! Two variants are provided:
//!
//! - **gelu_approx** — tanh approximation (GPT-2, BERT, Gemma):
//!   `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
//!
//! - **gelu_fast** — sigmoid approximation (max error <0.015):
//!   `gelu(x) = x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))`
//!
//! ## Vectorisation
//!
//! - **f32**: 2 elements per thread (work-per-thread = 2).
//! - **f16 / bf16**: 4 elements per thread (work-per-thread = 4) with
//!   numerically stable implementations that avoid half-precision overflow.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for GELU kernels.
pub const GELU_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── helpers ────────────────────────────────────────────────────────────────

// sqrt(2/pi) ≈ 0.7978845608
constant float SQRT_2_OVER_PI = 0.7978845608f;
constant float GELU_COEFF     = 0.044715f;
constant float GELU_FAST_COEFF = 1.702f;

// Numerically stable sigmoid for half: uses exp(-|x|) form to avoid overflow.
//   sigmoid(x) = x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))
// Both branches use exp(-|x|) which is always <= 1, safe for half.
inline float stable_sigmoid_f32(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline half stable_sigmoid_h(half x) {
    float xf = float(x);
    float e = exp(-abs(xf));
    return half(xf >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

// ─── gelu_approx_f32 — 2 elements per thread ─────────────────────────────

kernel void gelu_approx_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base];
        float x1 = input[base + 1];
        float inner0 = SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0);
        float inner1 = SQRT_2_OVER_PI * (x1 + GELU_COEFF * x1 * x1 * x1);
        output[base]     = 0.5f * x0 * (1.0f + metal::precise::tanh(inner0));
        output[base + 1] = 0.5f * x1 * (1.0f + metal::precise::tanh(inner1));
    } else if (base < numel) {
        float x0 = input[base];
        float inner0 = SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0);
        output[base] = 0.5f * x0 * (1.0f + metal::precise::tanh(inner0));
    }
}

// ─── gelu_approx_f16 — 4 elements per thread, accumulate in f32 ──────────

kernel void gelu_approx_f16(
    device const half* input  [[buffer(0)]],
    device       half* output [[buffer(1)]],
    constant     uint& numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        float x0 = float(input[base]);
        float x1 = float(input[base + 1]);
        float x2 = float(input[base + 2]);
        float x3 = float(input[base + 3]);
        float inner0 = SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0);
        float inner1 = SQRT_2_OVER_PI * (x1 + GELU_COEFF * x1 * x1 * x1);
        float inner2 = SQRT_2_OVER_PI * (x2 + GELU_COEFF * x2 * x2 * x2);
        float inner3 = SQRT_2_OVER_PI * (x3 + GELU_COEFF * x3 * x3 * x3);
        output[base]     = half(0.5f * x0 * (1.0f + tanh(inner0)));
        output[base + 1] = half(0.5f * x1 * (1.0f + tanh(inner1)));
        output[base + 2] = half(0.5f * x2 * (1.0f + tanh(inner2)));
        output[base + 3] = half(0.5f * x3 * (1.0f + tanh(inner3)));
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            float x = float(input[j]);
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            output[j] = half(0.5f * x * (1.0f + tanh(inner)));
        }
    }
}

// ─── gelu_approx_bf16 — 4 elements per thread, accumulate in f32 ─────────

kernel void gelu_approx_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        float x0 = float(input[base]);
        float x1 = float(input[base + 1]);
        float x2 = float(input[base + 2]);
        float x3 = float(input[base + 3]);
        float inner0 = SQRT_2_OVER_PI * (x0 + GELU_COEFF * x0 * x0 * x0);
        float inner1 = SQRT_2_OVER_PI * (x1 + GELU_COEFF * x1 * x1 * x1);
        float inner2 = SQRT_2_OVER_PI * (x2 + GELU_COEFF * x2 * x2 * x2);
        float inner3 = SQRT_2_OVER_PI * (x3 + GELU_COEFF * x3 * x3 * x3);
        output[base]     = bfloat(0.5f * x0 * (1.0f + tanh(inner0)));
        output[base + 1] = bfloat(0.5f * x1 * (1.0f + tanh(inner1)));
        output[base + 2] = bfloat(0.5f * x2 * (1.0f + tanh(inner2)));
        output[base + 3] = bfloat(0.5f * x3 * (1.0f + tanh(inner3)));
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            float x = float(input[j]);
            float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            output[j] = bfloat(0.5f * x * (1.0f + tanh(inner)));
        }
    }
}

// ─── gelu_fast_f32 — 2 elements per thread ────────────────────────────────

kernel void gelu_fast_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base];
        float x1 = input[base + 1];
        output[base]     = x0 * stable_sigmoid_f32(GELU_FAST_COEFF * x0);
        output[base + 1] = x1 * stable_sigmoid_f32(GELU_FAST_COEFF * x1);
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 * stable_sigmoid_f32(GELU_FAST_COEFF * x0);
    }
}

// ─── gelu_fast_f16 — 4 elements per thread, stable sigmoid ───────────────

kernel void gelu_fast_f16(
    device const half* input  [[buffer(0)]],
    device       half* output [[buffer(1)]],
    constant     uint& numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        half x0 = input[base];
        half x1 = input[base + 1];
        half x2 = input[base + 2];
        half x3 = input[base + 3];
        output[base]     = x0 * stable_sigmoid_h(half(GELU_FAST_COEFF) * x0);
        output[base + 1] = x1 * stable_sigmoid_h(half(GELU_FAST_COEFF) * x1);
        output[base + 2] = x2 * stable_sigmoid_h(half(GELU_FAST_COEFF) * x2);
        output[base + 3] = x3 * stable_sigmoid_h(half(GELU_FAST_COEFF) * x3);
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            half x = input[j];
            output[j] = x * stable_sigmoid_h(half(GELU_FAST_COEFF) * x);
        }
    }
}

// ─── gelu_fast_bf16 — 4 elements per thread, accumulate in f32 ───────────

kernel void gelu_fast_bf16(
    device const bfloat* input  [[buffer(0)]],
    device       bfloat* output [[buffer(1)]],
    constant     uint&   numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        float x0 = float(input[base]);
        float x1 = float(input[base + 1]);
        float x2 = float(input[base + 2]);
        float x3 = float(input[base + 3]);
        output[base]     = bfloat(x0 * stable_sigmoid_f32(GELU_FAST_COEFF * x0));
        output[base + 1] = bfloat(x1 * stable_sigmoid_f32(GELU_FAST_COEFF * x1));
        output[base + 2] = bfloat(x2 * stable_sigmoid_f32(GELU_FAST_COEFF * x2));
        output[base + 3] = bfloat(x3 * stable_sigmoid_f32(GELU_FAST_COEFF * x3));
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            float x = float(input[j]);
            output[j] = bfloat(x * stable_sigmoid_f32(GELU_FAST_COEFF * x));
        }
    }
}
"#;

/// Register GELU kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("gelu", GELU_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return (kernel_name, elements_per_thread) for the given dtype and variant.
fn gelu_kernel_info(dtype: DType, fast: bool) -> Result<(&'static str, u64), KernelError> {
    match (dtype, fast) {
        (DType::Float32, false) => Ok(("gelu_approx_f32", 2)),
        (DType::Float16, false) => Ok(("gelu_approx_f16", 4)),
        (DType::Bfloat16, false) => Ok(("gelu_approx_bf16", 4)),
        (DType::Float32, true) => Ok(("gelu_fast_f32", 2)),
        (DType::Float16, true) => Ok(("gelu_fast_f16", 4)),
        (DType::Bfloat16, true) => Ok(("gelu_fast_bf16", 4)),
        (DType::Float8E4M3 | DType::Float8E5M2, _) => Err(KernelError::InvalidShape(
            "gelu not supported for FP8 types; dequantize to f16 first".into(),
        )),
        (DType::Q4_0 | DType::Q4_1 | DType::Q8_0, _) => Err(KernelError::InvalidShape(
            "gelu not supported for quantized types; dequantize first".into(),
        )),
        (DType::UInt32, _) => Err(KernelError::InvalidShape(
            "gelu not supported for UInt32; cast to float first".into(),
        )),
    }
}

/// Create a constant `uint` buffer on the device.
fn make_u32_buf(device: &metal::DeviceRef, val: u32) -> metal::Buffer {
    device.new_buffer_with_data(
        &val as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply GELU activation element-wise using the tanh approximation (GPT-2, BERT, Gemma):
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let (kernel_name, elems_per_thread) = gelu_kernel_info(input.dtype(), false)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);

    // Grid = ceil(numel / elems_per_thread) threads
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, super::ExecMode::Sync);

    Ok(out)
}

/// Apply GELU activation element-wise using the fast sigmoid approximation:
/// `gelu_fast(x) = x * sigmoid(1.702 * x)`.
pub fn gelu_fast(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let (kernel_name, elems_per_thread) = gelu_kernel_info(input.dtype(), true)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(2, Some(&numel_buf), 0);

    // Grid = ceil(numel / elems_per_thread) threads
    let grid_threads = (numel as u64).div_ceil(elems_per_thread);
    let grid_size = MTLSize::new(grid_threads, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, super::ExecMode::Sync);

    Ok(out)
}
