//! SiLU (Sigmoid Linear Unit) activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! Includes fused SiLU*gate (SwiGLU) kernel: silu_gate(x, g) = silu(x) * g.
//!
//! ## Vectorisation
//!
//! - **f32**: 2 elements per thread (work-per-thread = 2).
//! - **f16 / bf16**: 4 elements per thread (work-per-thread = 4) with a
//!   numerically stable sigmoid that avoids half-precision overflow by using
//!   `exp(|x|)` instead of `exp(-x)`.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for SiLU kernels.
pub const SILU_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── helpers ────────────────────────────────────────────────────────────────

// Numerically stable sigmoid for half: uses exp(|x|) form to avoid overflow.
//   sigmoid(x) = x >= 0 ? 1/(1+exp(-x)) : exp(x)/(1+exp(x))
// Both branches use exp(-|x|) which is always <= 1, safe for half.
inline float stable_sigmoid_f32(float x) {
    // Standard form is fine for f32 range.
    return 1.0f / (1.0f + exp(-x));
}

inline half stable_sigmoid_h(half x) {
    float xf = float(x);
    float e = exp(-abs(xf));
    return half(xf >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e));
}

// ─── silu_f32 — 2 elements per thread ──────────────────────────────────────

kernel void silu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base];
        float x1 = input[base + 1];
        output[base]     = x0 * stable_sigmoid_f32(x0);
        output[base + 1] = x1 * stable_sigmoid_f32(x1);
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 * stable_sigmoid_f32(x0);
    }
}

// ─── silu_f16 — 4 elements per thread, stable sigmoid ─────────────────────

kernel void silu_f16(
    device const half* input  [[buffer(0)]],
    device       half* output [[buffer(1)]],
    constant     uint& numel  [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    // Fast path: 4 elements available
    if (base + 3 < numel) {
        half x0 = input[base];
        half x1 = input[base + 1];
        half x2 = input[base + 2];
        half x3 = input[base + 3];
        output[base]     = x0 * stable_sigmoid_h(x0);
        output[base + 1] = x1 * stable_sigmoid_h(x1);
        output[base + 2] = x2 * stable_sigmoid_h(x2);
        output[base + 3] = x3 * stable_sigmoid_h(x3);
    } else {
        // Tail elements
        for (uint j = base; j < min(base + 4, numel); j++) {
            half x = input[j];
            output[j] = x * stable_sigmoid_h(x);
        }
    }
}

// ─── silu_bf16 — 4 elements per thread, accumulate in f32 ─────────────────

kernel void silu_bf16(
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
        output[base]     = bfloat(x0 * stable_sigmoid_f32(x0));
        output[base + 1] = bfloat(x1 * stable_sigmoid_f32(x1));
        output[base + 2] = bfloat(x2 * stable_sigmoid_f32(x2));
        output[base + 3] = bfloat(x3 * stable_sigmoid_f32(x3));
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            float x = float(input[j]);
            output[j] = bfloat(x * stable_sigmoid_f32(x));
        }
    }
}

// ─── Fused SiLU * gate  (SwiGLU pattern) ──────────────────────────────────
// output[i] = silu(input[i]) * gate[i]

kernel void silu_gate_f32(
    device const float* input  [[buffer(0)]],
    device const float* gate   [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  numel  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 2;
    if (base + 1 < numel) {
        float x0 = input[base];
        float x1 = input[base + 1];
        float g0 = gate[base];
        float g1 = gate[base + 1];
        output[base]     = x0 * stable_sigmoid_f32(x0) * g0;
        output[base + 1] = x1 * stable_sigmoid_f32(x1) * g1;
    } else if (base < numel) {
        float x0 = input[base];
        output[base] = x0 * stable_sigmoid_f32(x0) * gate[base];
    }
}

kernel void silu_gate_f16(
    device const half* input  [[buffer(0)]],
    device const half* gate   [[buffer(1)]],
    device       half* output [[buffer(2)]],
    constant     uint& numel  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        half x0 = input[base];
        half x1 = input[base + 1];
        half x2 = input[base + 2];
        half x3 = input[base + 3];
        output[base]     = x0 * stable_sigmoid_h(x0) * gate[base];
        output[base + 1] = x1 * stable_sigmoid_h(x1) * gate[base + 1];
        output[base + 2] = x2 * stable_sigmoid_h(x2) * gate[base + 2];
        output[base + 3] = x3 * stable_sigmoid_h(x3) * gate[base + 3];
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            half x = input[j];
            output[j] = x * stable_sigmoid_h(x) * gate[j];
        }
    }
}

kernel void silu_gate_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* gate   [[buffer(1)]],
    device       bfloat* output [[buffer(2)]],
    constant     uint&   numel  [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint base = id * 4;
    if (base + 3 < numel) {
        float x0 = float(input[base]);
        float x1 = float(input[base + 1]);
        float x2 = float(input[base + 2]);
        float x3 = float(input[base + 3]);
        output[base]     = bfloat(x0 * stable_sigmoid_f32(x0) * float(gate[base]));
        output[base + 1] = bfloat(x1 * stable_sigmoid_f32(x1) * float(gate[base + 1]));
        output[base + 2] = bfloat(x2 * stable_sigmoid_f32(x2) * float(gate[base + 2]));
        output[base + 3] = bfloat(x3 * stable_sigmoid_f32(x3) * float(gate[base + 3]));
    } else {
        for (uint j = base; j < min(base + 4, numel); j++) {
            float x = float(input[j]);
            output[j] = bfloat(x * stable_sigmoid_f32(x) * float(gate[j]));
        }
    }
}

// ─── Strided SiLU * gate (reads gate+up from single merged buffer) ──────────
// merged layout: [seq_len, total_dim] where gate=[0..gate_dim), up=[gate_dim..total_dim)
// output: [seq_len, gate_dim] contiguous

kernel void silu_gate_strided_f32(
    device const float* merged   [[buffer(0)]],
    device       float* output   [[buffer(1)]],
    constant     uint&  gate_dim [[buffer(2)]],
    constant     uint&  total_dim[[buffer(3)]],
    constant     uint&  seq_len  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint base_d = gid.x * 2;
    uint s      = gid.y;
    if (base_d >= gate_dim || s >= seq_len) return;

    uint gate_base = s * total_dim + base_d;
    uint up_base   = s * total_dim + gate_dim + base_d;
    uint out_base  = s * gate_dim  + base_d;

    uint remain = min(gate_dim - base_d, 2u);
    if (remain == 2) {
        float x0 = merged[gate_base];
        float x1 = merged[gate_base + 1];
        float g0 = merged[up_base];
        float g1 = merged[up_base + 1];
        output[out_base]     = x0 * stable_sigmoid_f32(x0) * g0;
        output[out_base + 1] = x1 * stable_sigmoid_f32(x1) * g1;
    } else {
        float x0 = merged[gate_base];
        output[out_base] = x0 * stable_sigmoid_f32(x0) * merged[up_base];
    }
}

kernel void silu_gate_strided_f16(
    device const half*  merged   [[buffer(0)]],
    device       half*  output   [[buffer(1)]],
    constant     uint&  gate_dim [[buffer(2)]],
    constant     uint&  total_dim[[buffer(3)]],
    constant     uint&  seq_len  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint base_d = gid.x * 4;
    uint s      = gid.y;
    if (base_d >= gate_dim || s >= seq_len) return;

    uint gate_base = s * total_dim + base_d;
    uint up_base   = s * total_dim + gate_dim + base_d;
    uint out_base  = s * gate_dim  + base_d;

    uint remain = min(gate_dim - base_d, 4u);
    if (remain == 4) {
        half x0 = merged[gate_base];
        half x1 = merged[gate_base + 1];
        half x2 = merged[gate_base + 2];
        half x3 = merged[gate_base + 3];
        output[out_base]     = x0 * stable_sigmoid_h(x0) * merged[up_base];
        output[out_base + 1] = x1 * stable_sigmoid_h(x1) * merged[up_base + 1];
        output[out_base + 2] = x2 * stable_sigmoid_h(x2) * merged[up_base + 2];
        output[out_base + 3] = x3 * stable_sigmoid_h(x3) * merged[up_base + 3];
    } else {
        for (uint i = 0; i < remain; i++) {
            half x = merged[gate_base + i];
            output[out_base + i] = x * stable_sigmoid_h(x) * merged[up_base + i];
        }
    }
}

kernel void silu_gate_strided_bf16(
    device const bfloat* merged   [[buffer(0)]],
    device       bfloat* output   [[buffer(1)]],
    constant     uint&   gate_dim [[buffer(2)]],
    constant     uint&   total_dim[[buffer(3)]],
    constant     uint&   seq_len  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint base_d = gid.x * 4;
    uint s      = gid.y;
    if (base_d >= gate_dim || s >= seq_len) return;

    uint gate_base = s * total_dim + base_d;
    uint up_base   = s * total_dim + gate_dim + base_d;
    uint out_base  = s * gate_dim  + base_d;

    uint remain = min(gate_dim - base_d, 4u);
    if (remain == 4) {
        float x0 = float(merged[gate_base]);
        float x1 = float(merged[gate_base + 1]);
        float x2 = float(merged[gate_base + 2]);
        float x3 = float(merged[gate_base + 3]);
        output[out_base]     = bfloat(x0 * stable_sigmoid_f32(x0) * float(merged[up_base]));
        output[out_base + 1] = bfloat(x1 * stable_sigmoid_f32(x1) * float(merged[up_base + 1]));
        output[out_base + 2] = bfloat(x2 * stable_sigmoid_f32(x2) * float(merged[up_base + 2]));
        output[out_base + 3] = bfloat(x3 * stable_sigmoid_f32(x3) * float(merged[up_base + 3]));
    } else {
        for (uint i = 0; i < remain; i++) {
            float x = float(merged[gate_base + i]);
            output[out_base + i] = bfloat(x * stable_sigmoid_f32(x) * float(merged[up_base + i]));
        }
    }
}
"#;

/// Register SiLU kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("silu", SILU_SHADER_SOURCE)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return (kernel_name, elements_per_thread) for the strided fused SiLU*gate variant.
pub fn silu_gate_strided_kernel_info(dtype: DType) -> Result<(&'static str, u64), KernelError> {
    match dtype {
        DType::Float32 => Ok(("silu_gate_strided_f32", 2)),
        DType::Float16 => Ok(("silu_gate_strided_f16", 4)),
        DType::Bfloat16 => Ok(("silu_gate_strided_bf16", 4)),
        other => Err(KernelError::InvalidShape(format!(
            "silu_gate_strided: unsupported dtype {:?}",
            other
        ))),
    }
}

/// Return (kernel_name, elements_per_thread) for the given dtype.
fn silu_kernel_info(dtype: DType, fused_gate: bool) -> Result<(&'static str, u64), KernelError> {
    match (dtype, fused_gate) {
        (DType::Float32, false) => Ok(("silu_f32", 2)),
        (DType::Float16, false) => Ok(("silu_f16", 4)),
        (DType::Bfloat16, false) => Ok(("silu_bf16", 4)),
        (DType::Float32, true) => Ok(("silu_gate_f32", 2)),
        (DType::Float16, true) => Ok(("silu_gate_f16", 4)),
        (DType::Bfloat16, true) => Ok(("silu_gate_bf16", 4)),
        (DType::Float8E4M3 | DType::Float8E5M2, _) => Err(KernelError::InvalidShape(
            "silu not supported for FP8 types; dequantize to f16 first".into(),
        )),
        (DType::Q4_0 | DType::Q4_1 | DType::Q8_0, _) => Err(KernelError::InvalidShape(
            "silu not supported for quantized types; dequantize first".into(),
        )),
        (DType::UInt32, _) => Err(KernelError::InvalidShape(
            "silu not supported for UInt32; cast to float first".into(),
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

/// Apply SiLU activation element-wise: silu(x) = x * sigmoid(x).
pub fn silu(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let (kernel_name, elems_per_thread) = silu_kernel_info(input.dtype(), false)?;
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

/// Fused SiLU * gate (SwiGLU pattern): output = silu(input) * gate.
///
/// `input` and `gate` must have the same shape and dtype.
pub fn silu_gate(
    registry: &KernelRegistry,
    input: &Array,
    gate: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    // Validate shapes match
    if input.shape() != gate.shape() {
        return Err(KernelError::InvalidShape(format!(
            "silu_gate: input shape {:?} != gate shape {:?}",
            input.shape(),
            gate.shape()
        )));
    }
    if input.dtype() != gate.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "silu_gate: input dtype {:?} != gate dtype {:?}",
            input.dtype(),
            gate.dtype()
        )));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let gate_contig = super::make_contiguous(gate, registry, queue)?;
    let gate = gate_contig.as_ref().unwrap_or(gate);

    let (kernel_name, elems_per_thread) = silu_kernel_info(input.dtype(), true)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());
    let numel_buf = make_u32_buf(registry.device().raw(), numel as u32);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(gate.metal_buffer()), gate.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_buffer(3, Some(&numel_buf), 0);

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

// ---------------------------------------------------------------------------
// Into-CB variant (encode into existing command buffer, no commit/wait)
// ---------------------------------------------------------------------------

/// Encode SiLU activation into an existing command buffer (no commit/wait).
///
/// **Caller must ensure `input` is contiguous.**
pub fn silu_into_cb(
    registry: &KernelRegistry,
    input: &Array,
    cb: &metal::CommandBufferRef,
) -> Result<Array, KernelError> {
    let (kernel_name, elems_per_thread) = silu_kernel_info(input.dtype(), false)?;
    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel();

    let out = Array::uninit(registry.device().raw(), input.shape(), input.dtype());
    let numel_u32 = numel as u32;

    let grid_threads = (numel as u64).div_ceil(elems_per_thread);

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);
    encoder.set_bytes(2, 4, &numel_u32 as *const u32 as *const std::ffi::c_void);

    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), grid_threads),
        1,
        1,
    );
    encoder.dispatch_threads(MTLSize::new(grid_threads, 1, 1), threadgroup_size);
    encoder.end_encoding();

    Ok(out)
}
