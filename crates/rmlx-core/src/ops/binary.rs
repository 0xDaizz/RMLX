//! Element-wise binary operations (add, mul, sub, div).

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for binary kernels.
pub const BINARY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] + b[id];
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] * b[id];
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] - b[id];
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] / b[id];
}

kernel void add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] + b[id];
}

kernel void mul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] * b[id];
}

kernel void sub_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] - b[id];
}

kernel void div_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] / b[id];
}

kernel void add_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] + b[id];
}

kernel void mul_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] * b[id];
}

kernel void sub_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] - b[id];
}

kernel void div_bf16(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = a[id] / b[id];
}
"#;

/// Binary operation type.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
}

impl BinaryOp {
    fn kernel_name(&self, dtype: DType) -> Result<&'static str, KernelError> {
        match (self, dtype) {
            (BinaryOp::Add, DType::Float32) => Ok("add_f32"),
            (BinaryOp::Mul, DType::Float32) => Ok("mul_f32"),
            (BinaryOp::Sub, DType::Float32) => Ok("sub_f32"),
            (BinaryOp::Div, DType::Float32) => Ok("div_f32"),
            (BinaryOp::Add, DType::Float16) => Ok("add_f16"),
            (BinaryOp::Mul, DType::Float16) => Ok("mul_f16"),
            (BinaryOp::Sub, DType::Float16) => Ok("sub_f16"),
            (BinaryOp::Div, DType::Float16) => Ok("div_f16"),
            (BinaryOp::Add, DType::Bfloat16) => Ok("add_bf16"),
            (BinaryOp::Mul, DType::Bfloat16) => Ok("mul_bf16"),
            (BinaryOp::Sub, DType::Bfloat16) => Ok("sub_bf16"),
            (BinaryOp::Div, DType::Bfloat16) => Ok("div_bf16"),
            (_, DType::Q4_0 | DType::Q4_1 | DType::Q8_0) => Err(KernelError::InvalidShape(
                "binary ops not supported for quantized types; dequantize first".into(),
            )),
        }
    }
}

/// Register binary kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("binary", BINARY_SHADER_SOURCE)
}

/// Execute a binary operation element-wise.
pub fn binary_op(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op_with_mode(registry, a, b, op, queue, super::ExecMode::Sync)
}

/// Execute a binary operation element-wise with explicit execution mode.
pub fn binary_op_with_mode(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    if a.shape() != b.shape() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching dtypes: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let kernel_name = op.kernel_name(a.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let numel = a.numel();

    let out = Array::zeros(registry.device().raw(), a.shape(), a.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    encoder.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);

    let grid_size = MTLSize::new(numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Execute a binary operation asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
pub fn binary_op_async(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    op: BinaryOp,
    queue: &metal::CommandQueue,
) -> Result<super::LaunchResult, KernelError> {
    if a.shape() != b.shape() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching shapes: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(KernelError::InvalidShape(format!(
            "binary op requires matching dtypes: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    let a_contig = super::make_contiguous(a, registry, queue)?;
    let a = a_contig.as_ref().unwrap_or(a);
    let b_contig = super::make_contiguous(b, registry, queue)?;
    let b = b_contig.as_ref().unwrap_or(b);

    let kernel_name = op.kernel_name(a.dtype())?;
    let pipeline = registry.get_pipeline(kernel_name, a.dtype())?;
    let numel = a.numel();

    let out = Array::zeros(registry.device().raw(), a.shape(), a.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(a.metal_buffer()), a.offset() as u64);
    encoder.set_buffer(1, Some(b.metal_buffer()), b.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), out.offset() as u64);

    let grid_size = MTLSize::new(numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    let handle = super::commit_with_mode(command_buffer, super::ExecMode::Async)
        .expect("async mode always returns a handle");

    Ok(super::LaunchResult::new(out, handle))
}

/// Convenience functions.
pub fn add(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Add, queue)
}

pub fn mul(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Mul, queue)
}

pub fn sub(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Sub, queue)
}

pub fn div(
    registry: &KernelRegistry,
    a: &Array,
    b: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    binary_op(registry, a, b, BinaryOp::Div, queue)
}
