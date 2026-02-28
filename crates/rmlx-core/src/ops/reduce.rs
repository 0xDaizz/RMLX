//! Reduction operations: sum, max, argmax.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for reduction kernels.
pub const REDUCE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Sum reduction along all elements (simple single-threadgroup for now)
kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    float sum = 0.0;
    for (uint i = id; i < size; i += tgsize) {
        sum += input[i];
    }
    shared_data[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_data[0];
    }
}

// Max reduction along all elements
kernel void reduce_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    float max_val = -INFINITY;
    for (uint i = id; i < size; i += tgsize) {
        max_val = max(max_val, input[i]);
    }
    shared_data[tid] = max_val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_data[0];
    }
}

// Argmax reduction
kernel void reduce_argmax_f32(
    device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_vals[1024];
    threadgroup uint shared_idxs[1024];

    float max_val = -INFINITY;
    uint max_idx = 0;
    for (uint i = tid; i < size; i += tgsize) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    shared_vals[tid] = max_val;
    shared_idxs[tid] = max_idx;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_idxs[0];
    }
}

// Row-wise sum: each threadgroup handles one row
kernel void reduce_sum_row_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    float sum = 0.0;
    for (uint i = tid; i < cols; i += tgsize) {
        sum += input[row * cols + i];
    }
    shared_data[tid] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_data[0];
    }
}
"#;

/// Reduction type.
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
    ArgMax,
}

/// Reduction axis.
#[derive(Debug, Clone, Copy)]
pub enum ReduceAxis {
    All,
    Row,
}

/// Register reduce kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("reduce", REDUCE_SHADER_SOURCE)
}

/// Reduce all elements of the array.
pub fn reduce_all(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let (kernel_name, out_dtype) = match (op, input.dtype()) {
        (ReduceOp::Sum, DType::Float32) => ("reduce_sum_f32", DType::Float32),
        (ReduceOp::Max, DType::Float32) => ("reduce_max_f32", DType::Float32),
        (ReduceOp::ArgMax, DType::Float32) => ("reduce_argmax_f32", DType::Float32),
        _ => {
            return Err(KernelError::NotFound(format!(
                "reduce {:?} not supported for {:?}",
                op,
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel() as u32;

    // Output is a single scalar
    let out = Array::zeros(registry.device().raw(), &[1], out_dtype);

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&size_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Reduce along rows (for 2D arrays).
pub fn reduce_row(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    assert_eq!(input.ndim(), 2, "reduce_row requires 2D array");
    let rows = input.shape()[0];
    let cols = input.shape()[1] as u32;

    let kernel_name = match (op, input.dtype()) {
        (ReduceOp::Sum, DType::Float32) => "reduce_sum_row_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "reduce_row {:?} not supported for {:?}",
                op,
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let out = Array::zeros(registry.device().raw(), &[rows], input.dtype());

    let cols_buf = registry.device().raw().new_buffer_with_data(
        &cols as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), 0);
    encoder.set_buffer(2, Some(&cols_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}

/// Convenience: sum all elements.
pub fn sum(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Sum, queue)
}

/// Convenience: max of all elements.
pub fn max(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::Max, queue)
}

/// Convenience: argmax of all elements.
pub fn argmax(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    reduce_all(registry, input, ReduceOp::ArgMax, queue)
}
