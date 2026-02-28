//! Reduction operations: sum, max, argmax.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Threshold for switching from single-pass to two-pass reduction.
/// For arrays with <= this many elements, single threadgroup is used.
const TWO_PASS_THRESHOLD: usize = 1024;

/// Metal shader source for reduction kernels.
pub const REDUCE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// === Single-threadgroup reductions (for small arrays or pass 2) ===

// Sum reduction along all elements (single threadgroup)
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
    device float* output [[buffer(1)]],
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
        // Cast uint index to float so output buffer matches DType::Float32
        output[0] = (float)shared_idxs[0];
    }
}

// === Multi-threadgroup reductions (pass 1 of two-pass) ===

// Pass 1 sum: each threadgroup reduces a contiguous chunk and writes to output[gid]
kernel void reduce_sum_pass1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& chunk_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    // Each threadgroup handles a contiguous chunk of the input
    uint chunk_start = gid * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, size);

    float sum = 0.0;
    for (uint i = chunk_start + tid; i < chunk_end; i += tgsize) {
        sum += input[i];
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
        output[gid] = shared_data[0];
    }
}

// Pass 1 max: each threadgroup reduces a contiguous chunk and writes to output[gid]
kernel void reduce_max_pass1_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    constant uint& chunk_size [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];

    uint chunk_start = gid * chunk_size;
    uint chunk_end = min(chunk_start + chunk_size, size);

    float max_val = -INFINITY;
    for (uint i = chunk_start + tid; i < chunk_end; i += tgsize) {
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
        output[gid] = shared_data[0];
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
    reduce_all_with_mode(registry, input, op, queue, super::ExecMode::Sync)
}

/// Reduce all elements with explicit execution mode.
///
/// For arrays with > 1024 elements, uses a two-pass reduction:
/// - Pass 1: Multiple threadgroups each reduce a chunk to a partial result
/// - Pass 2: A single threadgroup reduces the partial results to the final scalar
///
/// For small arrays (<= 1024 elements), uses a single-pass reduction.
pub fn reduce_all_with_mode(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let numel = input.numel();

    // ArgMax always uses single-pass (needs index tracking)
    if matches!(op, ReduceOp::ArgMax) || numel <= TWO_PASS_THRESHOLD {
        return reduce_all_single_pass(registry, input, op, queue, mode);
    }

    // Two-pass reduction for Sum and Max on large arrays
    reduce_all_two_pass(registry, input, op, queue, mode)
}

/// Single-pass reduction (original path, for small arrays and argmax).
fn reduce_all_single_pass(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
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
    let numel = super::checked_u32(input.numel(), "numel")?;

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
    super::commit_with_mode(command_buffer, mode);

    Ok(out)
}

/// Two-pass reduction for large arrays (Sum and Max only).
fn reduce_all_two_pass(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    let numel = input.numel();
    let numel_u32 = super::checked_u32(numel, "numel")?;

    // Choose kernel names for pass 1 and pass 2
    let (pass1_kernel, pass2_kernel) = match (op, input.dtype()) {
        (ReduceOp::Sum, DType::Float32) => ("reduce_sum_pass1_f32", "reduce_sum_f32"),
        (ReduceOp::Max, DType::Float32) => ("reduce_max_pass1_f32", "reduce_max_f32"),
        _ => {
            return Err(KernelError::NotFound(format!(
                "two-pass reduce {:?} not supported for {:?}",
                op,
                input.dtype()
            )))
        }
    };

    let pass1_pipeline = registry.get_pipeline(pass1_kernel, input.dtype())?;
    let tg_size = std::cmp::min(1024u64, pass1_pipeline.max_total_threads_per_threadgroup());

    // Number of threadgroups for pass 1: enough to cover all elements with good occupancy.
    // Must be power of 2 because pass 2 uses tree reduction over the partial results.
    let raw_tg_count = (numel as u64).div_ceil(tg_size).min(256);
    let num_threadgroups = raw_tg_count.next_power_of_two().min(256) as u32;

    // Allocate intermediate buffer for partial results
    let partial = Array::zeros(
        registry.device().raw(),
        &[num_threadgroups as usize],
        DType::Float32,
    );

    // Each threadgroup handles a contiguous chunk
    let chunk_size = (numel as u64).div_ceil(num_threadgroups as u64) as u32;

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel_u32 as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let chunk_buf = registry.device().raw().new_buffer_with_data(
        &chunk_size as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // --- Pass 1: reduce input -> partial results ---
    let cb1 = queue.new_command_buffer();
    let enc1 = cb1.new_compute_command_encoder();
    enc1.set_compute_pipeline_state(&pass1_pipeline);
    enc1.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    enc1.set_buffer(1, Some(partial.metal_buffer()), 0);
    enc1.set_buffer(2, Some(&size_buf), 0);
    enc1.set_buffer(3, Some(&chunk_buf), 0);

    enc1.dispatch_thread_groups(
        MTLSize::new(num_threadgroups as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc1.end_encoding();
    cb1.commit();
    cb1.wait_until_completed();

    // --- Pass 2: reduce partial results -> scalar ---
    let pass2_pipeline = registry.get_pipeline(pass2_kernel, DType::Float32)?;
    let out = Array::zeros(registry.device().raw(), &[1], DType::Float32);

    let partial_size_buf = registry.device().raw().new_buffer_with_data(
        &num_threadgroups as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&pass2_pipeline);
    enc2.set_buffer(0, Some(partial.metal_buffer()), 0);
    enc2.set_buffer(1, Some(out.metal_buffer()), 0);
    enc2.set_buffer(2, Some(&partial_size_buf), 0);

    // Threadgroup size for pass 2 must be a power of 2 for the tree reduction
    let tg2_size = (num_threadgroups as u64)
        .next_power_of_two()
        .min(pass2_pipeline.max_total_threads_per_threadgroup());
    enc2.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(tg2_size.max(1), 1, 1));
    enc2.end_encoding();
    super::commit_with_mode(cb2, mode);

    Ok(out)
}

/// Reduce along rows (for 2D arrays).
pub fn reduce_row(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    if input.ndim() != 2 {
        return Err(KernelError::InvalidShape(format!(
            "reduce_row requires 2D array, got {}D",
            input.ndim()
        )));
    }
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let rows = input.shape()[0];
    let cols = super::checked_u32(input.shape()[1], "cols")?;

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

/// Reduce all elements asynchronously, returning a `LaunchResult`.
///
/// The output `Array` is only accessible after the GPU completes via
/// `LaunchResult::into_array()`.
pub fn reduce_all_async(
    registry: &KernelRegistry,
    input: &Array,
    op: ReduceOp,
    queue: &metal::CommandQueue,
) -> Result<super::LaunchResult, KernelError> {
    if input.numel() == 0 {
        return Err(KernelError::InvalidShape(
            "cannot reduce empty array".into(),
        ));
    }

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let numel = input.numel();

    // ArgMax always uses single-pass (needs index tracking)
    // For async reduce, we need a single command buffer approach.
    // Use single-pass for simplicity (covers most async use cases).
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
    let numel_u32 = super::checked_u32(numel, "numel")?;

    let out = Array::zeros(registry.device().raw(), &[1], out_dtype);

    let size_buf = registry.device().raw().new_buffer_with_data(
        &numel_u32 as *const u32 as *const std::ffi::c_void,
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

    let handle = super::commit_with_mode(command_buffer, super::ExecMode::Async)
        .expect("async mode always returns a handle");

    Ok(super::LaunchResult::new(out, handle))
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
