//! Softmax: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const SOFTMAX_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Row-wise softmax with numerically stable max subtraction
kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_data[1024];
    uint base = row * cols;

    // Phase 1: Find max
    float max_val = -INFINITY;
    for (uint i = tid; i < cols; i += tgsize) {
        max_val = max(max_val, input[base + i]);
    }
    shared_data[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exp(x - max) and sum
    float sum_exp = 0.0;
    for (uint i = tid; i < cols; i += tgsize) {
        float val = exp(input[base + i] - row_max);
        output[base + i] = val;
        sum_exp += val;
    }
    shared_data[tid] = sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    float inv_total = (total > 0.0) ? (1.0 / total) : 0.0;
    for (uint i = tid; i < cols; i += tgsize) {
        output[base + i] *= inv_total;
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("softmax", SOFTMAX_SHADER_SOURCE)
}

/// Apply row-wise softmax on a 2D array.
pub fn softmax(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    assert_eq!(input.ndim(), 2, "softmax requires 2D input");

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let kernel_name = match input.dtype() {
        DType::Float32 => "softmax_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "softmax not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let rows = input.shape()[0];
    let cols = super::checked_u32(input.shape()[1], "cols")?;

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

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
