//! RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const RMS_NORM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup float shared_sum[1024];

    // Compute sum of squares for this row
    float sum_sq = 0.0;
    uint base = row * axis_size;
    for (uint i = tid; i < axis_size; i += tgsize) {
        float val = input[base + i];
        sum_sq += val * val;
    }
    shared_sum[tid] = sum_sq;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum[0] / float(axis_size) + eps);

    // Apply normalization and weight
    for (uint i = tid; i < axis_size; i += tgsize) {
        output[base + i] = input[base + i] * rms * weight[i];
    }
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("rms_norm", RMS_NORM_SHADER_SOURCE)
}

/// Apply RMS normalization: y = x * rsqrt(mean(x^2) + eps) * weight
/// Input shape: [rows, axis_size], weight shape: [axis_size]
pub fn rms_norm(
    registry: &KernelRegistry,
    input: &Array,
    weight: &Array,
    eps: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    assert_eq!(input.ndim(), 2, "rms_norm requires 2D input");
    assert_eq!(weight.ndim(), 1, "rms_norm requires 1D weight");
    assert_eq!(input.shape()[1], weight.shape()[0], "axis size mismatch");

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let weight_contig = super::make_contiguous(weight, registry, queue)?;
    let weight = weight_contig.as_ref().unwrap_or(weight);

    let axis_size = input.shape()[1];
    if axis_size > 1024 {
        return Err(KernelError::InvalidShape(format!(
            "rms_norm axis_size {} exceeds max threadgroup size 1024",
            axis_size
        )));
    }

    let kernel_name = match input.dtype() {
        DType::Float32 => "rms_norm_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "rms_norm not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let rows = input.shape()[0];
    let axis_size = super::checked_u32(input.shape()[1], "axis_size")?;

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    let axis_buf = registry.device().raw().new_buffer_with_data(
        &axis_size as *const u32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let eps_buf = registry.device().raw().new_buffer_with_data(
        &eps as *const f32 as *const std::ffi::c_void,
        4,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(weight.metal_buffer()), weight.offset() as u64);
    encoder.set_buffer(2, Some(out.metal_buffer()), 0);
    encoder.set_buffer(3, Some(&axis_buf), 0);
    encoder.set_buffer(4, Some(&eps_buf), 0);

    let tg_size = std::cmp::min(1024, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}
