//! RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const RMS_NORM_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// RMS norm using simdgroup reductions — supports arbitrary axis_size.
// Only needs 32 floats of shared memory (one per simdgroup).
kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& axis_size [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    constexpr int SIMD_SIZE = 32;
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_inv_rms[1];

    // Accumulate sum of squares across strided elements
    float acc = 0.0;
    uint base = row * axis_size;
    for (uint i = tid; i < axis_size; i += tgsize) {
        float val = input[base + i];
        acc += val * val;
    }

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

    float rms = local_inv_rms[0];

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
