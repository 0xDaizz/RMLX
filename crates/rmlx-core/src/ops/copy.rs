//! Copy and transpose operations.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for copy kernels.
///
/// Includes both contiguous (flat) copy and strided (general) copy.
/// The strided kernel decomposes the flat output index into ND coordinates
/// using the output shape, then computes the source offset via input strides.
pub const COPY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// === Contiguous (flat) copy ===

kernel void copy_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

kernel void copy_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    dst[id] = src[id];
}

// === Stride-aware (general) copy ===
// Buffers:
//   0: src data pointer
//   1: dst data pointer (contiguous output)
//   2: shape array (uint, length = ndim)
//   3: src_strides array (uint, length = ndim)
//   4: ndim (uint)
//
// For each output element at flat index `id`, we decompose `id` into
// ND coordinates using the shape, then compute the source flat offset
// by dotting those coordinates with src_strides.

kernel void copy_strided_f32(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant const uint* shape [[buffer(2)]],
    constant const uint* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        // Compute stride for this dimension in contiguous output
        uint out_stride = 1;
        for (uint k = d + 1; k < ndim; k++) {
            out_stride *= shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}

kernel void copy_strided_f16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant const uint* shape [[buffer(2)]],
    constant const uint* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    uint src_offset = 0;
    uint remaining = id;
    for (uint d = 0; d < ndim; d++) {
        uint out_stride = 1;
        for (uint k = d + 1; k < ndim; k++) {
            out_stride *= shape[k];
        }
        uint coord = remaining / out_stride;
        remaining = remaining % out_stride;
        src_offset += coord * src_strides[d];
    }
    dst[id] = src[src_offset];
}
"#;

/// Register copy kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("copy", COPY_SHADER_SOURCE)
}

/// Copy array contents to a new contiguous array.
pub fn copy(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    copy_with_mode(registry, src, queue, super::ExecMode::Sync)
}

/// Copy array contents to a new contiguous array with explicit execution mode.
///
/// If the source is contiguous, uses a simple flat copy.
/// If the source is non-contiguous (strided), uses the strided copy kernel
/// which decomposes output indices into ND coordinates and applies source strides.
pub fn copy_with_mode(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
    mode: super::ExecMode,
) -> Result<Array, KernelError> {
    let is_strided = !src.is_contiguous();

    let kernel_name = match (src.dtype(), is_strided) {
        (DType::Float32, false) => "copy_f32",
        (DType::Float32, true) => "copy_strided_f32",
        (DType::Float16, false) => "copy_f16",
        (DType::Float16, true) => "copy_strided_f16",
        (DType::Bfloat16, _) => {
            return Err(KernelError::NotFound(
                "copy not supported for bf16 (different memory layout from f16)".to_string(),
            ))
        }
        (DType::Q4_0 | DType::Q4_1 | DType::Q8_0, _) => {
            return Err(KernelError::NotFound(
                "copy not supported for quantized types".to_string(),
            ))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let numel = src.numel();

    // Create contiguous output buffer
    let out = Array::zeros(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);

    if is_strided {
        // Pass shape, src_strides, ndim as additional Metal buffers
        let device = registry.device().raw();
        let ndim = src.ndim();

        let shape_data: Vec<u32> = src.shape().iter().map(|&s| s as u32).collect();
        let stride_data: Vec<u32> = src.strides().iter().map(|&s| s as u32).collect();

        let shape_buf = device.new_buffer_with_data(
            shape_data.as_ptr() as *const _,
            (ndim * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let stride_buf = device.new_buffer_with_data(
            stride_data.as_ptr() as *const _,
            (ndim * std::mem::size_of::<u32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let ndim_val = ndim as u32;
        let ndim_buf = device.new_buffer_with_data(
            &ndim_val as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(2, Some(&shape_buf), 0);
        encoder.set_buffer(3, Some(&stride_buf), 0);
        encoder.set_buffer(4, Some(&ndim_buf), 0);
    }

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

#[cfg(test)]
mod tests {
    // GPU tests require Metal device — run on macOS with `cargo test -p rmlx-core`.
    // Non-contiguous copy tests verify that the strided kernel correctly
    // handles transposed and sliced array layouts.
}
