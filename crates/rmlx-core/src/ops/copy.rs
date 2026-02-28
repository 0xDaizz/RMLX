//! Copy and transpose operations.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for copy kernels.
pub const COPY_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

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
"#;

/// Register copy kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("copy", COPY_SHADER_SOURCE)
}

/// Copy array contents to a new array.
pub fn copy(
    registry: &KernelRegistry,
    src: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let kernel_name = match src.dtype() {
        DType::Float32 => "copy_f32",
        DType::Float16 => "copy_f16",
        DType::Bfloat16 => "copy_f16", // bf16 uses same copy logic as f16
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let numel = src.numel();

    // Create output buffer
    let out = Array::zeros(registry.device().raw(), src.shape(), src.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    encoder.set_buffer(1, Some(out.metal_buffer()), out.offset() as u64);

    let grid_size = MTLSize::new(numel as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}
