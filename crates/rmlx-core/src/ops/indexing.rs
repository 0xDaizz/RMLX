//! Indexing operations: gather and scatter.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};

pub const INDEXING_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Gather: output[i] = src[indices[i]]
kernel void gather_f32(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = src[indices[id]];
}

// Scatter: output[indices[i]] = src[i]  (last write wins)
kernel void scatter_f32(
    device const float* src [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[indices[id]] = src[id];
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("indexing", INDEXING_SHADER_SOURCE)
}

/// Gather elements from src at given indices.
pub fn gather(
    registry: &KernelRegistry,
    src: &Array,
    indices: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let kernel_name = match src.dtype() {
        DType::Float32 => "gather_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "gather not supported for {:?}",
                src.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, src.dtype())?;
    let numel = indices.numel();
    let out = Array::zeros(registry.device().raw(), indices.shape(), src.dtype());

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(src.metal_buffer()), src.offset() as u64);
    enc.set_buffer(1, Some(indices.metal_buffer()), indices.offset() as u64);
    enc.set_buffer(2, Some(out.metal_buffer()), 0);

    let grid = metal::MTLSize::new(numel as u64, 1, 1);
    let tg = metal::MTLSize::new(
        std::cmp::min(pipeline.max_total_threads_per_threadgroup(), numel as u64),
        1,
        1,
    );
    enc.dispatch_threads(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    Ok(out)
}
