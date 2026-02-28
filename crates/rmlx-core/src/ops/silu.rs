//! SiLU (Sigmoid Linear Unit) activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

/// Metal shader source for SiLU kernels.
pub const SILU_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    output[id] = x / (1.0f + exp(-x));
}

kernel void silu_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    half x = input[id];
    output[id] = x / (half(1.0h) + exp(-x));
}

kernel void silu_bf16(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = float(input[id]);
    output[id] = bfloat(x / (1.0f + exp(-x)));
}
"#;

/// Register SiLU kernels with the registry via JIT.
pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("silu", SILU_SHADER_SOURCE)
}

/// Apply SiLU activation element-wise: silu(x) = x / (1 + exp(-x))
pub fn silu(
    registry: &KernelRegistry,
    input: &Array,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);

    let kernel_name = match input.dtype() {
        DType::Float32 => "silu_f32",
        DType::Float16 => "silu_f16",
        DType::Bfloat16 => "silu_bf16",
        DType::Q4_0 | DType::Q4_1 | DType::Q8_0 => {
            return Err(KernelError::InvalidShape(
                "silu not supported for quantized types; dequantize first".into(),
            ))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let numel = input.numel();

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
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
