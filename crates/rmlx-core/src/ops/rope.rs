//! Rotary Position Embedding (RoPE).
//! Applies rotation to pairs of dimensions using precomputed sin/cos frequencies.

use crate::array::Array;
use crate::dtype::DType;
use crate::kernels::{KernelError, KernelRegistry};
use metal::MTLSize;

pub const ROPE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Apply RoPE to pairs of dimensions
// Input shape: [seq_len, head_dim], head_dim must be even
// cos_freqs/sin_freqs shape: [seq_len, head_dim/2]
kernel void rope_f32(
    device const float* input [[buffer(0)]],
    device const float* cos_freqs [[buffer(1)]],
    device const float* sin_freqs [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& offset [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint pair_idx = gid.x;  // index into pairs (0..head_dim/2)

    if (row >= seq_len || pair_idx >= head_dim / 2) return;

    uint input_base = row * head_dim;
    uint freq_base = (row + offset) * (head_dim / 2);

    float x0 = input[input_base + 2 * pair_idx] * scale;
    float x1 = input[input_base + 2 * pair_idx + 1] * scale;
    float cos_val = cos_freqs[freq_base + pair_idx];
    float sin_val = sin_freqs[freq_base + pair_idx];

    output[input_base + 2 * pair_idx]     = x0 * cos_val - x1 * sin_val;
    output[input_base + 2 * pair_idx + 1] = x0 * sin_val + x1 * cos_val;
}
"#;

pub fn register(registry: &KernelRegistry) -> Result<(), KernelError> {
    registry.register_jit_source("rope", ROPE_SHADER_SOURCE)
}

/// Apply Rotary Position Embedding.
/// - input: [seq_len, head_dim] (head_dim must be even)
/// - cos_freqs: [max_seq_len, head_dim/2]
/// - sin_freqs: [max_seq_len, head_dim/2]
/// - offset: position offset for incremental decoding
/// - scale: scaling factor (typically 1.0)
pub fn rope(
    registry: &KernelRegistry,
    input: &Array,
    cos_freqs: &Array,
    sin_freqs: &Array,
    offset: u32,
    scale: f32,
    queue: &metal::CommandQueue,
) -> Result<Array, KernelError> {
    assert_eq!(
        input.ndim(),
        2,
        "rope requires 2D input [seq_len, head_dim]"
    );
    assert_eq!(input.shape()[1] % 2, 0, "head_dim must be even");

    let input_contig = super::make_contiguous(input, registry, queue)?;
    let input = input_contig.as_ref().unwrap_or(input);
    let cos_contig = super::make_contiguous(cos_freqs, registry, queue)?;
    let cos_freqs = cos_contig.as_ref().unwrap_or(cos_freqs);
    let sin_contig = super::make_contiguous(sin_freqs, registry, queue)?;
    let sin_freqs = sin_contig.as_ref().unwrap_or(sin_freqs);

    let kernel_name = match input.dtype() {
        DType::Float32 => "rope_f32",
        _ => {
            return Err(KernelError::NotFound(format!(
                "rope not supported for {:?}",
                input.dtype()
            )))
        }
    };

    let pipeline = registry.get_pipeline(kernel_name, input.dtype())?;
    let seq_len = super::checked_u32(input.shape()[0], "seq_len")?;
    let head_dim = super::checked_u32(input.shape()[1], "head_dim")?;

    // Validate freq table dimensions
    let freq_rows_needed = seq_len as usize + offset as usize;
    if cos_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs rows ({}) < seq_len ({}) + offset ({})",
            cos_freqs.shape()[0],
            seq_len,
            offset
        )));
    }
    if sin_freqs.shape()[0] < freq_rows_needed {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs rows ({}) < seq_len ({}) + offset ({})",
            sin_freqs.shape()[0],
            seq_len,
            offset
        )));
    }
    let half_dim = (head_dim / 2) as usize;
    if cos_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "cos_freqs cols ({}) != head_dim/2 ({})",
            cos_freqs.shape()[1],
            half_dim
        )));
    }
    if sin_freqs.shape()[1] != half_dim {
        return Err(KernelError::InvalidShape(format!(
            "sin_freqs cols ({}) != head_dim/2 ({})",
            sin_freqs.shape()[1],
            half_dim
        )));
    }

    let out = Array::zeros(registry.device().raw(), input.shape(), input.dtype());

    // Create constant buffers
    let dev = registry.device().raw();
    let opts = metal::MTLResourceOptions::StorageModeShared;
    let seq_buf = dev.new_buffer_with_data(&seq_len as *const u32 as *const _, 4, opts);
    let dim_buf = dev.new_buffer_with_data(&head_dim as *const u32 as *const _, 4, opts);
    let off_buf = dev.new_buffer_with_data(&offset as *const u32 as *const _, 4, opts);
    let scl_buf = dev.new_buffer_with_data(&scale as *const f32 as *const _, 4, opts);

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()), input.offset() as u64);
    encoder.set_buffer(1, Some(cos_freqs.metal_buffer()), cos_freqs.offset() as u64);
    encoder.set_buffer(2, Some(sin_freqs.metal_buffer()), sin_freqs.offset() as u64);
    encoder.set_buffer(3, Some(out.metal_buffer()), 0);
    encoder.set_buffer(4, Some(&seq_buf), 0);
    encoder.set_buffer(5, Some(&dim_buf), 0);
    encoder.set_buffer(6, Some(&off_buf), 0);
    encoder.set_buffer(7, Some(&scl_buf), 0);

    let grid = MTLSize::new((head_dim / 2) as u64, seq_len as u64, 1);
    let tg = MTLSize::new(
        std::cmp::min(64, (head_dim / 2) as u64),
        std::cmp::min(16, seq_len as u64),
        1,
    );
    encoder.dispatch_threads(grid, tg);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(out)
}
